import json
import subprocess
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if __package__ in (None, ""):
    repo_root = MODULE_DIR.parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiment.reproducibility import (
    activate_reproducibility,
    attach_reproducibility_metadata,
    current_reproducibility_state,
    write_reproducibility_file,
)
from experiment.run_metadata import namespace_to_dict, write_run_metadata
from experiment.locomo.utils.io import ensure_dir
from experiment.locomo.utils.log import log_event
from experiment.locomo.models import RunConfig, RunRuntime
from experiment.locomo.decision import (
    _current_context,
    _judge_dir_for_aggregate,
    _next_context,
    _select_strategy,
    _update_run_state,
    build_sample_plan,
    should_skip_refresh,
)
from experiment.locomo.aggregate import maybe_aggregate_run, maybe_upload_aggregate
from experiment.locomo.helpers.run_hooks import (
    _after_worker,
    _log_success,
    _record_sample_outputs,
    _refresh_system,
    _worker_paths_for_sample,
)
from experiment.locomo.snapshot import _snapshot_builder
from experiment.locomo.workers import run_worker


_STATELESS_RETRIEVAL_MODES = {
    "gold_summary_only",
    "gold_raw_text_only",
    "replay_summary_raw_text_from_run",
    "replay_summary_fact_from_run",
}


def _write_run_metadata(
    run_root: Path,
    output_root: Path,
    args,
    *,
    sample_ids: list[int],
    dataset_json_path: Path,
    sessions_jsonl_path: Path | None,
    selected_stages: list[str],
) -> None:
    metadata = {
        "entrypoint": "locomo.pipeline",
        "run_tag": run_root.name,
        "run_root": str(run_root.resolve()),
        "output_root": str(output_root.resolve()),
        "dataset": getattr(args, "dataset", ""),
        "retrieval_mode": getattr(args, "retrieval_mode", ""),
        "artifact_dir": str(Path(args.artifact_dir).resolve()) if getattr(args, "artifact_dir", None) else None,
        "replay_run_dir": str(Path(args.replay_run_dir).resolve()) if getattr(args, "replay_run_dir", None) else None,
        "baseline_run_dir": str(Path(args.baseline_run_dir).resolve()) if getattr(args, "baseline_run_dir", None) else None,
        "dataset_json_path": str(dataset_json_path.resolve()),
        "sessions_jsonl_path": str(sessions_jsonl_path.resolve()) if sessions_jsonl_path is not None else None,
        "sample_ids": list(sample_ids),
        "stages": list(selected_stages),
        "cli": {
            "argv": list(getattr(args, "raw_argv", [])),
            "resolved_args": namespace_to_dict(args),
        },
    }
    write_run_metadata(run_root / "run_metadata.json", attach_reproducibility_metadata(metadata))
    write_reproducibility_file(run_root)

# ---------------------------------------------------------------------------
# Runtime builder
# ---------------------------------------------------------------------------

def _build_runtime(args) -> RunRuntime:
    from experiment.locomo.helpers import (
        default_output_variant_dir,
        load_raw_samples,
        normalize_dataset_name,
        resolve_dataset_path,
    )
    from experiment.locomo.cli import parse_sample_ids, resolve_stages

    selected_stages = resolve_stages(
        getattr(args, "stages", None),
        no_judge=args.no_judge,
        artifact_dir=getattr(args, "artifact_dir", None),
    )
    if not args.sample_ids:
        selected_stage_set = set(selected_stages)
        if selected_stage_set != {"upload"}:
            raise SystemExit("--sample-ids is required (e.g. 0,2,5-7)")

    dataset = normalize_dataset_name(args.dataset)
    dataset_json_path = resolve_dataset_path(
        dataset=dataset, kind="qa_json", explicit_path=args.dataset_json
    )
    sessions_jsonl_path = resolve_dataset_path(
        dataset=dataset,
        kind="sessions_jsonl",
        explicit_path=args.sessions_jsonl,
        required=False,
    )
    sample_ids = parse_sample_ids(args.sample_ids)
    if not sample_ids:
        raise SystemExit("No valid sample ids parsed from --sample-ids")

    output_root = Path(args.out_root)
    default_output_base = Path("experiment/locomo/output")
    if output_root == default_output_base:
        ensure_dir(default_output_base)
        ensure_dir(default_output_base / "standard")
        ensure_dir(default_output_base / "plus")
        output_root = default_output_base / default_output_variant_dir(dataset)
    else:
        ensure_dir(output_root)
    run_root = ensure_dir(output_root / args.run_tag)
    config = RunConfig.from_args(
        args=args,
        dataset=dataset,
        dataset_json_path=dataset_json_path,
        sessions_jsonl_path=sessions_jsonl_path,
        run_root=run_root,
        sample_ids=sample_ids,
    )
    _write_run_metadata(
        run_root,
        output_root,
        args,
        sample_ids=sample_ids,
        dataset_json_path=dataset_json_path,
        sessions_jsonl_path=sessions_jsonl_path,
        selected_stages=selected_stages,
    )
    all_samples_plus = load_raw_samples(dataset_json_path) if dataset == "locomo-plus" else None
    return RunRuntime(
        config=config,
        run_summary_json=run_root / "correctness_summary.json",
        per_sample_stats={},
        all_samples_plus=all_samples_plus,
    )


# ---------------------------------------------------------------------------
# Orchestrator entry point
# ---------------------------------------------------------------------------

def run_orchestrator(args) -> None:
    from experiment.locomo.helpers import is_cognitive_item
    from experiment.locomo.cli import build_worker_command, resolve_stages

    runtime = _build_runtime(args)
    config = runtime.config
    strategy = _select_strategy(config.dataset)
    is_stateless_mode = getattr(args, "retrieval_mode", "") in _STATELESS_RETRIEVAL_MODES
    selected_stages = set(
        resolve_stages(
            getattr(args, "stages", None),
            no_judge=args.no_judge,
            artifact_dir=getattr(args, "artifact_dir", None),
        )
    )
    run_sample_stages = any(stage in selected_stages for stage in ("ingest", "qa_eval", "judge"))
    should_aggregate = bool({"judge", "upload"} & selected_stages)

    if run_sample_stages:
        for position, sample_index in enumerate(config.sample_ids):
            print(f"\n{'='*60}")
            print(f"=== SAMPLE {sample_index} ===")
            print(f"{'='*60}")

            sample_plan = build_sample_plan(
                dataset=config.dataset,
                sample_index=sample_index,
                worker_paths=_worker_paths_for_sample(runtime, sample_index, strategy),
                run_state=runtime.run_state,
                all_samples_plus=runtime.all_samples_plus,
                is_cognitive_item=is_cognitive_item,
            )
            cmd = build_worker_command(args=args, config=config, plan=sample_plan)

            log_event(
                "SUBPROCESS",
                "Launching worker",
                sample=sample_index,
                dataset=config.dataset,
                skip_graph_restore=sample_plan.skip_graph_restore,
                stages=sorted(selected_stages),
            )
            result = subprocess.run(cmd)
            success = result.returncode == 0
            if not success:
                log_event("ERROR", "Worker exited with non-zero status", sample=sample_index, exit_code=result.returncode)
            else:
                _record_sample_outputs(runtime, args, sample_plan.worker_paths, strategy)
            _after_worker(runtime, strategy)

            if not success:
                log_event("CONTINUE", "Skipping failed sample", sample=sample_index)
            else:
                _log_success(runtime, args, sample_plan, strategy)

            next_sample_index = config.sample_ids[position + 1] if position + 1 < len(config.sample_ids) else None
            next_skip_refresh = should_skip_refresh(
                current=_current_context(sample_plan, strategy),
                next_sample=_next_context(runtime, next_sample_index, is_cognitive_item, strategy),
                current_success=success,
            )

            if is_stateless_mode:
                log_event(
                    "SKIP REFRESH",
                    "Stateless retrieval mode does not require graph refresh",
                    sample=sample_index,
                    retrieval_mode=getattr(args, "retrieval_mode", ""),
                )
            elif next_skip_refresh:
                log_event(
                    "SKIP REFRESH",
                    "Next sample can reuse current graph state",
                    conv_id=sample_plan.conv_id,
                )
            else:
                _refresh_system(sleep_seconds=config.post_refresh_sleep)

            _update_run_state(runtime, sample_plan, success, strategy)
    else:
        log_event("STAGE", "Skipping sample workers", stages=sorted(selected_stages))

    aggregate_result = None
    if should_aggregate:
        aggregate_result = maybe_aggregate_run(
            dataset=config.dataset,
            run_root=config.run_root,
            no_judge=config.no_judge,
            judge_dir=_judge_dir_for_aggregate(runtime, strategy),
            include_adversarial=config.include_adversarial,
        )
    maybe_upload_aggregate(
        aggregate_result,
        enabled="upload" in selected_stages,
        dataset=config.dataset,
        run_root=config.run_root,
        sample_ids=config.sample_ids,
        no_judge=config.no_judge,
        include_adversarial=config.include_adversarial,
    )

    if run_sample_stages:
        print(f"\nAll {len(config.sample_ids)} samples completed.")
    else:
        print("\nNo per-sample stages were requested.")
    print(f"Results: {config.run_root}")


def dispatch_pipeline(args) -> None:
    if args.build_snapshots:
        _snapshot_builder(args)
        return

    if args.worker:
        run_worker(args)
        return

    run_orchestrator(args)


def main(argv=None) -> None:
    from experiment.locomo.cli import parse_args

    args = parse_args(argv)
    args.raw_argv = list(argv) if argv is not None else sys.argv[1:]
    cfg = activate_reproducibility(log_prefix="[locomo.pipeline] Reproducibility:")
    print(json.dumps({"event": "reproducibility", **current_reproducibility_state(cfg)}, ensure_ascii=False))
    dispatch_pipeline(args)


if __name__ == "__main__":
    main()
