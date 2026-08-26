"""Orchestrator: runs each LoCoMo sample as a separate worker subprocess.

Samples run in subprocesses rather than in-process threads, and that is the
central design decision here. Each sample builds a knowledge graph and loads
model weights; process isolation means a worker that leaks memory, corrupts its
graph, or dies outright takes nothing with it, and the orchestrator can record
the failure and continue. It also makes KG_ARTIFACTS_DIR a genuine per-sample
boundary, since it is set in the child's environment.

The cost is that everything crossing the boundary must be a file or a CLI
argument -- hence the artifact protocol in `helpers/sample_hooks.py`.

Run metadata is written before any sample executes, so an interrupted run is
still self-describing.
"""

import json
import subprocess
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if __package__ in (None, ""):
    repo_root = MODULE_DIR.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiment.common.reproducibility import (
    activate_reproducibility,
    attach_reproducibility_metadata,
    current_reproducibility_state,
    write_reproducibility_file,
)
from experiment.common.run_metadata import namespace_to_dict, write_run_metadata
from experiment.locomo.utils.io import ensure_dir
from experiment.locomo.utils.log import log_event
from experiment.locomo.models import RunConfig, SamplePlan
from experiment.locomo.analysis.aggregate import maybe_aggregate_run
from experiment.locomo.helpers.run_hooks import _refresh_system, _worker_paths_for_sample
from experiment.locomo.artifacts.snapshot import _snapshot_builder
from experiment.locomo.pipeline.worker import run_worker


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
    """Record the run's configuration before any sample executes.

    Written up front so an interrupted run is still self-describing -- results
    without the configuration that produced them cannot be interpreted later.
    """
    metadata = {
        "entrypoint": "locomo.pipeline.runner",
        "run_tag": run_root.name,
        "run_root": str(run_root.resolve()),
        "output_root": str(output_root.resolve()),
        "dataset": "locomo",
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

def _build_config(args) -> RunConfig:
    """Resolve paths and build the immutable standard LoCoMo run config."""
    from experiment.locomo.helpers.dataset import (
        resolve_dataset_path,
    )
    from experiment.locomo.cli import parse_sample_ids, resolve_stages

    selected_stages = resolve_stages(
        getattr(args, "stages", None),
        no_judge=args.no_judge,
        artifact_dir=getattr(args, "artifact_dir", None),
    )
    if not args.sample_ids:
        raise SystemExit("--sample-ids is required (e.g. 0,2,5-7)")

    dataset_json_path = resolve_dataset_path(
        kind="qa_json", explicit_path=args.dataset_json
    )
    sessions_jsonl_path = resolve_dataset_path(
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
        output_root = default_output_base / "standard"
    else:
        ensure_dir(output_root)
    run_root = ensure_dir(output_root / args.run_tag)
    config = RunConfig.from_args(
        args=args,
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
    return config


# ---------------------------------------------------------------------------
# Orchestrator entry point
# ---------------------------------------------------------------------------

def run_orchestrator(args) -> None:
    """Run every requested sample in sequence, one subprocess each.

    A failed sample is recorded and the loop continues: a sweep of ten samples
    should not be lost to one bad conversation. That does mean a run can finish
    "successfully" with fewer samples than requested, which is why the summary
    records per-sample outcomes rather than only the aggregate.
    """
    from experiment.locomo.cli import build_worker_command, resolve_stages

    config = _build_config(args)
    is_stateless_mode = getattr(args, "retrieval_mode", "") in _STATELESS_RETRIEVAL_MODES
    selected_stages = set(
        resolve_stages(
            getattr(args, "stages", None),
            no_judge=args.no_judge,
            artifact_dir=getattr(args, "artifact_dir", None),
        )
    )
    run_sample_stages = any(stage in selected_stages for stage in ("ingest", "qa_eval", "judge"))
    should_aggregate = "judge" in selected_stages

    if run_sample_stages:
        for sample_index in config.sample_ids:
            print(f"\n{'='*60}")
            print(f"=== SAMPLE {sample_index} ===")
            print(f"{'='*60}")

            sample_plan = SamplePlan(
                sample_index=sample_index,
                worker_paths=_worker_paths_for_sample(config, sample_index),
            )
            cmd = build_worker_command(args=args, config=config, plan=sample_plan)

            log_event(
                "SUBPROCESS",
                "Launching worker",
                sample=sample_index,
                dataset="locomo",
                stages=sorted(selected_stages),
            )
            result = subprocess.run(cmd)
            success = result.returncode == 0
            if not success:
                log_event("ERROR", "Worker exited with non-zero status", sample=sample_index, exit_code=result.returncode)
            if not success:
                log_event("CONTINUE", "Skipping failed sample", sample=sample_index)

            if is_stateless_mode:
                log_event(
                    "SKIP REFRESH",
                    "Stateless retrieval mode does not require graph refresh",
                    sample=sample_index,
                    retrieval_mode=getattr(args, "retrieval_mode", ""),
                )
            else:
                _refresh_system(sleep_seconds=config.post_refresh_sleep)
    else:
        log_event("STAGE", "Skipping sample workers", stages=sorted(selected_stages))

    if should_aggregate:
        maybe_aggregate_run(
            run_root=config.run_root,
            no_judge=config.no_judge,
            include_adversarial=config.include_adversarial,
        )
    if run_sample_stages:
        print(f"\nAll {len(config.sample_ids)} samples completed.")
    else:
        print("\nNo per-sample stages were requested.")
    print(f"Results: {config.run_root}")


def dispatch_pipeline(args) -> None:
    """Route to the orchestrator or to a single worker, per the parsed arguments.

    One entry point serving both roles is what lets the orchestrator spawn
    children with `python -m` on this same module.
    """
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
