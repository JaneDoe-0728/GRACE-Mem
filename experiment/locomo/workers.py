import os
import shutil
import json
from pathlib import Path

from experiment.locomo.utils.io import (
    backup_artifacts_and_logs,
    ensure_dir,
    load_csv_rows,
    load_jsonl_records,
    token_usage_log_path,
    write_empty_eval_csv,
    write_eval_csv,
    write_stats_json,
)
from experiment.locomo.utils.log import log_event
from experiment.locomo.utils.error_analysis import (
    append_analysis_record,
    append_pretty_block,
    build_bridge_label,
    build_top_miss_snapshot,
    coerce_float,
    derive_anomaly_flags,
    derive_failure_type,
    render_failure_digest,
)
from experiment.locomo.snapshot import (
    _resolve_conv_id_and_sessions,
    load_snapshot_files_only,
    restore_graph,
    snapshot_dir,
    snapshot_exists,
)
from experiment.locomo.stage_adapter import (
    build_eval_rows,
    configure_retriever,
    run_ingest_stage_for_locomo,
    run_ingest_stage_for_records,
    run_judge_stage,
    skipped_judge_stats,
)
from experiment.locomo.helpers.sample_hooks import (
    artifact_dir_for_sample,
    ensure_worker_repo_path,
    export_graph_to_artifacts,
    invoke_snapshot_builder,
    reload_mgr_state_from_artifacts,
    restore_artifacts_from_dir,
    restore_graph_from_artifact_dir,
    validate_and_export_graph,
)
from experiment.locomo.utils.graph import ARTIFACTS_SRC, GRAPH_EXPORT_FILE, validate_graph_export

try:
    from experiment.experiment_config import RERANKER_PARAMS, RETRIEVAL_PARAMS as _RETRIEVAL_PARAMS
except Exception:
    RERANKER_PARAMS = {
        "use_reranker": True,
        "reranker_threshold": -3.0,
        "reranker_topk": 3,
    }
    _RETRIEVAL_PARAMS = {}

# Merge both param dicts so retriever_initialized log reflects the full config.
# RETRIEVAL_PARAMS keys match RetrieverConfig field names exactly.
# RERANKER_PARAMS takes precedence where keys overlap.
RERANKER_PARAMS = {**_RETRIEVAL_PARAMS, **RERANKER_PARAMS}

try:
    from experiment.experiment_config import INGEST_PARAMS as _INGEST_PARAMS
except Exception:
    _INGEST_PARAMS = {}

# Map experiment_config.py INGEST_PARAMS keys to IngestorConfig field names so
# that the ingestor_initialized log reflects the actual values used at runtime.
_INGESTOR_CONFIG = {
    "ingest_mode": _INGEST_PARAMS.get("ingest_mode", "turn_pairs"),
    "summary_context_prev_k_default": _INGEST_PARAMS.get("prev_k", 2),
    "similar_entity_top_k": _INGEST_PARAMS.get("entity_sim_topk", 3),
    "entity_sim_threshold": _INGEST_PARAMS.get("entity_sim_threshold", 0.6),
}


def _configure_sample_pretty_trace_log(*, run_root: Path, sample_index: int) -> Path:
    sample_log_dir = ensure_dir(run_root / f"sample_{sample_index}" / "logs")
    os.environ["KG_TRACE_PRETTY_LOG_DIR"] = str(sample_log_dir)
    return sample_log_dir


def _selected_stages(args) -> set[str]:
    from experiment.locomo.cli import resolve_stages

    return set(
        resolve_stages(
            getattr(args, "stages", None),
            no_judge=args.no_judge,
            artifact_dir=getattr(args, "artifact_dir", None),
        )
    )


def _resolve_existing_artifact_dir(args, *, run_root: Path) -> Path | None:
    explicit_artifact_dir = artifact_dir_for_sample(args)
    if explicit_artifact_dir is not None:
        return explicit_artifact_dir

    sample_artifact_dir = run_root / f"sample_{args.sample_index}" / "artifacts"
    if sample_artifact_dir.exists():
        return sample_artifact_dir
    return None


def _require_existing_file(path: Path, *, stage: str, flag_hint: str) -> None:
    if path.exists():
        return
    raise FileNotFoundError(
        f"{stage} stage requires an existing file at {path}. "
        f"Run the prerequisite stage first or provide reusable artifacts via {flag_hint}."
    )


def _refresh_sample_outputs(
    *,
    sample_dir: Path,
    eval_csv: Path,
    judge_csv: Path,
    no_judge: bool,
) -> None:
    also_copy = [eval_csv]
    if not no_judge:
        also_copy.append(judge_csv)
    backup_artifacts_and_logs(
        sample_dir,
        also_copy=also_copy,
        include_artifacts=False,
    )


def _parse_json_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _emit_error_analysis_bundle(
    *,
    sample_index: int,
    sample_dir: Path,
    eval_csv: Path,
    judge_csv: Path,
    no_judge: bool,
) -> None:
    log_dir = ensure_dir(sample_dir / "logs")
    ingest_records: list[dict[str, object]] = []
    ingest_path = log_dir / "error_analysis_ingest_delta.jsonl"
    if ingest_path.exists():
        ingest_records = load_jsonl_records(ingest_path)

    if no_judge or not eval_csv.exists() or not judge_csv.exists():
        digest = render_failure_digest(
            sample_index=sample_index,
            ingest_records=ingest_records,
            failures=[],
        )
        append_pretty_block(log_dir, "error_analysis_failure_digest.log", digest)
        return

    eval_rows = load_csv_rows(eval_csv)
    judge_rows = {str(row.get("question", "")).strip(): row for row in load_csv_rows(judge_csv)}
    ingest_entities_added = sum(int(record.get("entities_added", 0) or 0) for record in ingest_records)
    failed_cases: list[dict[str, object]] = []

    for eval_row in eval_rows:
        question = str(eval_row.get("question", "")).strip()
        if not question:
            continue
        judge_row = judge_rows.get(question, {})
        correctness = coerce_float(judge_row.get("correctness"))
        summary = {
            "question": question,
            "request_id": eval_row.get("retrieval_request_id", ""),
            "stop_reason": eval_row.get("retrieval_stop_reason", ""),
            "conf_final": eval_row.get("retrieval_confidence"),
            "tau_confidence": eval_row.get("retrieval_tau"),
            "selected_evidence_count": int(coerce_float(eval_row.get("selected_evidence_count")) or 0),
            "final_entity_count": len(_parse_json_list(eval_row.get("final_entity_names"))),
            "final_relationship_count": len(_parse_json_list(eval_row.get("final_relationship_names"))),
            "pass1_entity_ids": _parse_json_list(eval_row.get("pass1_entity_ids")),
            "has_temporal_evidence": "[t:" in str(eval_row.get("retrieved_context", "")),
            "coverage_percent": judge_row.get("coverage_percent", eval_row.get("coverage_percent")),
            "ingest_entities_added": ingest_entities_added,
            "selected_evidence_preview": _parse_json_list(eval_row.get("selected_evidence_preview")),
            "gold_answer": eval_row.get("gold_answer", ""),
            "model_answer": eval_row.get("model_answer", ""),
        }
        top_miss = build_top_miss_snapshot(log_dir=log_dir, request_id=str(summary["request_id"]))
        anomaly_flags = derive_anomaly_flags(summary=summary, correctness=correctness)
        if anomaly_flags:
            append_analysis_record(
                log_dir,
                "anomaly_flags",
                {
                    "sample_index": sample_index,
                    "question": question,
                    "request_id": summary["request_id"],
                    "correctness": correctness,
                    "flags": anomaly_flags,
                },
            )

        if correctness is None or correctness >= 1:
            continue

        failure_type = derive_failure_type(summary=summary, correctness=correctness)
        bridge_label = build_bridge_label(summary=summary, correctness=correctness)
        verdict_record = {
            "scope": "question",
            "sample_index": sample_index,
            "question": question,
            "request_id": summary["request_id"],
            "failure_type": failure_type,
            "correctness": correctness,
            "stop_reason": summary["stop_reason"],
            "selected_evidence_count": summary["selected_evidence_count"],
            "top_miss_count": len(top_miss),
            "gold_answer": summary["gold_answer"],
            "model_answer": summary["model_answer"],
        }
        append_analysis_record(log_dir, "failure_verdict", verdict_record)
        append_analysis_record(
            log_dir,
            "top_miss",
            {
                "sample_index": sample_index,
                "question": question,
                "request_id": summary["request_id"],
                "candidates": top_miss,
            },
        )
        append_analysis_record(
            log_dir,
            "evidence_bridge",
            {
                "sample_index": sample_index,
                "question": question,
                "request_id": summary["request_id"],
                "correctness": correctness,
                "bridge_label": bridge_label,
                "failure_type": failure_type,
                "selected_evidence_count": summary["selected_evidence_count"],
                "selected_evidence_preview": summary["selected_evidence_preview"],
            },
        )
        failed_cases.append(
            {
                **verdict_record,
                "anomaly_flags": anomaly_flags,
                "top_miss": top_miss,
            }
        )

    digest = render_failure_digest(
        sample_index=sample_index,
        ingest_records=ingest_records,
        failures=failed_cases,
    )
    append_pretty_block(log_dir, "error_analysis_failure_digest.log", digest)


def _load_replay_question_context(sample_run_dir: Path) -> dict[str, dict[str, object]]:
    retrieval_summary_path = sample_run_dir / "logs" / "error_analysis_retrieval_summary.jsonl"
    if not retrieval_summary_path.exists():
        raise FileNotFoundError(f"Replay retrieval summary not found: {retrieval_summary_path}")

    records = load_jsonl_records(retrieval_summary_path)
    out: dict[str, dict[str, object]] = {}
    for record in records:
        question = str(record.get("question", "")).strip()
        if question:
            out[question] = record
    if not out:
        raise RuntimeError(f"No replayable question records found in: {retrieval_summary_path}")
    return out


def _load_entity_meta_by_name(sample_run_dir: Path) -> dict[str, dict[str, object]]:
    entity_meta_path = sample_run_dir / "artifacts" / "entities_meta.jsonl"
    if not entity_meta_path.exists():
        return {}
    out: dict[str, dict[str, object]] = {}
    for record in load_jsonl_records(entity_meta_path):
        name = str(record.get("name", "")).strip()
        if name and name not in out:
            out[name] = record
    return out


def _load_relationship_meta_by_label(sample_run_dir: Path) -> dict[str, dict[str, object]]:
    relationship_meta_path = sample_run_dir / "artifacts" / "relationships_meta.jsonl"
    if not relationship_meta_path.exists():
        return {}
    out: dict[str, dict[str, object]] = {}
    for record in load_jsonl_records(relationship_meta_path):
        src_name = str(record.get("source_entity", "")).strip()
        tgt_name = str(record.get("target_entity", "")).strip()
        description = str(record.get("description", "")).strip()
        if not src_name or not tgt_name:
            continue
        label = f"{src_name} -> {tgt_name}" if not description else f"{src_name} -> {tgt_name} | {description}"
        if label not in out:
            out[label] = record
    return out


def _load_session_summaries_from_artifacts(sample_run_dir: Path) -> dict[str, str]:
    """Load session summaries from a baseline run's summaries_meta.jsonl.

    Expects records with ``session_id`` (e.g. "3__1") and ``summary_text``.
    Returns {"session_1_summary": "<text>", ...} matching the format expected by
    qa_eval._gold_session_summaries.
    """
    summaries_path = sample_run_dir / "artifacts" / "summaries_meta.jsonl"
    if not summaries_path.exists():
        raise FileNotFoundError(f"Baseline summaries not found: {summaries_path}")
    out: dict[str, str] = {}
    for record in load_jsonl_records(summaries_path):
        session_id = str(record.get("session_id", "")).strip()
        text = str(record.get("summary_text", "")).strip()
        if not session_id or not text:
            continue
        # session_id format: "{sample_index}__{session_num}" → key "session_{session_num}_summary"
        parts = session_id.split("__", 1)
        session_num = parts[-1]
        key = f"session_{session_num}_summary"
        if key not in out:
            out[key] = text
    return out


def _run_locomo_gold_summary_only(args) -> None:
    """Gold-summary-only ablation: skip ingest/graph, answer from session_summary in the dataset JSON."""
    import json as _json
    import pandas as pd

    from KG.llm import token_tracker
    from experiment.locomo.stages import judge, qa_eval
    from experiment.locomo.helpers import normalize_dataset_name, resolve_dataset_path

    dataset = normalize_dataset_name(args.dataset)
    dataset_json = resolve_dataset_path(dataset=dataset, kind="qa_json", explicit_path=args.dataset_json)
    sample_index = args.sample_index
    eval_csv = Path(args.eval_csv)
    judge_csv = Path(args.judge_csv)
    run_root = Path(args.run_root)
    sample_dir = ensure_dir(run_root / f"sample_{sample_index}")
    token_log_path = token_usage_log_path(run_root, sample_index)
    _configure_sample_pretty_trace_log(run_root=run_root, sample_index=sample_index)
    selected_stages = _selected_stages(args)
    run_qa = "qa_eval" in selected_stages
    run_judge = "judge" in selected_stages and not args.no_judge

    raw_samples = _json.loads(Path(dataset_json).read_text())
    qa_eval.retrieval_mode = "gold_summary_only"
    baseline_run_dir = getattr(args, "baseline_run_dir", None)
    if baseline_run_dir:
        sample_run_dir = Path(baseline_run_dir) / f"sample_{sample_index}"
        qa_eval._gold_session_summaries = _load_session_summaries_from_artifacts(sample_run_dir)
        log_event("INFO", "Loaded gold session summaries from baseline artifacts", baseline=baseline_run_dir, sample=sample_index, count=len(qa_eval._gold_session_summaries))
    else:
        qa_eval._gold_session_summaries = raw_samples[sample_index].get("session_summary", {})
    if not qa_eval._gold_session_summaries:
        log_event("WARN", "No session_summary found for gold_summary_only mode", sample=sample_index)

    log_event("1/3", "Ingest skipped (gold_summary_only)", sample=sample_index)
    if run_qa:
        log_event("2/3", "Eval (gold_summary_only)", sample=sample_index)
        token_tracker.set_context(dataset=f"{dataset}:{sample_index}", stage="qa", log_path=token_log_path)

        qa_items = qa_eval.load_questions(str(dataset_json), sample_index=sample_index, include_adversarial=args.adv)
        if not qa_items:
            log_event("2/3", "Eval skipped (no questions after adversarial filter)", sample=sample_index)
            write_empty_eval_csv(pandas_module=pd, eval_csv=eval_csv)
            if run_judge:
                write_stats_json(args.stats_json, skipped_judge_stats(exclude_adversarial=not args.adv))
            log_event("DONE", "Worker finished", sample=sample_index)
            return

        rows = build_eval_rows(qa_eval_module=qa_eval, qa_items=qa_items, simplify_gold_evidence=True)
        write_eval_csv(pandas_module=pd, eval_csv=eval_csv, rows=rows)
    else:
        log_event("2/3", "Eval skipped (stage selection)", sample=sample_index)

    stats: dict = {}
    if args.no_judge:
        log_event("3/3", "Judge skipped (--no-judge)", sample=sample_index)
    elif not run_judge:
        log_event("3/3", "Judge skipped (stage selection)", sample=sample_index)
    else:
        _require_existing_file(eval_csv, stage="judge", flag_hint="--stage qa_eval")
        log_event("3/3", "Judge", sample=sample_index)
        token_tracker.set_context(dataset=f"{dataset}:{sample_index}", stage="judge", log_path=token_log_path)
        stats = run_judge_stage(
            judge_module=judge,
            input_csv=eval_csv,
            output_csv=judge_csv,
            sample_index=sample_index,
            dataset_json=dataset_json,
            dataset=dataset,
            exclude_adversarial=not args.adv,
        )
        write_stats_json(args.stats_json, stats or {})

    _refresh_sample_outputs(sample_dir=sample_dir, eval_csv=eval_csv, judge_csv=judge_csv, no_judge=not run_judge)
    _emit_error_analysis_bundle(
        sample_index=sample_index,
        sample_dir=sample_dir,
        eval_csv=eval_csv,
        judge_csv=judge_csv,
        no_judge=not run_judge,
    )
    log_event("DONE", "Worker finished", sample=sample_index)


def _run_locomo_gold_raw_text_only(args) -> None:
    """Gold-raw-text-only ablation: skip ingest/graph, answer from raw conversation turns in the dataset JSON."""
    import json as _json
    import pandas as pd

    from KG.llm import token_tracker
    from experiment.locomo.stages import judge, qa_eval
    from experiment.locomo.helpers import normalize_dataset_name, resolve_dataset_path

    dataset = normalize_dataset_name(args.dataset)
    dataset_json = resolve_dataset_path(dataset=dataset, kind="qa_json", explicit_path=args.dataset_json)
    sample_index = args.sample_index
    eval_csv = Path(args.eval_csv)
    judge_csv = Path(args.judge_csv)
    run_root = Path(args.run_root)
    sample_dir = ensure_dir(run_root / f"sample_{sample_index}")
    token_log_path = token_usage_log_path(run_root, sample_index)
    _configure_sample_pretty_trace_log(run_root=run_root, sample_index=sample_index)
    selected_stages = _selected_stages(args)
    run_qa = "qa_eval" in selected_stages
    run_judge = "judge" in selected_stages and not args.no_judge

    raw_samples = _json.loads(Path(dataset_json).read_text())
    qa_eval.retrieval_mode = "gold_raw_text_only"
    qa_eval._gold_session_raw_texts = raw_samples[sample_index].get("conversation", {})
    if not qa_eval._gold_session_raw_texts:
        log_event("WARN", "No conversation found in dataset for gold_raw_text_only mode", sample=sample_index)

    log_event("1/3", "Ingest skipped (gold_raw_text_only)", sample=sample_index)
    if run_qa:
        log_event("2/3", "Eval (gold_raw_text_only)", sample=sample_index)
        token_tracker.set_context(dataset=f"{dataset}:{sample_index}", stage="qa", log_path=token_log_path)

        qa_items = qa_eval.load_questions(str(dataset_json), sample_index=sample_index, include_adversarial=args.adv)
        if not qa_items:
            log_event("2/3", "Eval skipped (no questions after adversarial filter)", sample=sample_index)
            write_empty_eval_csv(pandas_module=pd, eval_csv=eval_csv)
            if run_judge:
                write_stats_json(args.stats_json, skipped_judge_stats(exclude_adversarial=not args.adv))
            log_event("DONE", "Worker finished", sample=sample_index)
            return

        rows = build_eval_rows(qa_eval_module=qa_eval, qa_items=qa_items, simplify_gold_evidence=True)
        write_eval_csv(pandas_module=pd, eval_csv=eval_csv, rows=rows)
    else:
        log_event("2/3", "Eval skipped (stage selection)", sample=sample_index)

    stats: dict = {}
    if args.no_judge:
        log_event("3/3", "Judge skipped (--no-judge)", sample=sample_index)
    elif not run_judge:
        log_event("3/3", "Judge skipped (stage selection)", sample=sample_index)
    else:
        _require_existing_file(eval_csv, stage="judge", flag_hint="--stage qa_eval")
        log_event("3/3", "Judge", sample=sample_index)
        token_tracker.set_context(dataset=f"{dataset}:{sample_index}", stage="judge", log_path=token_log_path)
        stats = run_judge_stage(
            judge_module=judge,
            input_csv=eval_csv,
            output_csv=judge_csv,
            sample_index=sample_index,
            dataset_json=dataset_json,
            dataset=dataset,
            exclude_adversarial=not args.adv,
        )
        write_stats_json(args.stats_json, stats or {})

    _refresh_sample_outputs(sample_dir=sample_dir, eval_csv=eval_csv, judge_csv=judge_csv, no_judge=not run_judge)
    _emit_error_analysis_bundle(
        sample_index=sample_index,
        sample_dir=sample_dir,
        eval_csv=eval_csv,
        judge_csv=judge_csv,
        no_judge=not run_judge,
    )
    log_event("DONE", "Worker finished", sample=sample_index)


def _run_locomo_replay_summary_raw_text_from_run(args) -> None:
    """Replay selected evidence ids from a prior run, but replace summaries with raw session text."""
    import json as _json
    import pandas as pd

    from KG.llm import token_tracker
    from experiment.locomo.stages import judge, qa_eval
    from experiment.locomo.helpers import normalize_dataset_name, resolve_dataset_path

    if not args.replay_run_dir:
        raise ValueError("--replay-run-dir is required for replay_summary_raw_text_from_run mode")

    dataset = normalize_dataset_name(args.dataset)
    dataset_json = resolve_dataset_path(dataset=dataset, kind="qa_json", explicit_path=args.dataset_json)
    sample_index = args.sample_index
    eval_csv = Path(args.eval_csv)
    judge_csv = Path(args.judge_csv)
    run_root = Path(args.run_root)
    sample_dir = ensure_dir(run_root / f"sample_{sample_index}")
    token_log_path = token_usage_log_path(run_root, sample_index)
    _configure_sample_pretty_trace_log(run_root=run_root, sample_index=sample_index)
    selected_stages = _selected_stages(args)
    run_qa = "qa_eval" in selected_stages
    run_judge = "judge" in selected_stages and not args.no_judge

    raw_samples = _json.loads(Path(dataset_json).read_text())
    qa_eval.retrieval_mode = "replay_summary_raw_text_from_run"
    qa_eval._gold_session_raw_texts = raw_samples[sample_index].get("conversation", {})
    qa_eval._replay_question_context = _load_replay_question_context(
        Path(args.replay_run_dir) / f"sample_{sample_index}"
    )
    qa_eval._replay_entity_meta_by_name = _load_entity_meta_by_name(
        Path(args.replay_run_dir) / f"sample_{sample_index}"
    )
    qa_eval._replay_relationship_meta_by_label = _load_relationship_meta_by_label(
        Path(args.replay_run_dir) / f"sample_{sample_index}"
    )

    if not qa_eval._gold_session_raw_texts:
        log_event("WARN", "No conversation found in dataset for replay_summary_raw_text_from_run mode", sample=sample_index)

    log_event("1/3", "Ingest skipped (replay_summary_raw_text_from_run)", sample=sample_index)
    if run_qa:
        log_event("2/3", "Eval (replay_summary_raw_text_from_run)", sample=sample_index)
        token_tracker.set_context(dataset=f"{dataset}:{sample_index}", stage="qa", log_path=token_log_path)

        qa_items = qa_eval.load_questions(str(dataset_json), sample_index=sample_index, include_adversarial=args.adv)
        if not qa_items:
            log_event("2/3", "Eval skipped (no questions after adversarial filter)", sample=sample_index)
            write_empty_eval_csv(pandas_module=pd, eval_csv=eval_csv)
            if run_judge:
                write_stats_json(args.stats_json, skipped_judge_stats(exclude_adversarial=not args.adv))
            log_event("DONE", "Worker finished", sample=sample_index)
            return

        rows = build_eval_rows(qa_eval_module=qa_eval, qa_items=qa_items, simplify_gold_evidence=True)
        write_eval_csv(pandas_module=pd, eval_csv=eval_csv, rows=rows)
    else:
        log_event("2/3", "Eval skipped (stage selection)", sample=sample_index)

    stats: dict = {}
    if args.no_judge:
        log_event("3/3", "Judge skipped (--no-judge)", sample=sample_index)
    elif not run_judge:
        log_event("3/3", "Judge skipped (stage selection)", sample=sample_index)
    else:
        _require_existing_file(eval_csv, stage="judge", flag_hint="--stage qa_eval")
        log_event("3/3", "Judge", sample=sample_index)
        token_tracker.set_context(dataset=f"{dataset}:{sample_index}", stage="judge", log_path=token_log_path)
        stats = run_judge_stage(
            judge_module=judge,
            input_csv=eval_csv,
            output_csv=judge_csv,
            sample_index=sample_index,
            dataset_json=dataset_json,
            dataset=dataset,
            exclude_adversarial=not args.adv,
        )
        write_stats_json(args.stats_json, stats or {})

    _refresh_sample_outputs(sample_dir=sample_dir, eval_csv=eval_csv, judge_csv=judge_csv, no_judge=not run_judge)
    _emit_error_analysis_bundle(
        sample_index=sample_index,
        sample_dir=sample_dir,
        eval_csv=eval_csv,
        judge_csv=judge_csv,
        no_judge=not run_judge,
    )
    log_event("DONE", "Worker finished", sample=sample_index)


def _run_locomo_replay_summary_fact_from_run(args) -> None:
    """Replay selected evidence ids from a prior run, extract facts from raw session text."""
    import json as _json
    import pandas as pd

    from KG.llm import token_tracker
    from experiment.locomo.stages import judge, qa_eval
    from experiment.locomo.helpers import normalize_dataset_name, resolve_dataset_path

    if not args.replay_run_dir:
        raise ValueError("--replay-run-dir is required for replay_summary_fact_from_run mode")

    dataset = normalize_dataset_name(args.dataset)
    dataset_json = resolve_dataset_path(dataset=dataset, kind="qa_json", explicit_path=args.dataset_json)
    sample_index = args.sample_index
    eval_csv = Path(args.eval_csv)
    judge_csv = Path(args.judge_csv)
    run_root = Path(args.run_root)
    token_log_path = token_usage_log_path(run_root, sample_index)
    _configure_sample_pretty_trace_log(run_root=run_root, sample_index=sample_index)

    raw_samples = _json.loads(Path(dataset_json).read_text())
    qa_eval.retrieval_mode = "replay_summary_fact_from_run"
    qa_eval._gold_session_raw_texts = raw_samples[sample_index].get("conversation", {})
    qa_eval._replay_question_context = _load_replay_question_context(
        Path(args.replay_run_dir) / f"sample_{sample_index}"
    )
    qa_eval._replay_entity_meta_by_name = _load_entity_meta_by_name(
        Path(args.replay_run_dir) / f"sample_{sample_index}"
    )
    qa_eval._replay_relationship_meta_by_label = _load_relationship_meta_by_label(
        Path(args.replay_run_dir) / f"sample_{sample_index}"
    )

    if not qa_eval._gold_session_raw_texts:
        log_event("WARN", "No conversation found in dataset for replay_summary_fact_from_run mode", sample=sample_index)

    log_event("1/3", "Ingest skipped (replay_summary_fact_from_run)", sample=sample_index)
    log_event("2/3", "Eval (replay_summary_fact_from_run)", sample=sample_index)
    token_tracker.set_context(dataset=f"{dataset}:{sample_index}", stage="qa", log_path=token_log_path)

    qa_items = qa_eval.load_questions(str(dataset_json), sample_index=sample_index, include_adversarial=args.adv)
    if not qa_items:
        log_event("2/3", "Eval skipped (no questions after adversarial filter)", sample=sample_index)
        write_empty_eval_csv(pandas_module=pd, eval_csv=eval_csv)
        if not args.no_judge:
            write_stats_json(args.stats_json, skipped_judge_stats(exclude_adversarial=not args.adv))
        log_event("DONE", "Worker finished", sample=sample_index)
        return

    rows = build_eval_rows(qa_eval_module=qa_eval, qa_items=qa_items, simplify_gold_evidence=True)
    write_eval_csv(pandas_module=pd, eval_csv=eval_csv, rows=rows)

    stats: dict = {}
    if args.no_judge:
        log_event("3/3", "Judge skipped (--no-judge)", sample=sample_index)
    else:
        log_event("3/3", "Judge", sample=sample_index)
        token_tracker.set_context(dataset=f"{dataset}:{sample_index}", stage="judge", log_path=token_log_path)
        stats = run_judge_stage(
            judge_module=judge,
            input_csv=eval_csv,
            output_csv=judge_csv,
            sample_index=sample_index,
            dataset_json=dataset_json,
            dataset=dataset,
            exclude_adversarial=not args.adv,
        )
        write_stats_json(args.stats_json, stats or {})

    sample_dir = ensure_dir(run_root / f"sample_{sample_index}")
    _refresh_sample_outputs(sample_dir=sample_dir, eval_csv=eval_csv, judge_csv=judge_csv, no_judge=args.no_judge)
    _emit_error_analysis_bundle(
        sample_index=sample_index,
        sample_dir=sample_dir,
        eval_csv=eval_csv,
        judge_csv=judge_csv,
        no_judge=args.no_judge,
    )
    log_event("DONE", "Worker finished", sample=sample_index)


def run_locomo_worker(args) -> None:
    """Original locomo worker: ingest all sessions -> eval -> judge."""
    if getattr(args, "retrieval_mode", "") == "gold_summary_only":
        _run_locomo_gold_summary_only(args)
        return
    if getattr(args, "retrieval_mode", "") == "gold_raw_text_only":
        _run_locomo_gold_raw_text_only(args)
        return
    if getattr(args, "retrieval_mode", "") == "replay_summary_raw_text_from_run":
        _run_locomo_replay_summary_raw_text_from_run(args)
        return
    if getattr(args, "retrieval_mode", "") == "replay_summary_fact_from_run":
        _run_locomo_replay_summary_fact_from_run(args)
        return

    import pandas as pd

    from KG.llm import token_tracker
    from experiment.locomo.stages import judge
    from experiment.locomo.helpers import normalize_dataset_name, resolve_dataset_path

    dataset = normalize_dataset_name(args.dataset)
    dataset_json = resolve_dataset_path(
        dataset=dataset,
        kind="qa_json",
        explicit_path=args.dataset_json,
    )
    sample_index = args.sample_index
    eval_csv = Path(args.eval_csv)
    judge_csv = Path(args.judge_csv)
    run_root = Path(args.run_root)
    sample_dir = ensure_dir(run_root / f"sample_{sample_index}")
    token_log_path = token_usage_log_path(run_root, sample_index)
    _configure_sample_pretty_trace_log(run_root=run_root, sample_index=sample_index)
    selected_stages = _selected_stages(args)
    run_ingest = "ingest" in selected_stages
    run_qa = "qa_eval" in selected_stages
    run_judge = "judge" in selected_stages and not args.no_judge
    artifact_dir = None if run_ingest else _resolve_existing_artifact_dir(args, run_root=run_root)

    if run_qa and artifact_dir is None and not run_ingest:
        raise FileNotFoundError(
            f"qa_eval stage requires reusable artifacts for sample {sample_index}. "
            "Run --stage ingest first or provide --artifact-dir <previous_run>."
        )

    if run_ingest or run_qa:
        if artifact_dir is not None:
            restore_artifacts_from_dir(artifact_dir)

        from KG.storage import MGR
        if artifact_dir is not None:
            reload_mgr_state_from_artifacts(MGR)

        from KG.pipeline.factory import build_pipeline
        from experiment.locomo.stages import ingest, qa_eval

        pipeline = build_pipeline(retriever_config=RERANKER_PARAMS, ingestor_config=_INGESTOR_CONFIG)
        retriever = pipeline["retriever"]
        ingestor = pipeline["ingestor"]
        graph = pipeline["graph"]
        qa_eval.retriever = retriever

        configure_retriever(retriever, adaptive=args.adaptive, tau=args.tau)

        if run_ingest:
            log_event("1/3", "Ingest", sample=sample_index)
            token_tracker.set_context(
                dataset=f"{dataset}:{sample_index}",
                stage="ingest",
                log_path=token_log_path,
            )
            run_ingest_stage_for_locomo(
                ingest_module=ingest,
                ingestor=ingestor,
                dataset=dataset,
                dataset_json=dataset_json,
                sessions_jsonl=args.sessions_jsonl,
                sample_index=sample_index,
                prev_k=args.prev_k,
                entity_sim_topk=args.entity_sim_topk,
                entity_sim_threshold=args.entity_sim_threshold,
                chunk_turns=args.chunk_turns,
            )
            MGR.flush_persist()
        else:
            log_event("1/3", "Ingest skipped (stage selection)", sample=sample_index)

        if run_qa:
            log_event("2/3", "Eval", sample=sample_index)
            token_tracker.set_context(
                dataset=f"{dataset}:{sample_index}",
                stage="qa",
                log_path=token_log_path,
            )
            if artifact_dir is not None:
                restore_graph_from_artifact_dir(graph, artifact_dir)
            try:
                validate_and_export_graph(graph, sample_index=sample_index)
                backup_artifacts_and_logs(
                    sample_dir,
                    also_copy=(),
                    include_artifacts=True,
                )
                log_event("ARTIFACT", "Saved post-ingest snapshot", path=sample_dir / "artifacts")

                MGR.initialize()
                qa_items = qa_eval.load_questions(
                    str(dataset_json),
                    sample_index=sample_index,
                    include_adversarial=args.adv,
                )
                if not qa_items:
                    log_event("2/3", "Eval skipped (no questions after adversarial filter)", sample=sample_index)
                    write_empty_eval_csv(pandas_module=pd, eval_csv=eval_csv)
                    stats = skipped_judge_stats(exclude_adversarial=not args.adv)
                    if run_judge:
                        write_stats_json(args.stats_json, stats)
                    log_event("DONE", "Worker finished", sample=sample_index)
                    return
                rows = build_eval_rows(
                    qa_eval_module=qa_eval,
                    qa_items=qa_items,
                    simplify_gold_evidence=True,
                )
                write_eval_csv(pandas_module=pd, eval_csv=eval_csv, rows=rows)
            finally:
                try:
                    export_graph_to_artifacts(graph, sample_index=sample_index)
                except Exception as exc:
                    log_event("ARTIFACT][WARN", "Graph export failed", error=exc)
                graph.close()
        else:
            try:
                validate_and_export_graph(graph, sample_index=sample_index)
                backup_artifacts_and_logs(
                    sample_dir,
                    also_copy=(),
                    include_artifacts=True,
                )
                log_event("ARTIFACT", "Saved post-ingest snapshot", path=sample_dir / "artifacts")
            finally:
                try:
                    export_graph_to_artifacts(graph, sample_index=sample_index)
                except Exception as exc:
                    log_event("ARTIFACT][WARN", "Graph export failed", error=exc)
                graph.close()
    else:
        log_event("1/3", "Ingest skipped (stage selection)", sample=sample_index)
        log_event("2/3", "Eval skipped (stage selection)", sample=sample_index)

    stats: dict = {}
    if args.no_judge:
        log_event("3/3", "Judge skipped (--no-judge)", sample=sample_index)
    elif not run_judge:
        log_event("3/3", "Judge skipped (stage selection)", sample=sample_index)
    else:
        _require_existing_file(eval_csv, stage="judge", flag_hint="--stage qa_eval")
        log_event("3/3", "Judge", sample=sample_index)
        token_tracker.set_context(
            dataset=f"{dataset}:{sample_index}",
            stage="judge",
            log_path=token_log_path,
        )
        stats = run_judge_stage(
            judge_module=judge,
            input_csv=eval_csv,
            output_csv=judge_csv,
            sample_index=sample_index,
            dataset_json=dataset_json,
            dataset=dataset,
            exclude_adversarial=not args.adv,
        )
        write_stats_json(args.stats_json, stats or {})

    _refresh_sample_outputs(
        sample_dir=sample_dir,
        eval_csv=eval_csv,
        judge_csv=judge_csv,
        no_judge=not run_judge,
    )
    _emit_error_analysis_bundle(
        sample_index=sample_index,
        sample_dir=sample_dir,
        eval_csv=eval_csv,
        judge_csv=judge_csv,
        no_judge=not run_judge,
    )
    log_event("DONE", "Worker finished", sample=sample_index)


def _run_locomo_plus_gold_summary_only(args) -> None:
    """Gold-summary-only ablation for locomo-plus: look up session_summary from the locomo source JSON."""
    import json as _json
    import pandas as pd

    from experiment.locomo.helpers import (
        is_adversarial_item,
        load_raw_samples,
        normalize_qa_item,
        resolve_dataset_path,
    )
    from KG.llm import token_tracker
    from experiment.locomo.stages import judge, qa_eval

    dataset_json = Path(args.dataset_json)
    sample_index = args.sample_index
    eval_csv = Path(args.eval_csv)
    judge_csv = Path(args.judge_csv)
    run_root = Path(args.run_root)

    samples = load_raw_samples(dataset_json)
    if sample_index >= len(samples):
        raise RuntimeError(f"sample_index={sample_index} out of range (dataset has {len(samples)} items)")
    qa_item_raw = samples[sample_index]
    if not args.adv and is_adversarial_item(qa_item_raw):
        log_event("SKIP", "Sample is adversarial (--adv not provided)", sample=sample_index)
        write_empty_eval_csv(pandas_module=pd, eval_csv=eval_csv)
        if not args.no_judge:
            write_stats_json(args.stats_json, skipped_judge_stats(exclude_adversarial=True))
        log_event("DONE", "Worker finished (locomo-plus gold_summary_only)", sample=sample_index)
        return

    conv_id = qa_item_raw.get("conversation_id", "")
    token_log_path = token_usage_log_path(run_root, sample_index)
    _configure_sample_pretty_trace_log(run_root=run_root, sample_index=sample_index)

    baseline_run_dir = getattr(args, "baseline_run_dir", None)
    if baseline_run_dir:
        sample_run_dir = Path(baseline_run_dir) / f"sample_{sample_index}"
        session_summaries = _load_session_summaries_from_artifacts(sample_run_dir)
        log_event("INFO", "Loaded gold session summaries from baseline artifacts", baseline=baseline_run_dir, sample=sample_index, count=len(session_summaries))
    else:
        # Load session summaries from the locomo source JSON by matching conversation_id.
        source_json = resolve_dataset_path(dataset="locomo", kind="qa_json", explicit_path=args.source_json)
        with open(source_json) as _f:
            source_samples = _json.load(_f)
        source_sample = next((s for s in source_samples if s.get("sample_id") == conv_id), None)
        session_summaries = source_sample.get("session_summary", {}) if source_sample else {}
    if not session_summaries:
        log_event("WARN", "No session_summary found for gold_summary_only mode", conv=conv_id, sample=sample_index)

    qa_eval.retrieval_mode = "gold_summary_only"
    qa_eval._gold_session_summaries = session_summaries

    log_event("1/3", "Ingest skipped (gold_summary_only)", sample=sample_index)
    log_event("2/3", "Eval (gold_summary_only)", sample=sample_index)
    token_tracker.set_context(dataset=conv_id, stage="qa", log_path=token_log_path)

    qa_item = normalize_qa_item(qa_item_raw)
    rows = build_eval_rows(qa_eval_module=qa_eval, qa_items=[qa_item], simplify_gold_evidence=True)
    if not rows:
        raise ValueError(f"sample_index={sample_index}: QA item has no question")
    write_eval_csv(pandas_module=pd, eval_csv=eval_csv, rows=rows)

    stats: dict = {}
    if args.no_judge:
        log_event("3/3", "Judge skipped (--no-judge)")
    else:
        log_event("3/3", "Judge", sample=sample_index)
        token_tracker.set_context(dataset=conv_id, stage="judge", log_path=token_log_path)
        stats = run_judge_stage(
            judge_module=judge,
            input_csv=eval_csv,
            output_csv=judge_csv,
            sample_index=sample_index,
            dataset_json=dataset_json,
            dataset="locomo-plus",
            exclude_adversarial=not args.adv,
        )
        write_stats_json(args.stats_json, stats or {})

    sample_dir = ensure_dir(run_root / f"sample_{sample_index}")
    _refresh_sample_outputs(sample_dir=sample_dir, eval_csv=eval_csv, judge_csv=judge_csv, no_judge=args.no_judge)
    _emit_error_analysis_bundle(
        sample_index=sample_index,
        sample_dir=sample_dir,
        eval_csv=eval_csv,
        judge_csv=judge_csv,
        no_judge=args.no_judge,
    )
    log_event("DONE", "Worker finished (locomo-plus gold_summary_only)", sample=sample_index)


def _run_locomo_plus_gold_raw_text_only(args) -> None:
    """Gold-raw-text-only ablation for locomo-plus: look up raw conversation turns from the locomo source JSON."""
    import json as _json
    import pandas as pd

    from experiment.locomo.helpers import (
        is_adversarial_item,
        load_raw_samples,
        normalize_qa_item,
        resolve_dataset_path,
    )
    from KG.llm import token_tracker
    from experiment.locomo.stages import judge, qa_eval

    dataset_json = Path(args.dataset_json)
    sample_index = args.sample_index
    eval_csv = Path(args.eval_csv)
    judge_csv = Path(args.judge_csv)
    run_root = Path(args.run_root)

    samples = load_raw_samples(dataset_json)
    if sample_index >= len(samples):
        raise RuntimeError(f"sample_index={sample_index} out of range (dataset has {len(samples)} items)")
    qa_item_raw = samples[sample_index]
    if not args.adv and is_adversarial_item(qa_item_raw):
        log_event("SKIP", "Sample is adversarial (--adv not provided)", sample=sample_index)
        write_empty_eval_csv(pandas_module=pd, eval_csv=eval_csv)
        if not args.no_judge:
            write_stats_json(args.stats_json, skipped_judge_stats(exclude_adversarial=True))
        log_event("DONE", "Worker finished (locomo-plus gold_raw_text_only)", sample=sample_index)
        return

    conv_id = qa_item_raw.get("conversation_id", "")
    token_log_path = token_usage_log_path(run_root, sample_index)
    _configure_sample_pretty_trace_log(run_root=run_root, sample_index=sample_index)

    # Load raw conversation from the locomo source JSON by matching conversation_id.
    source_json = resolve_dataset_path(dataset="locomo", kind="qa_json", explicit_path=args.source_json)
    with open(source_json) as _f:
        source_samples = _json.load(_f)
    source_sample = next((s for s in source_samples if s.get("sample_id") == conv_id), None)
    raw_texts = source_sample.get("conversation", {}) if source_sample else {}
    if not raw_texts:
        log_event("WARN", "No conversation found for gold_raw_text_only mode", conv=conv_id, sample=sample_index)

    qa_eval.retrieval_mode = "gold_raw_text_only"
    qa_eval._gold_session_raw_texts = raw_texts

    log_event("1/3", "Ingest skipped (gold_raw_text_only)", sample=sample_index)
    log_event("2/3", "Eval (gold_raw_text_only)", sample=sample_index)
    token_tracker.set_context(dataset=conv_id, stage="qa", log_path=token_log_path)

    qa_item = normalize_qa_item(qa_item_raw)
    rows = build_eval_rows(qa_eval_module=qa_eval, qa_items=[qa_item], simplify_gold_evidence=True)
    if not rows:
        raise ValueError(f"sample_index={sample_index}: QA item has no question")
    write_eval_csv(pandas_module=pd, eval_csv=eval_csv, rows=rows)

    stats: dict = {}
    if args.no_judge:
        log_event("3/3", "Judge skipped (--no-judge)")
    else:
        log_event("3/3", "Judge", sample=sample_index)
        token_tracker.set_context(dataset=conv_id, stage="judge", log_path=token_log_path)
        stats = run_judge_stage(
            judge_module=judge,
            input_csv=eval_csv,
            output_csv=judge_csv,
            sample_index=sample_index,
            dataset_json=dataset_json,
            dataset="locomo-plus",
            exclude_adversarial=not args.adv,
        )
        write_stats_json(args.stats_json, stats or {})

    sample_dir = ensure_dir(run_root / f"sample_{sample_index}")
    _refresh_sample_outputs(sample_dir=sample_dir, eval_csv=eval_csv, judge_csv=judge_csv, no_judge=args.no_judge)
    _emit_error_analysis_bundle(
        sample_index=sample_index,
        sample_dir=sample_dir,
        eval_csv=eval_csv,
        judge_csv=judge_csv,
        no_judge=args.no_judge,
    )
    log_event("DONE", "Worker finished (locomo-plus gold_raw_text_only)", sample=sample_index)


def run_locomo_plus_worker(args) -> None:
    """locomo-plus worker with snapshot restore, optional ingest, eval, and judge."""
    if getattr(args, "retrieval_mode", "") == "gold_summary_only":
        _run_locomo_plus_gold_summary_only(args)
        return
    if getattr(args, "retrieval_mode", "") == "gold_raw_text_only":
        _run_locomo_plus_gold_raw_text_only(args)
        return

    import pandas as pd

    from experiment.locomo.helpers import (
        extract_injected_session_record,
        is_adversarial_item,
        load_raw_samples,
        normalize_qa_item,
        resolve_dataset_path,
    )

    dataset_json = Path(args.dataset_json)
    sample_index = args.sample_index
    eval_csv = Path(args.eval_csv)
    judge_csv = Path(args.judge_csv)
    run_root = Path(args.run_root)
    sample_dir = ensure_dir(run_root / f"sample_{sample_index}")
    selected_stages = _selected_stages(args)
    run_ingest = "ingest" in selected_stages
    run_qa = "qa_eval" in selected_stages
    run_judge = "judge" in selected_stages and not args.no_judge
    artifact_dir = None if run_ingest else _resolve_existing_artifact_dir(args, run_root=run_root)

    samples = load_raw_samples(dataset_json)
    if sample_index >= len(samples):
        raise RuntimeError(
            f"sample_index={sample_index} out of range (dataset has {len(samples)} items)"
        )
    qa_item_raw = samples[sample_index]
    if not args.adv and is_adversarial_item(qa_item_raw):
        log_event("SKIP", "Sample is adversarial (--adv not provided)", sample=sample_index)
        write_empty_eval_csv(pandas_module=pd, eval_csv=eval_csv)
        if not args.no_judge:
            write_stats_json(args.stats_json, skipped_judge_stats(exclude_adversarial=True))
        log_event("DONE", "Worker finished (locomo-plus)", sample=sample_index)
        return

    source_json = resolve_dataset_path(
        dataset="locomo",
        kind="qa_json",
        explicit_path=args.source_json,
    )

    conv_id, injected_session_id, source_session_records, is_cognitive = (
        _resolve_conv_id_and_sessions(sample_index, qa_item_raw, source_json)
    )
    max_session_id = max(r["session_id"] for r in source_session_records)

    from KG.llm import token_tracker

    token_log_path = token_usage_log_path(run_root, sample_index)
    _configure_sample_pretty_trace_log(run_root=run_root, sample_index=sample_index)
    token_tracker.set_context(
        dataset=conv_id,
        stage="ingest",
        log_path=token_log_path,
    )

    pre_ingest_session_id = injected_session_id - 1 if is_cognitive else max_session_id

    if artifact_dir is not None:
        restore_artifacts_from_dir(artifact_dir)
        log_event("ARTIFACT", "Loading artifacts", sample=sample_index, path=artifact_dir)
    elif not args.skip_graph_restore:
        if pre_ingest_session_id > 0 and not snapshot_exists(run_root, conv_id, pre_ingest_session_id):
            log_event("SNAP", "Snapshot missing, building now", conv=conv_id, session=pre_ingest_session_id)
            invoke_snapshot_builder(
                args=args,
                conv_id=conv_id,
                max_session_id=max_session_id,
                run_root=run_root,
            )
            if not snapshot_exists(run_root, conv_id, pre_ingest_session_id):
                raise RuntimeError(
                    f"Snapshot still missing after builder for conv={conv_id} session={pre_ingest_session_id}"
                )

        if pre_ingest_session_id > 0:
            log_event("SNAP", "Loading snapshot into artifacts", conv=conv_id, session=pre_ingest_session_id)
            load_snapshot_files_only(run_root, conv_id, pre_ingest_session_id)
        else:
            if ARTIFACTS_SRC.exists():
                shutil.rmtree(ARTIFACTS_SRC)
            ARTIFACTS_SRC.mkdir(parents=True, exist_ok=True)
    else:
        log_event("SKIP", "Graph restore skipped (non-Cognitive, same conv)", conv_id=conv_id)

    from KG.storage import MGR
    if artifact_dir is not None:
        reload_mgr_state_from_artifacts(MGR)

    from KG.pipeline.factory import build_pipeline
    from experiment.locomo.stages import ingest, judge, qa_eval

    pipeline = build_pipeline(retriever_config=RERANKER_PARAMS, ingestor_config=_INGESTOR_CONFIG)
    ingestor = pipeline["ingestor"]
    qa_eval.retriever = pipeline["retriever"]
    graph = pipeline["graph"]

    if run_qa and artifact_dir is None and not run_ingest:
        raise FileNotFoundError(
            f"qa_eval stage requires reusable artifacts for sample {sample_index}. "
            "Run --stage ingest first or provide --artifact-dir <previous_run>."
        )

    try:
        if artifact_dir is not None:
            restore_graph_from_artifact_dir(graph, artifact_dir)
        elif not args.skip_graph_restore:
            if pre_ingest_session_id > 0:
                snap_dir = snapshot_dir(run_root, conv_id, pre_ingest_session_id)
                validate_graph_export(snap_dir / GRAPH_EXPORT_FILE)
                restore_graph(run_root, conv_id, pre_ingest_session_id, graph)
            else:
                graph.clear_all()
                graph.init_schema()

        if run_ingest and is_cognitive:
            input_prompt = str(qa_item_raw.get("input_prompt", "")).strip()
            if not input_prompt:
                raise ValueError(
                    f"sample_index={sample_index}: cognitive item is missing 'input_prompt'"
                )
            injected_record = extract_injected_session_record(
                conv_id, input_prompt, injected_session_id, source_session_records
            )
            log_event(
                "COG", "Ingesting injected session",
                session=injected_session_id,
                turns=len(injected_record["dialogue"]),
                conv=conv_id,
            )
            run_ingest_stage_for_records(
                ingest_module=ingest,
                ingestor=ingestor,
                records=[injected_record],
                conv_id=conv_id,
                prev_k=args.prev_k,
                entity_sim_topk=args.entity_sim_topk,
                entity_sim_threshold=args.entity_sim_threshold,
                chunk_turns=args.chunk_turns,
            )
            MGR.flush_persist()
        elif run_ingest:
            log_event("1/3", "Ingest complete (no injected session needed)", sample=sample_index)
        else:
            log_event("1/3", "Ingest skipped (stage selection)", sample=sample_index)

        if run_qa:
            log_event("2/3", "Eval", sample=sample_index)
            token_tracker.set_context(
                dataset=conv_id,
                stage="qa",
                log_path=token_log_path,
            )
            MGR.initialize()
            qa_item = normalize_qa_item(qa_item_raw)
            rows = build_eval_rows(
                qa_eval_module=qa_eval,
                qa_items=[qa_item],
                simplify_gold_evidence=True,
            )
            if not rows:
                raise ValueError(f"sample_index={sample_index}: QA item has no question")
            write_eval_csv(pandas_module=pd, eval_csv=eval_csv, rows=rows)
        else:
            log_event("2/3", "Eval skipped (stage selection)", sample=sample_index)

        backup_artifacts_and_logs(
            sample_dir,
            also_copy=(),
            include_artifacts=True,
        )
    finally:
        try:
            export_graph_to_artifacts(graph, sample_index=sample_index)
        except Exception as exc:
            log_event("ARTIFACT][WARN", "Graph export failed", error=exc)
        graph.close()

    stats: dict = {}
    if args.no_judge:
        log_event("3/3", "Judge skipped (--no-judge)")
    elif not run_judge:
        log_event("3/3", "Judge skipped (stage selection)", sample=sample_index)
    else:
        _require_existing_file(eval_csv, stage="judge", flag_hint="--stage qa_eval")
        log_event("3/3", "Judge", sample=sample_index)
        token_tracker.set_context(
            dataset=conv_id,
            stage="judge",
            log_path=token_log_path,
        )
        stats = run_judge_stage(
            judge_module=judge,
            input_csv=eval_csv,
            output_csv=judge_csv,
            sample_index=sample_index,
            dataset_json=dataset_json,
            dataset="locomo-plus",
            exclude_adversarial=not args.adv,
        )
        write_stats_json(args.stats_json, stats or {})

    _refresh_sample_outputs(
        sample_dir=sample_dir,
        eval_csv=eval_csv,
        judge_csv=judge_csv,
        no_judge=not run_judge,
    )
    _emit_error_analysis_bundle(
        sample_index=sample_index,
        sample_dir=sample_dir,
        eval_csv=eval_csv,
        judge_csv=judge_csv,
        no_judge=not run_judge,
    )
    log_event("DONE", "Worker finished (locomo-plus)", sample=sample_index)


def run_worker(args) -> None:
    ensure_worker_repo_path()

    from experiment.locomo.helpers import normalize_dataset_name

    dataset = normalize_dataset_name(args.dataset)
    if dataset == "locomo":
        run_locomo_worker(args)
        return
    run_locomo_plus_worker(args)
