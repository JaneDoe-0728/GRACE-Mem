from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


_JSONL_FILES = {
    "ingest_delta": "error_analysis_ingest_delta.jsonl",
    "retrieval_summary": "error_analysis_retrieval_summary.jsonl",
    "failure_verdict": "error_analysis_failure_verdicts.jsonl",
    "drop_reasons": "error_analysis_drop_reasons.jsonl",
    "top_miss": "error_analysis_top_miss_candidates.jsonl",
    "anomaly_flags": "error_analysis_anomaly_flags.jsonl",
    "evidence_bridge": "error_analysis_evidence_bridge.jsonl",
    "grep_agent": "error_analysis_grep_agent.jsonl",
}


# ---------- IO helpers ----------

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)


def _load_csv_rows(path: Path, *, encoding: str = "utf-8-sig") -> list[dict[str, Any]]:
    with path.open("r", encoding=encoding, newline="") as fh:
        return list(csv.DictReader(fh))


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


# ---------- Public API ----------

def timestamp_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_analysis_record(log_dir: str | Path, artifact: str, record: dict[str, Any]) -> Path:
    target_dir = _ensure_dir(Path(log_dir))
    filename = _JSONL_FILES[artifact]
    payload = {"logged_at": timestamp_now(), **record}
    path = target_dir / filename
    _append_jsonl_record(path, payload)
    return path


def append_pretty_block(log_dir: str | Path, filename: str, text: str) -> Path:
    target_dir = _ensure_dir(Path(log_dir))
    path = target_dir / filename
    if path.exists() and path.stat().st_size > 0:
        _append_text(path, "\n")
    _append_text(path, text.rstrip() + "\n")
    return path


def read_reranker_rows(log_dir: str | Path, *, request_id: str) -> list[dict[str, Any]]:
    path = Path(log_dir) / "reranker_scores.csv"
    if not path.exists():
        return []
    rows = _load_csv_rows(path)
    return [row for row in rows if str(row.get("request_id", "")).strip() == str(request_id).strip()]


def build_top_miss_snapshot(
    *,
    log_dir: str | Path,
    request_id: str | None,
    limit_per_type: int = 3,
) -> list[dict[str, Any]]:
    if not request_id:
        return []
    rows = read_reranker_rows(log_dir, request_id=request_id)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            selected = str(row.get("selected", "")).strip().lower() == "true"
            above_threshold = str(row.get("above_threshold", "")).strip().lower() == "true"
            score = float(row.get("score", "nan"))
            rank = int(row.get("rank", 0))
        except ValueError:
            continue
        if selected:
            continue
        item_type = str(row.get("item_type", "")).strip() or "unknown"
        grouped.setdefault(item_type, []).append(
            {
                "item_type": item_type,
                "item_id": row.get("item_id", ""),
                "name": row.get("name", ""),
                "score": score,
                "rank": rank,
                "above_threshold": above_threshold,
                "rejected_stage": "reranker_cutoff" if above_threshold else "reranker_threshold",
            }
        )
    snapshots: list[dict[str, Any]] = []
    for item_type, candidates in grouped.items():
        ordered = sorted(candidates, key=lambda item: (-item["score"], item["rank"]))
        snapshots.extend(ordered[:limit_per_type])
    return snapshots


def extract_context_session_ids(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return sorted(set(re.findall(r"\[session=(\d+),", text)), key=lambda value: int(value))


def is_temporal_question(question: str) -> bool:
    lowered = str(question).lower()
    tokens = (
        "when", "date", "day", "month", "year", "time",
        "before", "after", "latest", "first", "last", "earlier", "later",
    )
    return any(token in lowered for token in tokens)


def coerce_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def derive_drop_reasons(summary: dict[str, Any]) -> list[dict[str, Any]]:
    traces = summary.get("branches") or {}
    records: list[dict[str, Any]] = []
    for branch_name, stages in traces.items():
        prev_entities: set[str] = set()
        prev_relationships: set[str] = set()
        for stage in stages or []:
            entity_names = set(stage.get("entity_names") or [])
            relationship_names = set(stage.get("relationship_names") or [])
            removed_entities = sorted(prev_entities - entity_names)
            removed_relationships = sorted(prev_relationships - relationship_names)
            reason = stage.get("reason")
            if removed_entities or removed_relationships or stage.get("skipped") or reason:
                records.append(
                    {
                        "request_id": summary.get("request_id"),
                        "question": summary.get("question"),
                        "branch": branch_name,
                        "step": stage.get("step"),
                        "stage": stage.get("stage"),
                        "reason": reason or ("branch_skipped" if stage.get("skipped") else "filtered"),
                        "removed_entities": removed_entities,
                        "removed_relationships": removed_relationships,
                    }
                )
            prev_entities = entity_names
            prev_relationships = relationship_names
    stop_reason = summary.get("stop_reason")
    if stop_reason:
        records.append(
            {
                "request_id": summary.get("request_id"),
                "question": summary.get("question"),
                "branch": "merged",
                "step": "final",
                "stage": "stop_reason",
                "reason": stop_reason,
                "removed_entities": [],
                "removed_relationships": [],
            }
        )
    return records


def derive_anomaly_flags(*, summary: dict[str, Any], correctness: float | None) -> list[str]:
    flags: list[str] = []
    if not summary.get("pass1_entity_ids"):
        flags.append("zero_seed_entities")
    if summary.get("final_entity_count", 0) >= 5 and summary.get("final_relationship_count", 0) == 0:
        flags.append("many_entities_no_relationships")
    if summary.get("ingest_entities_added", 0) > 0 and summary.get("final_entity_count", 0) == 0:
        flags.append("ingest_succeeded_but_retrieval_empty")
    if is_temporal_question(summary.get("question", "")) and not summary.get("has_temporal_evidence", False):
        flags.append("temporal_question_without_date_hits")
    confidence = coerce_float(summary.get("conf_final"))
    if confidence is not None:
        tau = coerce_float(summary.get("tau_confidence"))
        if tau is not None and confidence < tau:
            flags.append("retrieval_low_confidence")
    if correctness is not None and correctness < 1 and summary.get("selected_evidence_count", 0) == 0:
        flags.append("wrong_answer_with_no_selected_evidence")
    return flags


def derive_failure_type(*, summary: dict[str, Any], correctness: float | None) -> str:
    stop_reason = str(summary.get("stop_reason") or "").strip()
    if summary.get("exception"):
        return "retrieval_exception"
    if stop_reason in {"no_entity_hits", "node_subgraph_empty", "intersection_empty"}:
        return "retrieval_empty_seed"
    if summary.get("selected_evidence_count", 0) == 0:
        return "retrieval_empty_evidence"
    confidence = coerce_float(summary.get("conf_final"))
    tau = coerce_float(summary.get("tau_confidence"))
    if confidence is not None and tau is not None and confidence < tau:
        return "retrieval_low_confidence"
    coverage = coerce_float(summary.get("coverage_percent")) or 0.0
    if correctness is not None and correctness < 1 and coverage == 0.0:
        return "retrieval_off_topic"
    return "judge_wrong_answer_despite_good_retrieval"


def build_bridge_label(*, summary: dict[str, Any], correctness: float | None) -> str:
    if correctness is None or correctness >= 1:
        return "not_applicable"
    if summary.get("selected_evidence_count", 0) == 0:
        return "retrieval_failure"
    coverage = coerce_float(summary.get("coverage_percent")) or 0.0
    if coverage > 0:
        return "answer_construction_failure"
    return "retrieval_failure"


def render_failure_digest(
    *,
    sample_index: int,
    ingest_records: Iterable[dict[str, Any]],
    failures: Iterable[dict[str, Any]],
) -> str:
    lines = [
        "=" * 80,
        f"sample_{sample_index} failure digest",
        f"generated_at: {timestamp_now()}",
        "",
    ]

    ingest_records = list(ingest_records)
    failures = list(failures)

    lines.append("Ingestion")
    if not ingest_records:
        lines.append("  no ingestion diagnostics recorded")
    else:
        for record in ingest_records:
            status = record.get("failure_type", "ingest_ok")
            lines.append(
                "  "
                f"session={record.get('session_id')} message={record.get('message_id')} "
                f"entities_added={record.get('entities_added', 0)} "
                f"relationships_added={record.get('relationships_added', 0)} "
                f"status={status}"
            )
    lines.append("")

    lines.append("Failed Questions")
    if not failures:
        lines.append("  no failed questions")
    else:
        for failure in failures:
            top_miss = failure.get("top_miss") or []
            miss_preview = "; ".join(
                f"{item.get('item_type')}:{item.get('name')}({item.get('score')})"
                for item in top_miss[:3]
            ) or "-"
            flags = ", ".join(failure.get("anomaly_flags") or []) or "-"
            lines.append(
                "  "
                f"q={failure.get('question')} | verdict={failure.get('failure_type')} "
                f"| correctness={failure.get('correctness')} | flags={flags}"
            )
            lines.append(
                "    "
                f"request_id={failure.get('request_id')} stop_reason={failure.get('stop_reason') or '-'} "
                f"evidence={failure.get('selected_evidence_count', 0)} top_miss={miss_preview}"
            )
    return "\n".join(lines).rstrip()
