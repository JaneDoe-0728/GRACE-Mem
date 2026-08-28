"""Post-hoc failure analysis: why a question got the answer it got.

An accuracy number says a run went wrong; it does not say where. Retrieval
passes a candidate set through many narrowing stages -- search, filtering,
reranking, evidence selection -- and a wrong answer usually means the right
evidence was dropped at exactly one of them. This module derives enough per
stage to identify which.

The core idea is differential: `derive_drop_reasons` diffs the candidate set
between consecutive stages, so what disappeared, and where, is recovered
without every stage having to report its own losses.
`build_top_miss_snapshot` complements it with the near-misses -- the candidates
that scored well and still lost, which is where a threshold set slightly wrong
shows up.

Both benchmarks use this identically, which is what makes it common/. The
append-only writers it hands results to live in `grace_mem.runtime.analysis_log`,
because the ingestion pipeline writes those artifacts too.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from grace_mem.runtime.analysis_log import (
    append_analysis_record,
    append_pretty_block,
    timestamp_now,
)

__all__ = [
    "append_analysis_record",
    "append_pretty_block",
    "build_bridge_label",
    "build_top_miss_snapshot",
    "coerce_bool",
    "coerce_float",
    "compact_json",
    "derive_anomaly_flags",
    "derive_drop_reasons",
    "derive_failure_type",
    "extract_context_session_ids",
    "is_temporal_question",
    "read_reranker_rows",
    "render_failure_digest",
    "timestamp_now",
]




# utf-8-sig, not utf-8: these CSVs are routinely opened and re-saved in Excel,
# which prepends a BOM. Read as plain utf-8 that BOM becomes part of the first
# column's name, and every lookup against that column silently misses.
def _load_csv_rows(path: Path, *, encoding: str = "utf-8-sig") -> list[dict[str, Any]]:
    with path.open("r", encoding=encoding, newline="") as fh:
        return list(csv.DictReader(fh))


def read_reranker_rows(log_dir: str | Path, *, request_id: str) -> list[dict[str, Any]]:
    """Return this request's rows from the shared reranker score dump.

    The CSV holds every request in the run, so it is filtered rather than
    indexed. Both sides of the comparison are stringified and stripped because
    the column arrives as text while callers hold whatever type they were given.

    Returns [] when the file is absent -- reranking is optional, and its
    absence is a configuration, not a failure.
    """
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
    """Collect the highest-scoring candidates the reranker did *not* select.

    These are the informative failures. A missed item that scored near the top
    means the ranking was nearly right and a cutoff was wrong; one that scored
    far down means retrieval never surfaced it and the problem is upstream.
    `rejected_stage` records which of the two applies:

    - reranker_cutoff: passed the score threshold, lost on top-k.
    - reranker_threshold: never reached the threshold at all.

    Grouped and capped per item type so a category with many candidates cannot
    crowd out the near-misses of a sparser one.

    Args:
        limit_per_type: Misses kept per item type, best first.

    Returns:
        Snapshot records, or [] if there is no request_id or no score dump.
    """
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
            continue  # malformed row; a partial CSV write must not abort analysis
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
    """Pull the session ids out of a rendered retrieval context.

    The context handed to the generator is formatted text, so recovering which
    sessions it drew on means parsing it back out. That is what makes evidence
    coverage measurable: compare these against the gold sessions.

    Sorted numerically rather than lexically, so 10 follows 9 instead of 1 --
    these ids end up in reports read side by side across runs.
    """
    if not isinstance(text, str):
        return []
    return sorted(set(re.findall(r"\[session=(\d+),", text)), key=lambda value: int(value))


def is_temporal_question(question: str) -> bool:
    """Guess whether a question is about time, by keyword.

    A coarse substring test used only to bucket questions when reporting
    accuracy -- temporal questions fail for different reasons than factual
    ones, and mixing them hides both. It over-triggers freely ("last" matches
    "the last thing you said"), which is acceptable for a reporting split and
    would not be for anything that changed retrieval behaviour.
    """
    lowered = str(question).lower()
    tokens = (
        "when", "date", "day", "month", "year", "time",
        "before", "after", "latest", "first", "last", "earlier", "later",
    )
    return any(token in lowered for token in tokens)


def coerce_float(value: Any) -> float | None:
    """Parse a CSV field as a float, returning None when it is not one.

    None rather than 0.0: these values feed averages, and a missing score
    counted as zero would drag a mean down as though the item had scored badly
    rather than not been scored at all.
    """
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
    """Recover what each retrieval stage dropped, by diffing consecutive stages.

    The central analysis routine. Retrieval narrows a candidate set across many
    stages, and a wrong answer is nearly always the right evidence being
    discarded at one of them. Rather than requiring every stage to report its
    own losses -- which each would do differently, and new stages would forget
    -- this diffs each stage's surviving names against the previous stage's.
    Whatever vanished was dropped there.

    Because it is a diff, it is bounded by what the trace records: a stage that
    logs no names is invisible here, and its drops are attributed to the next
    stage that does.

    A stage is reported when it removed something, was skipped, or gave an
    explicit reason. Stages that changed nothing are omitted, which keeps the
    output to the points where the candidate set actually moved. Each branch is
    walked independently, since branches narrow separately before merging.

    Args:
        summary: A retrieval trace, expected to carry `branches` (stage lists
            per branch) and optionally `stop_reason`.

    Returns:
        One record per lossy stage, plus a final "merged" record if the run
        stopped for a stated reason.
    """
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
    """Flag internally inconsistent runs, whether or not the answer was right.

    Distinct from `derive_failure_type`, which classifies a known-bad answer.
    These flags catch states that should not occur at all, and each names a
    specific broken invariant:

    - zero_seed_entities: retrieval started from nothing.
    - many_entities_no_relationships: entities were found but the graph
      returned no edges between them, which usually means the sync step failed
      rather than that the entities are genuinely unconnected.
    - ingest_succeeded_but_retrieval_empty: data went in and nothing came back
      -- an indexing problem, not a retrieval one.
    - temporal_question_without_date_hits: a time question that retrieved no
      temporal evidence, so it cannot have been answered on evidence.
    - retrieval_low_confidence: below the run's own tau threshold.
    - wrong_answer_with_no_selected_evidence: wrong, and nothing was even
      offered to the generator.

    Flags are not exclusive; a badly broken run trips several. An empty list
    means nothing anomalous was detected, not that the answer was correct.

    Args:
        correctness: Judge score, or None when unjudged. The last flag is
            skipped when it is None, since it needs a known-wrong answer.
    """
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
    """Assign one failure category, testing causes from earliest to latest.

    Order is the whole design. The checks run in pipeline order -- exception,
    then empty seed, then empty evidence, then low confidence, then off-topic
    -- so the earliest thing that went wrong wins. A run whose seed was empty
    will also have no evidence, and reporting it as an evidence failure would
    point at the wrong stage.

    The final `judge_wrong_answer_despite_good_retrieval` is reached only when
    every retrieval check passed, which localises the failure to answer
    generation or to the judge itself.

    Returns:
        A single category string. Always returns one, so a correct run
        classifies as the fallback category -- read it alongside `correctness`,
        not on its own.
    """
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
    """Attribute a wrong answer to retrieval or to answer construction.

    The coarsest and most useful split, since the two call for entirely
    different fixes. Non-zero coverage means the right evidence was in fact
    retrieved and the generator still got it wrong -- an answer-construction
    failure. Zero coverage, or no evidence at all, means retrieval never
    supplied what was needed.

    Returns:
        "retrieval_failure", "answer_construction_failure", or
        "not_applicable" for correct or unjudged runs.
    """
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
    """Render one sample's ingestion and retrieval failures as a readable report.

    The end of the analysis chain: the JSONL artifacts are for querying, this
    is for reading. Ingestion is reported before retrieval because a retrieval
    failure over a corpus that never ingested properly is a symptom, and
    presenting it first sends the reader after the wrong cause.

    Args:
        ingest_records: Per-turn ingest diagnostics for this sample.
        failures: Classified retrieval failures.

    Returns:
        The digest text. Sections that have no records say so explicitly rather
        than being omitted -- absent output should not be ambiguous between
        "nothing failed" and "nothing was recorded".
    """
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
