"""Rewriting relative time expressions in the question before retrieval.

"last Tuesday" cannot be matched against stored dates; it has to be resolved
against the question's own reference time first. This runs at query time, which
is the mirror of what ingestion does per turn.

Separate from `grace_mem.temporal` on purpose: that package resolves time
expressions in general, while this decides whether a *question* should be
rewritten at all -- an ablation switch, a guard against rewriting when no query
time is known, and the logging that records which questions were touched.
"""


from grace_mem.runtime.logger_config import make_module_jlog, setup_logger
from grace_mem.temporal import (
    build_time_context,
    rewrite_temporal_text,
    time_rewrite_ablation_enabled,
)
from grace_mem.temporal.query_time_parser import parse_query_time

_jlog = make_module_jlog(name="grace_mem.Retriever", filename="kg_retriever.jsonl")
logger = setup_logger("grace_mem.Retriever")

def maybe_rewrite_retrieval_question(
    question: str,
    query_time: str | None,
    request_id: str | None,
) -> str:
    """Step 0b: rewrite relative temporal expressions for retrieval only."""
    if time_rewrite_ablation_enabled():
        _jlog(
            "query_temporal_rewrite_skipped",
            request_id,
            step="0b",
            reason="ablation_no_time_rewrite",
        )
        return question

    if not query_time:
        _jlog(
            "query_temporal_rewrite_skipped",
            request_id,
            step="0b",
            reason="no_query_time",
        )
        return question

    reference_dt = parse_query_time(query_time)
    if reference_dt is None:
        _jlog(
            "query_temporal_rewrite_failed",
            request_id,
            step="0b",
            reason="parse_query_time_failed",
            query_time=query_time,
        )
        return question

    context = build_time_context(
        reference_dt=reference_dt,
        reference_time_str=query_time,
        source="retriever",
    )
    rewritten_question, temporal_meta = rewrite_temporal_text(question, context)
    constraints = temporal_meta.get("constraints", [])
    expressions_count = temporal_meta.get("expressions_count", 0)
    if rewritten_question != question:
        _jlog(
            "query_temporal_rewrite",
            request_id,
            step="0b",
            original=question,
            rewritten=rewritten_question,
            reference_time=temporal_meta.get("reference_time"),
            expressions_count=expressions_count,
            constraints=[
                {
                    "original_text": c.get("original_text"),
                    "operator": c.get("operator"),
                    "status": (c.get("resolution") or {}).get("status"),
                    "confidence": (c.get("resolution") or {}).get("confidence"),
                    "granularity": (c.get("resolution") or {}).get("granularity"),
                    "start": (c.get("resolution") or {}).get("start"),
                    "end": (c.get("resolution") or {}).get("end"),
                    "normalized_text": (c.get("resolution") or {}).get("normalized_text"),
                }
                for c in constraints
            ],
        )
    else:
        _jlog(
            "query_temporal_rewrite_no_change",
            request_id,
            step="0b",
            original=question,
            reference_time=temporal_meta.get("reference_time"),
            expressions_count=expressions_count,
            constraints=[
                {
                    "original_text": c.get("original_text"),
                    "status": (c.get("resolution") or {}).get("status"),
                    "confidence": (c.get("resolution") or {}).get("confidence"),
                }
                for c in constraints
            ],
        )
    return rewritten_question
