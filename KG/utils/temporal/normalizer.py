"""Shared high-level temporal parsing, extraction, and rewrite entrypoints."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from .classifier import classify_single_expression, classify_temporal_matches
from .resolver import resolve_match
from .types import (
    ResolutionStatus,
    TemporalConstraint,
    TimeContext,
    TimeGranularity,
)


_TIME_REWRITE_ABLATION_ENV_VARS = (
    "KG_ABLATION_NO_TIME_REWRITE",
    "KG_ABLATION_NO_QUERY_TIME_REWRITE",  # legacy alias
)


def time_rewrite_ablation_enabled() -> bool:
    """Ablation G: query-side temporal rewrite off (ingest-side rewrite is out of scope)."""
    for name in _TIME_REWRITE_ABLATION_ENV_VARS:
        if os.getenv(name, "0").strip().lower() not in ("0", "", "false"):
            return True
    return False


def build_time_context(
    *,
    reference_dt: datetime,
    reference_time_str: Optional[str] = None,
    timezone: str = "Asia/Taipei",
    source: Optional[str] = None,
    last_weekday_policy: str = "nearest_previous",
    daypart_anchor_times: Optional[dict[str, str]] = None,
) -> TimeContext:
    return TimeContext(
        reference_dt=reference_dt,
        reference_time_str=reference_time_str,
        timezone=timezone,
        source=source,
        last_weekday_policy=last_weekday_policy,
        daypart_anchor_times=daypart_anchor_times,
    )


def parse_temporal_expressions(text: str, context: TimeContext) -> list[TemporalConstraint]:
    constraints: list[TemporalConstraint] = []
    for match in classify_temporal_matches(text):
        resolution = resolve_match(match.text, match.span, match.category, context)
        operator = resolution.operator
        anchor_text = None
        anchor_resolution = None
        if operator:
            parts = match.text.split(None, 1)
            anchor_text = parts[1] if len(parts) > 1 else None
            if anchor_text:
                anchor_match = classify_single_expression(anchor_text)
                anchor_resolution = resolve_match(anchor_match.text, (0, len(anchor_text)), anchor_match.category, context)
        constraints.append(
            TemporalConstraint(
                original_text=match.text,
                span=match.span,
                operator=operator,
                anchor_text=anchor_text,
                anchor_resolution=anchor_resolution,
                resolution=resolution,
            )
        )
    return constraints


def extract_temporal_constraints(text: str, context: TimeContext) -> list[TemporalConstraint]:
    return parse_temporal_expressions(text, context)


def _replacement_prefix(source_text: str, start: int) -> str:
    prefix = source_text[max(0, start - 7):start].lower()
    if any(token in prefix for token in ("from ", "since ", "on ", "in ", "during ", "before ", "after ")):
        return ""
    return "on "


def _replacement_prefix_for_granularity(
    granularity: TimeGranularity | None,
    source_text: str,
    start: int,
) -> str:
    prefix = source_text[max(0, start - 10):start].lower()
    if any(token in prefix for token in ("from ", "since ", "on ", "in ", "during ", "before ", "after ")):
        return ""
    if granularity is TimeGranularity.DAY:
        return "on "
    if granularity is TimeGranularity.TIME:
        return "at "
    if granularity in {TimeGranularity.WEEK, TimeGranularity.WEEKEND, TimeGranularity.SEASON, TimeGranularity.RANGE}:
        return "during "
    return "in "


def _marker_entries_for_constraint(constraint: TemporalConstraint) -> list[dict[str, str]]:
    resolution = constraint.resolution
    if resolution.status is not ResolutionStatus.RESOLVED or not resolution.validation_result.ok:
        return []
    if resolution.granularity is TimeGranularity.DAY and resolution.start:
        return [{"entity_type": "Date", "marker_type": "DATE", "entity_name": resolution.start.date().isoformat()}]
    if resolution.category.name in {"RELATIVE_DAYPART", "DAYPART_TIME"} and resolution.display_value:
        return [{"entity_type": "Timespan", "marker_type": "TIMESPAN", "entity_name": resolution.display_value}]
    if resolution.granularity is TimeGranularity.TIME and resolution.start:
        return [{"entity_type": "Time", "marker_type": "TIME", "entity_name": resolution.start.strftime("%Y-%m-%dT%H:%M")}]
    if resolution.display_value:
        return [{"entity_type": "Timespan", "marker_type": "TIMESPAN", "entity_name": resolution.display_value}]
    return []


def _marker_text_for_constraint(constraint: TemporalConstraint) -> str | None:
    entries = _marker_entries_for_constraint(constraint)
    if not entries:
        return None
    return " ".join(f'[{entry["marker_type"]}: {entry["entity_name"]}]' for entry in entries)


def _rewrite_constraint(constraint: TemporalConstraint, source_text: str) -> str | None:
    resolution = constraint.resolution
    if resolution.status is not ResolutionStatus.RESOLVED:
        return None
    if resolution.validation_result.ok is not True:
        return None

    if resolution.operator and resolution.display_value:
        return resolution.display_value

    if resolution.granularity is TimeGranularity.DAY and resolution.start:
        return f"{_replacement_prefix(source_text, constraint.span[0])}{resolution.start.date().isoformat()}"

    if resolution.category.name in {"RELATIVE_DAYPART", "DAYPART_TIME"} and resolution.display_value:
        return f"{_replacement_prefix_for_granularity(TimeGranularity.RANGE, source_text, constraint.span[0])}{resolution.display_value}"

    if resolution.granularity is TimeGranularity.TIME and resolution.start:
        return f"{_replacement_prefix_for_granularity(resolution.granularity, source_text, constraint.span[0])}{resolution.start.strftime('%H:%M')}"

    if resolution.display_value:
        return f"{_replacement_prefix_for_granularity(resolution.granularity, source_text, constraint.span[0])}{resolution.display_value}"

    return None


def extract_temporal_hints(texts: list[str], context: TimeContext) -> list[dict]:
    """Return resolved temporal expressions across all text segments.

    Each entry includes display and normalized metadata while preserving
    ``resolved_to`` as a compatibility alias.
    Only RESOLVED constraints are included. Deduplicates by original
    text (case-insensitive); first occurrence wins.
    """
    seen: dict[str, dict] = {}
    for text in texts:
        for constraint in parse_temporal_expressions(text, context):
            r = constraint.resolution
            if r.status is not ResolutionStatus.RESOLVED:
                continue
            if not r.validation_result.ok:
                continue
            key = constraint.original_text.lower()
            if key in seen:
                continue
            if r.granularity is TimeGranularity.DAY and r.start:
                resolved_to = r.start.date().isoformat()
            elif r.display_value:
                resolved_to = r.display_value
            else:
                continue
            seen[key] = {
                "original": constraint.original_text,
                "resolved_to": resolved_to,
                "display_value": r.display_value or resolved_to,
                "normalized_time": r.start.strftime("%H:%M") if r.start and (r.granularity is TimeGranularity.TIME or r.category.name in {"RELATIVE_DAYPART", "DAYPART_TIME"}) else None,
                "normalized_start": r.start.date().isoformat() if r.start else None,
                "normalized_end": r.end.date().isoformat() if r.end else None,
                "granularity": r.granularity.value if r.granularity else None,
                "category": r.category.value,
                "status": r.status.value,
                "reference_time": context.reference_time_str or context.reference_dt.isoformat(),
                "markers": _marker_entries_for_constraint(constraint),
            }
    return list(seen.values())


def format_temporal_hints_for_prompt(hints: list[dict]) -> str:
    """Format resolved temporal hints as a compact block for LLM prompt injection."""
    if not hints:
        return "(none)"
    lines = [f'  • "{h["original"]}" → {h["resolved_to"]}' for h in hints]
    return "\n".join(lines)


def augment_temporal_text(
    text: str, context: TimeContext
) -> tuple[str, list[dict], list[TemporalConstraint]]:
    """Return text with high-confidence relative phrases replaced in-place.

    Relative temporal expressions are replaced by typed marker tags for LLM
    extraction, e.g. ``[DATE: 2023-07-05]`` or ``[TIMESPAN: 2022]``.
    Returns (rewritten_text, hints, unresolved_constraints).
    hints has the same schema as extract_temporal_hints.
    """
    constraints = parse_temporal_expressions(text, context)
    augmented = text
    hints: list[dict] = []
    unresolved: list[TemporalConstraint] = []
    seen_keys: set[str] = set()

    for constraint in sorted(constraints, key=lambda c: c.span[0], reverse=True):
        r = constraint.resolution
        if r.status is not ResolutionStatus.RESOLVED or not r.validation_result.ok:
            if r.status is not ResolutionStatus.RESOLVED:
                unresolved.append(constraint)
            continue

        replacement = _marker_text_for_constraint(constraint)
        if replacement is None:
            continue

        if r.granularity is TimeGranularity.DAY and r.start:
            resolved_to = r.start.date().isoformat()
        elif r.display_value:
            resolved_to = r.display_value
        elif r.start and r.end:
            resolved_to = f"{r.start.date().isoformat()} to {r.end.date().isoformat()}"
        else:
            continue

        start, end = constraint.span
        augmented = augmented[:start] + replacement + augmented[end:]

        key = constraint.original_text.lower()
        if key not in seen_keys:
            seen_keys.add(key)
            hints.append({
                "original": constraint.original_text,
                "resolved_to": resolved_to,
                "display_value": r.display_value or resolved_to,
                "normalized_time": r.start.strftime("%H:%M") if r.start and (r.granularity is TimeGranularity.TIME or r.category.name in {"RELATIVE_DAYPART", "DAYPART_TIME"}) else None,
                "normalized_start": r.start.date().isoformat() if r.start else None,
                "normalized_end": r.end.date().isoformat() if r.end else None,
                "granularity": r.granularity.value if r.granularity else None,
                "category": r.category.value,
                "status": r.status.value,
                "reference_time": context.reference_time_str or context.reference_dt.isoformat(),
                "markers": _marker_entries_for_constraint(constraint),
            })

    return augmented, hints, unresolved


def rewrite_temporal_text(text: str, context: TimeContext) -> tuple[str, dict]:
    constraints = parse_temporal_expressions(text, context)
    rewritten_text = text
    unresolved_constraints = []
    for constraint in sorted(constraints, key=lambda item: item.span[0], reverse=True):
        replacement = _rewrite_constraint(constraint, text)
        if replacement is None:
            if constraint.resolution.status is not ResolutionStatus.RESOLVED:
                unresolved_constraints.append(constraint)
            continue
        start, end = constraint.span
        rewritten_text = rewritten_text[:start] + replacement + rewritten_text[end:]

    metadata = {
        "constraints": [constraint.to_dict() for constraint in constraints],
        "reference_time": context.reference_dt.isoformat(),
        "reference_time_str": context.reference_time_str,
        "expressions_count": len(constraints),
        "unresolved_constraints": [c.to_dict() for c in unresolved_constraints],
    }
    return rewritten_text, metadata
