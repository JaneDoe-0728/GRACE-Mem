"""
Compatibility wrapper for temporal query rewriting.

This module preserves the public entrypoints used across the repo while
delegating high-confidence temporal parsing and rewrite to the shared
deterministic core under ``grace_mem.temporal``.
"""

from __future__ import annotations

import re
from datetime import datetime

import dateparser

from grace_mem.temporal import build_time_context, rewrite_temporal_text


def parse_query_time(query_time_str: str) -> datetime | None:
    """Parse supported project timestamp formats into a datetime object."""
    if not query_time_str:
        return None

    try:
        cleaned = re.sub(r"\s*\([^)]+\)\s*", " ", query_time_str).strip()
        return datetime.strptime(cleaned, "%Y/%m/%d %H:%M")
    except Exception:
        pass

    cleaned = query_time_str.strip()
    extra_formats = [
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %b, %Y",
        "%I:%M %p on %d %B %Y",
        "%I:%M %p on %d %b %Y",
    ]
    for fmt in extra_formats:
        try:
            return datetime.strptime(cleaned, fmt)
        except Exception:
            continue

    try:
        dt = dateparser.parse(cleaned)
        if dt:
            return dt
    except Exception:
        pass

    print(f"[QueryTimeParser] Failed to parse query_time '{query_time_str}'")
    return None




def detect_and_parse_time_expressions(
    query: str,
    query_time: str | None = None,
    rewrite_query: bool = True,
) -> tuple[str, dict]:
    """Detect supported high-confidence English time expressions and optionally rewrite them."""
    if not query_time:
        return query, {
            "detected_expressions": [],
            "reference_time": None,
            "warning": "No query_time provided - cannot parse relative time expressions",
            "constraints": [],
            "expressions_count": 0,
        }

    reference_dt = parse_query_time(query_time)
    if not reference_dt:
        return query, {
            "detected_expressions": [],
            "reference_time": None,
            "error": f"Failed to parse query_time: {query_time}",
            "constraints": [],
            "expressions_count": 0,
        }

    context = build_time_context(
        reference_dt=reference_dt,
        reference_time_str=query_time,
        source="query_time_parser",
    )
    rewritten_query, metadata = rewrite_temporal_text(query, context)
    constraints = metadata.get("constraints", [])

    detected_expressions = []
    for constraint in constraints:
        resolution = constraint.get("resolution", {})
        detected_expressions.append(
            {
                "original": constraint.get("original_text", ""),
                "parsed_datetime": resolution.get("start"),
                "parsed_end_datetime": resolution.get("end"),
                "absolute_date": resolution.get("normalized_text"),
                "position": tuple(constraint.get("span", [0, 0])),
                "status": resolution.get("status"),
                "confidence": resolution.get("confidence"),
                "granularity": resolution.get("granularity"),
                "operator": constraint.get("operator"),
                "validation_result": resolution.get("validation_result"),
            }
        )

    return (rewritten_query if rewrite_query else query), {
        "detected_expressions": detected_expressions,
        "reference_time": metadata.get("reference_time"),
        "reference_time_str": query_time,
        "expressions_count": metadata.get("expressions_count", len(detected_expressions)),
        "constraints": constraints,
    }


if __name__ == "__main__":
    examples = [
        ("Which book did I finish a week ago?", "2023/02/07 (Tue) 09:09"),
        ("What happened before April 10, 2023?", "2023/03/09 (Thu) 15:47"),
        ("What are my plans for next Monday?", "2023/03/09 (Thu) 15:47"),
        ("What happened in the last 2 weeks?", "2023/03/09 (Thu) 15:47"),
        ("Who did I meet yesterday?", "8:18 pm on 6 July, 2023"),
    ]
    for query, query_time in examples:
        rewritten, info = detect_and_parse_time_expressions(query, query_time=query_time, rewrite_query=True)
        print(query)
        print(rewritten)
        print(info)
