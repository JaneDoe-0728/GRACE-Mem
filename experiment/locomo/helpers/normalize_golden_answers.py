"""Normalize LoCoMo golden answers to ISO date ranges.

Applies the project temporal parser to all QA answers, with lightweight
pre-processing for LoCoMo-specific formatting quirks (comma separators,
weekday-before-date expressions) that the main parser does not handle.

The original temporal parser (grace_mem/utils/temporal) is unchanged.

Usage
-----
    python -m experiment.locomo.helpers.normalize_golden_answers \
        [--input  experiment/locomo/data/locomo10.json] \
        [--output experiment/locomo/data/locomo10_temporal_normalized.json]

Output schema (added as ``temporal_norm`` on every QA item)
-----------------------------------------------------------
    {
        "status":           "resolved" | "partially_resolved" | "unresolved" | "not_temporal",
        "normalized_start": "YYYY-MM-DD" | null,
        "normalized_end":   "YYYY-MM-DD" | null,
        "display_value":    str | null,
        "method":           str,
    }
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from grace_mem.utils.temporal.normalizer import build_time_context
from grace_mem.utils.temporal.classifier import classify_single_expression
from grace_mem.utils.temporal.resolver import resolve_match
from grace_mem.utils.temporal.types import ResolutionStatus

# ---------------------------------------------------------------------------
# constants (local copies – do not touch the project parser)
# ---------------------------------------------------------------------------

_MONTHS: dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
    "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_WEEKDAYS: dict[str, int] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

# Neutral reference date (most LoCoMo content is 2022–2023; only matters for
# relative expressions that don't embed their own anchor date).
_DEFAULT_REF = datetime(2023, 6, 15)
_CTX = build_time_context(reference_dt=_DEFAULT_REF)

# ---------------------------------------------------------------------------
# text pre-processing
# ---------------------------------------------------------------------------

_ORDINAL_RE = re.compile(r"(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_TYPOS: dict[str, str] = {"januarty": "january", "janury": "january"}

# "The [weekday] before [date]" — full pattern with embedded anchor date
_WD_BEFORE_RE = re.compile(
    r"(?:the\s+|on\s+the\s+)?(\w+day)\s+before\s+"
    r"(?:(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})"   # "15 July 2023"
    r"|([A-Za-z]+)\s+(\d{1,2})\s+(\d{4}))",   # "July 15 2023"
    re.IGNORECASE,
)

_HAS_DATE_HINT = re.compile(
    r"\b(\d{4})\b"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"|(?:before|after|week|weekend|friday|saturday|sunday"
    r"|monday|tuesday|wednesday|thursday)",
    re.IGNORECASE,
)


def _preprocess(text: str) -> str:
    """Normalise common formatting quirks without changing semantics."""
    s = text.strip().rstrip(".")
    # Strip leading prepositions: "on 5 June, 2023" → "5 June 2023"
    s = re.sub(r"^(?:on|in)\s+", "", s, flags=re.IGNORECASE)
    # Remove ordinal suffixes: "13th" → "13"
    s = _ORDINAL_RE.sub(r"\1", s)
    # Fix known month typos
    for bad, good in _TYPOS.items():
        s = re.sub(rf"\b{bad}\b", good, s, flags=re.IGNORECASE)
    # No-space between digit and letter: "14August" → "14 August"
    s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)
    # No-space between letter and 4-digit year: "August2023" → "August 2023"
    s = re.sub(r"([A-Za-z])(\d{4})", r"\1 \2", s)
    # Comma before 4-digit year: "May, 2023" → "May 2023"
    s = re.sub(r",\s*(\d{4})", r" \1", s)
    # Period as separator: "April.2023" → "April 2023"
    s = re.sub(r"([A-Za-z])\.\s*(\d{4})", r"\1 \2", s)
    return s


# ---------------------------------------------------------------------------
# weekday-before-date (not in the main parser)
# ---------------------------------------------------------------------------

def _parse_date_parts(day: str, month: str, year: str) -> Optional[date]:
    """Extract (day, month, year) from a date written either day- or month-first.

    Both orders are present in the corpus, so which capture groups matched is
    what disambiguates them -- the numbers alone cannot, since 3/4 is valid
    either way.
    """
    m = _MONTHS.get(month.strip().lower())
    if not m:
        return None
    try:
        return date(int(year), m, int(day))
    except ValueError:
        return None


def _weekday_before(anchor: date, weekday_name: str) -> Optional[date]:
    """Resolve "the <weekday> before <date>" to an absolute date.

    Strictly before: when the anchor date is itself that weekday, the previous
    week is meant, not the anchor itself.
    """
    wd = _WEEKDAYS.get(weekday_name.lower())
    if wd is None:
        return None
    d = anchor - timedelta(days=1)
    for _ in range(7):
        if d.weekday() == wd:
            return d
        d -= timedelta(days=1)
    return None


def _try_weekday_before(raw: str) -> Optional[dict]:
    """Return a norm dict if raw matches '[weekday] before [date]', else None."""
    m = _WD_BEFORE_RE.search(raw)
    if not m:
        return None
    wd_name = m.group(1)
    if m.group(2):  # day-first: "15 July 2023"
        anchor = _parse_date_parts(m.group(2), m.group(3), m.group(4))
    else:           # month-first: "July 15 2023"
        anchor = _parse_date_parts(m.group(6), m.group(5), m.group(7))
    if not anchor:
        return None
    result = _weekday_before(anchor, wd_name)
    if not result:
        return None
    return {
        "status": "resolved",
        "normalized_start": result.isoformat(),
        "normalized_end": result.isoformat(),
        "display_value": result.isoformat(),
        "method": "weekday_before",
    }


# ---------------------------------------------------------------------------
# delegate to the project temporal parser
# ---------------------------------------------------------------------------

def _try_parser(text: str) -> Optional[dict]:
    """Try the project temporal parser; return norm dict on success."""
    m = classify_single_expression(text)
    res = resolve_match(m.text, m.span, m.category, _CTX)
    if res.status == ResolutionStatus.RESOLVED and res.start:
        return {
            "status": "resolved",
            "normalized_start": res.start.date().isoformat(),
            "normalized_end": res.end.date().isoformat() if res.end else None,
            "display_value": res.display_value,
            "method": f"parser/{m.category.value}",
        }
    return None


# ---------------------------------------------------------------------------
# main normalizer
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> dict:
    """Rewrite a gold answer's dates into a canonical form.

    Gold answers write dates as prose ("15 July 2023", "July 15 2023"), and a
    judge comparing those against a model's ISO output can mark a correct answer
    wrong on formatting alone. Normalizing both sides removes that from the
    measurement.
    """
    raw = _preprocess(text)

    # 1. Weekday-before-date (not handled by parser)
    result = _try_weekday_before(raw)
    if result:
        return result

    # 2. Project temporal parser (handles absolute dates, months, weeks, seasons …)
    result = _try_parser(raw)
    if result:
        return result

    # 3. Unresolved
    if not _HAS_DATE_HINT.search(text):
        return {"status": "not_temporal", "normalized_start": None,
                "normalized_end": None, "display_value": None, "method": "skip"}

    return {"status": "unresolved", "normalized_start": None,
            "normalized_end": None, "display_value": None, "method": "no_match"}


# ---------------------------------------------------------------------------
# file processing
# ---------------------------------------------------------------------------

def normalize_locomo_file(input_path: Path, output_path: Path) -> None:
    """Rewrite a dataset file's gold answers in place, reporting how many changed."""
    with open(input_path) as f:
        data = json.load(f)

    total = resolved = 0
    for sample in data:
        for qa in sample.get("qa", []):
            total += 1
            qa["temporal_norm"] = normalize_answer(str(qa.get("answer", "")))
            if qa["temporal_norm"]["status"] == "resolved":
                resolved += 1

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Processed {total} QA pairs → {resolved} resolved")
    print(f"Written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    _data_dir = Path(__file__).resolve().parents[1] / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  type=Path, default=_data_dir / "locomo10.json")
    parser.add_argument("--output", type=Path, default=_data_dir / "locomo10_temporal_normalized.json")
    args = parser.parse_args(argv)

    if not args.input.exists():
        sys.exit(f"Error: input not found: {args.input}")

    normalize_locomo_file(args.input, args.output)


if __name__ == "__main__":
    main()
