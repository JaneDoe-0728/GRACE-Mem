"""Deterministic regex-based temporal classification with overlap handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from . import patterns
from .types import ResolutionStatus, TimeCategory


@dataclass(frozen=True)
class TemporalMatch:
    text: str
    span: tuple[int, int]
    category: TimeCategory
    status: ResolutionStatus
    priority: int


_PATTERN_SPECS: tuple[tuple[object, TimeCategory, ResolutionStatus, int], ...] = (
    (patterns.BOUNDARY_RE, TimeCategory.BOUNDARY, ResolutionStatus.RESOLVED, 100),
    (patterns.MONTH_WEEK_RANGE_RE, TimeCategory.MONTH_WEEK_RANGE, ResolutionStatus.RESOLVED, 90),
    (patterns.ABSOLUTE_DATE_RE, TimeCategory.ABSOLUTE_DATE, ResolutionStatus.RESOLVED, 80),
    (patterns.ABSOLUTE_TIME_RE, TimeCategory.ABSOLUTE_TIME, ResolutionStatus.RESOLVED, 79),
    (patterns.RELATIVE_WEEKDAY_RE, TimeCategory.RELATIVE_WEEKDAY, ResolutionStatus.RESOLVED, 70),
    (patterns.RELATIVE_DAYPART_RE, TimeCategory.RELATIVE_DAYPART, ResolutionStatus.RESOLVED, 67),
    (patterns.DAYPART_TIME_RE, TimeCategory.DAYPART_TIME, ResolutionStatus.RESOLVED, 66),
    (patterns.RELATIVE_DAY_RE, TimeCategory.RELATIVE_DAY, ResolutionStatus.RESOLVED, 65),
    (patterns.IN_LAST_RE, TimeCategory.RELATIVE_WINDOW, ResolutionStatus.RESOLVED, 60),
    (patterns.RELATIVE_HOUR_RE, TimeCategory.RELATIVE_HOUR, ResolutionStatus.RESOLVED, 58),
    (patterns.AGO_RE, TimeCategory.RELATIVE_WINDOW, ResolutionStatus.RESOLVED, 55),
    (patterns.WEEK_POINT_RE, TimeCategory.WEEK_POINT, ResolutionStatus.RESOLVED, 50),
    (patterns.WEEKEND_RE, TimeCategory.WEEKEND, ResolutionStatus.RESOLVED, 48),
    (patterns.MONTH_POINT_RE, TimeCategory.MONTH_POINT, ResolutionStatus.RESOLVED, 40),
    (patterns.SEASON_POINT_RE, TimeCategory.SEASON_POINT, ResolutionStatus.RESOLVED, 35),
    (patterns.YEAR_POINT_RE, TimeCategory.YEAR_POINT, ResolutionStatus.RESOLVED, 30),
    (patterns.FUZZY_RE, TimeCategory.UNKNOWN, ResolutionStatus.RESOLVED, 10),
)


def _iter_candidates(text: str) -> Iterable[TemporalMatch]:
    for regex, category, status, priority in _PATTERN_SPECS:
        for match in regex.finditer(text):
            yield TemporalMatch(
                text=match.group(0),
                span=(match.start(), match.end()),
                category=category,
                status=status,
                priority=priority,
            )


def classify_temporal_matches(text: str) -> list[TemporalMatch]:
    """Return non-overlapping temporal matches ordered by start position."""
    candidates = list(_iter_candidates(text))
    candidates.sort(
        key=lambda item: (
            -(item.span[1] - item.span[0]),
            -item.priority,
            item.span[0],
            item.span[1],
        )
    )

    accepted: list[TemporalMatch] = []
    occupied: list[tuple[int, int]] = []
    for candidate in candidates:
        start, end = candidate.span
        overlaps = any(not (end <= occ_start or start >= occ_end) for occ_start, occ_end in occupied)
        if overlaps:
            continue
        accepted.append(candidate)
        occupied.append(candidate.span)

    return sorted(accepted, key=lambda item: item.span[0])


def classify_single_expression(text: str) -> TemporalMatch:
    """Classify a single expression string without overlap logic."""
    matches = classify_temporal_matches(text)
    if matches:
        exact = [match for match in matches if match.span == (0, len(text))]
        if exact:
            return exact[0]
        return matches[0]
    return TemporalMatch(
        text=text,
        span=(0, len(text)),
        category=TimeCategory.UNKNOWN,
        status=ResolutionStatus.UNRESOLVED,
        priority=0,
    )
