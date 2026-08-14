"""Typed structures for deterministic temporal parsing and resolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class TimeCategory(str, Enum):
    """What kind of time expression a phrase is, which selects how it resolves.

    The distinctions are resolution strategies, not linguistic categories. An
    ABSOLUTE_DATE needs no reference time; a RELATIVE_DAY ("yesterday") needs
    one; a WEEK_POINT ("last week") resolves to a range rather than an instant;
    and a BOUNDARY ("before July") produces an open-ended constraint. Merging
    any two would mean resolving one of them by the wrong rule.

    UNKNOWN is the explicit no-match, so an unclassified phrase is visible in
    the trace rather than silently defaulting into some other category.
    """
    ABSOLUTE_DATE = "absolute_date"
    ABSOLUTE_TIME = "absolute_time"
    RELATIVE_DAY = "relative_day"
    RELATIVE_DAYPART = "relative_daypart"
    RELATIVE_WEEKDAY = "relative_weekday"
    RELATIVE_HOUR = "relative_hour"
    DAYPART_TIME = "daypart_time"
    RELATIVE_WINDOW = "relative_window"
    WEEK_POINT = "week_point"
    WEEKEND = "weekend"
    MONTH_POINT = "month_point"
    SEASON_POINT = "season_point"
    YEAR_POINT = "year_point"
    BOUNDARY = "boundary"
    MONTH_WEEK_RANGE = "month_week_range"
    UNKNOWN = "unknown"


class ResolutionStatus(str, Enum):
    """How completely a temporal expression was resolved.

    Graded rather than boolean because the middle states are actionable.
    PARTIALLY_RESOLVED (a month with no year) still narrows retrieval; AMBIGUOUS
    means several readings are equally supported and picking one would be a
    guess; INVALID means the phrase resolved to an impossible date, which is a
    parser bug rather than missing input. Collapsing these to resolved/unresolved
    loses the distinction between "cannot" and "should not".
    """
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"
    INVALID = "invalid"


class TimeGranularity(str, Enum):
    """How wide a resolved time value is.

    Carried through to the graph and consulted at retrieval: a query bounded to
    a day must not match a node spanning a year. WEEKEND is separate from WEEK
    because it is a two-day subrange, not a shorter week, and matching it as a
    week would pull in five irrelevant days.
    """
    DAY = "day"
    TIME = "time"
    WEEK = "week"
    WEEKEND = "weekend"
    MONTH = "month"
    SEASON = "season"
    YEAR = "year"
    RANGE = "range"


@dataclass(frozen=True)
class ValidationResult:
    """Whether a resolution passed its sanity checks, and why not if it failed.

    Attached to every resolution rather than raised, because an invalid one is
    data worth keeping -- the trace shows what the parser produced and which
    check rejected it, which is how parser bugs get found. `code` is stable for
    programmatic grouping; `message` is for people.
    """
    ok: bool
    code: str
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class TimeContext:
    """The reference frame relative expressions resolve against.

    Without this, "last Tuesday" has no meaning. `reference_dt` is the turn's
    own timestamp, which is why resolution happens at ingest time -- by query
    time the frame is gone and cannot be reconstructed.

    Attributes:
        last_weekday_policy: How "last Tuesday" resolves when the reference day
            is itself a Tuesday. "nearest_previous" picks the week before rather
            than the same day, which is what speakers mean.
        daypart_anchor_times: Clock times for "morning", "evening" and friends.
            Configurable because the mapping is a convention, not a fact.
    """
    reference_dt: datetime
    reference_time_str: Optional[str] = None
    timezone: str = "Asia/Taipei"
    source: Optional[str] = None
    last_weekday_policy: str = "nearest_previous"
    daypart_anchor_times: Optional[dict[str, str]] = None


@dataclass(frozen=True)
class ResolvedTimeRange:
    """One temporal expression, fully resolved, with its provenance.

    Always a range -- start and end -- even for an expression that names an
    instant, so downstream comparisons are interval overlaps in every case
    rather than branching on granularity.

    The original text and span are retained so a resolution can be traced back
    to the phrase that produced it, which is the only way to tell a
    mis-resolution from a mis-detection.

    `to_dict` emits several redundant spellings of the same values
    (original_phrase alongside original_text, reference_time alongside
    anchor_reference_time). That redundancy is deliberate: consumers written
    against earlier versions of this schema read the older names, and dropping
    them would break stored artifacts rather than only new code.
    """
    original_text: str
    span: tuple[int, int]
    category: TimeCategory
    status: ResolutionStatus
    method: str
    anchor_reference_time: Optional[str]
    granularity: Optional[TimeGranularity]
    start: Optional[datetime]
    end: Optional[datetime]
    operator: Optional[str]
    display_value: Optional[str]
    validation_result: ValidationResult

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "original_phrase": self.original_text,
            "span": list(self.span),
            "category": self.category.value,
            "status": self.status.value,
            "method": self.method,
            "anchor_reference_time": self.anchor_reference_time,
            "reference_time": self.anchor_reference_time,
            "granularity": self.granularity.value if self.granularity else None,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "normalized_time": self.start.strftime("%H:%M") if self.start and self.granularity == TimeGranularity.TIME else None,
            "normalized_start": self.start.date().isoformat() if self.start else None,
            "normalized_end": self.end.date().isoformat() if self.end else None,
            "operator": self.operator,
            "normalized_text": self.display_value,
            "display_value": self.display_value,
            "validation_result": self.validation_result.to_dict(),
        }


@dataclass(frozen=True)
class TemporalConstraint:
    """A resolved expression plus the operator relating a question to it.

    The difference from a bare `ResolvedTimeRange` is the operator: "in July",
    "before July", and "after July" share one resolved range but select disjoint
    sets of events. `anchor_resolution` holds the second range for expressions
    anchored to another ("the week before my birthday").
    """
    original_text: str
    span: tuple[int, int]
    operator: Optional[str]
    anchor_text: Optional[str]
    anchor_resolution: Optional[ResolvedTimeRange]
    resolution: ResolvedTimeRange

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "span": list(self.span),
            "operator": self.operator,
            "anchor_text": self.anchor_text,
            "anchor_resolution": self.anchor_resolution.to_dict() if self.anchor_resolution else None,
            "resolution": self.resolution.to_dict(),
        }
