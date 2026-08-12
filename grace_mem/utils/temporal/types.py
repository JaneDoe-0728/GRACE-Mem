"""Typed structures for deterministic temporal parsing and resolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class TimeCategory(str, Enum):
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
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"
    INVALID = "invalid"


class TimeGranularity(str, Enum):
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
    reference_dt: datetime
    reference_time_str: Optional[str] = None
    timezone: str = "Asia/Taipei"
    source: Optional[str] = None
    last_weekday_policy: str = "nearest_previous"
    daypart_anchor_times: Optional[dict[str, str]] = None


@dataclass(frozen=True)
class ResolvedTimeRange:
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
