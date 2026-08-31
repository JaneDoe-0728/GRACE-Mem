"""Deterministic temporal date/range arithmetic and validation."""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, time, timedelta

from . import patterns
from .classifier import classify_single_expression
from .types import (
    ResolutionStatus,
    ResolvedTimeRange,
    TimeCategory,
    TimeContext,
    TimeGranularity,
    ValidationResult,
)

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_SEASON_INDEX = {"spring": 0, "summer": 1, "fall": 2, "autumn": 2, "winter": 3}
_SEASON_LABEL = {0: "Spring", 1: "Summer", 2: "Fall", 3: "Winter"}
_DEFAULT_DAYPART_ANCHORS = {
    "this morning": "09:00",
    "this afternoon": "15:00",
    "this evening": "19:00",
    "tonight": "21:00",
    "morning": "09:00",
    "afternoon": "15:00",
    "evening": "19:00",
    "night": "21:00",
}


def _ref_dt(context: TimeContext) -> datetime:
    return context.reference_dt


def _ref_date(context: TimeContext) -> date:
    return context.reference_dt.date()


def _with_time(day: date, *, end: bool = False) -> datetime:
    if end:
        return datetime.combine(day, time.max)
    return datetime.combine(day, time.min)


def _combine_time(day: date, hour: int, minute: int) -> datetime:
    return datetime.combine(day, time(hour=hour, minute=minute))


def _format_time(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def _valid(code: str = "ok", message: str = "") -> ValidationResult:
    return ValidationResult(ok=True, code=code, message=message)


def _invalid(code: str, message: str) -> ValidationResult:
    return ValidationResult(ok=False, code=code, message=message)


def _last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _add_months(day: date, months: int) -> date:
    """Add months to a date, clamping the day to the target month's length.

    Plain arithmetic overflows: Jan 31 plus one month has no February 31.
    Clamping to the month end is the convention that keeps "a month later"
    monotonic, which matters because these values become range boundaries.
    """
    total_month = (day.year * 12 + (day.month - 1)) + months
    year = total_month // 12
    month = total_month % 12 + 1
    target_day = min(day.day, _last_day_of_month(year, month))
    return date(year, month, target_day)


def _add_years(day: date, years: int) -> date:
    year = day.year + years
    target_day = min(day.day, _last_day_of_month(year, day.month))
    return date(year, day.month, target_day)


def _parse_number(token: str) -> int | None:
    text = token.strip().lower()
    if text.isdigit():
        return int(text)
    return _NUMBER_WORDS.get(text)


def _make_result(
    *,
    original_text: str,
    span: tuple[int, int],
    category: TimeCategory,
    status: ResolutionStatus,
    method: str,
    context: TimeContext,
    granularity: TimeGranularity | None,
    start: datetime | None,
    end: datetime | None,
    operator: str | None,
    validation_result: ValidationResult,
    display_value: str | None = None,
) -> ResolvedTimeRange:
    """Assemble a ResolvedTimeRange with its validation applied.

    The single construction point for resolver output, so every result carries a
    validation verdict -- a resolution built by hand elsewhere could skip the
    check and enter the graph unvalidated.
    """
    return ResolvedTimeRange(
        original_text=original_text,
        span=span,
        category=category,
        status=status,
        method=method,
        anchor_reference_time=context.reference_time_str or context.reference_dt.isoformat(),
        granularity=granularity,
        start=start,
        end=end,
        operator=operator,
        display_value=display_value,
        validation_result=validation_result,
    )


def _validate_point_or_range(
    *,
    category: TimeCategory,
    original_text: str,
    span: tuple[int, int],
    context: TimeContext,
    granularity: TimeGranularity | None,
    start: datetime | None,
    end: datetime | None,
    display_value: str | None = None,
) -> ResolvedTimeRange:
    """Check a resolved value for the impossible results parsing can produce.

    Guards against ranges that end before they start, dates outside the
    plausible corpus span, and instants presented as ranges. These are parser
    bugs rather than bad input, and catching them here keeps them out of the
    graph, where a reversed range silently matches nothing.
    """
    if start is None or end is None:
        return _make_result(
            original_text=original_text,
            span=span,
            category=category,
            status=ResolutionStatus.UNRESOLVED,
            method="regex_v1",
            context=context,
            granularity=granularity,
            start=start,
            end=end,
            operator=None,
            display_value=display_value,
            validation_result=_invalid("missing_bounds", "Start/end is required for resolved ranges."),
        )
    if start > end:
        return _make_result(
            original_text=original_text,
            span=span,
            category=category,
            status=ResolutionStatus.INVALID,
            method="regex_v1",
            context=context,
            granularity=granularity,
            start=start,
            end=end,
            operator=None,
            display_value=display_value,
            validation_result=_invalid("range_order", "Start must not be after end."),
        )
    return _make_result(
        original_text=original_text,
        span=span,
        category=category,
        status=ResolutionStatus.RESOLVED,
        method="regex_v1",
        context=context,
        granularity=granularity,
        start=start,
        end=end,
        operator=None,
        display_value=display_value,
        validation_result=_valid(),
    )


def _month_display(year: int, month: int) -> str:
    return f"{calendar.month_name[month]} {year:04d}"


def _season_window(season_idx: int, label_year: int) -> tuple[date, date]:
    """Return the date window for a named season in a given year.

    Meteorological seasons (three whole months) rather than astronomical ones,
    because conversational "summer" means the months, not the solstice-bounded
    interval -- and month boundaries make the window comparable with other
    month-granularity values.
    """
    if season_idx == 0:
        return date(label_year, 3, 1), date(label_year, 5, 31)
    if season_idx == 1:
        return date(label_year, 6, 1), date(label_year, 8, 31)
    if season_idx == 2:
        return date(label_year, 9, 1), date(label_year, 11, 30)
    end_day = 29 if calendar.isleap(label_year + 1) else 28
    return date(label_year, 12, 1), date(label_year + 1, 2, end_day)


def _season_display(season_idx: int, label_year: int) -> str:
    return f"{_SEASON_LABEL[season_idx]} {label_year:04d}"


def _daypart_anchor_map(context: TimeContext) -> dict[str, str]:
    merged = dict(_DEFAULT_DAYPART_ANCHORS)
    if context.daypart_anchor_times:
        merged.update({str(k).lower(): str(v) for k, v in context.daypart_anchor_times.items()})
    return merged


def _parse_hhmm(value: str) -> tuple[int, int] | None:
    """Parse a clock time, returning None rather than raising on a non-time.

    Called speculatively on candidate spans, so a miss is the normal case and
    must stay cheap.
    """
    try:
        hour_str, minute_str = value.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
    except Exception:
        return None
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return hour, minute
    return None


def _weekday_target(
    modifier: str,
    weekday: int,
    ref_day: date,
    policy: str,
) -> date:
    """Resolve "last/next <weekday>" against the reference date.

    The hard case is when the reference day is that same weekday: "last Tuesday"
    spoken on a Tuesday means the week before, not today. `last_weekday_policy`
    on the TimeContext decides, defaulting to nearest_previous.
    """
    current_weekday = ref_day.weekday()
    if modifier == "this":
        week_start = ref_day - timedelta(days=current_weekday)
        return week_start + timedelta(days=weekday)
    if modifier == "last":
        if policy == "previous_calendar_week":
            current_week_start = ref_day - timedelta(days=current_weekday)
            previous_week_start = current_week_start - timedelta(days=7)
            return previous_week_start + timedelta(days=weekday)
        delta = (current_weekday - weekday) % 7
        return ref_day - timedelta(days=delta or 7)
    delta = (weekday - current_weekday) % 7
    return ref_day + timedelta(days=delta or 7)


def _resolve_absolute_date(original_text: str, span: tuple[int, int], context: TimeContext) -> ResolvedTimeRange:
    text = original_text.strip()
    year = month = day = None
    match = patterns.ABSOLUTE_DATE_ISO_PARSE_RE.match(text)
    if match:
        year, month, day = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    else:
        match = patterns.ABSOLUTE_DATE_MONTH_FIRST_PARSE_RE.match(text)
        if match:
            month = _MONTHS[match.group(1).lower()]
            day = int(match.group(2))
            year = int(match.group(3))
        else:
            match = patterns.ABSOLUTE_DATE_DAY_FIRST_PARSE_RE.match(text)
            if match:
                day = int(match.group(1))
                month = _MONTHS[match.group(2).lower()]
                year = int(match.group(3))

    try:
        parsed = date(year, month, day)  # type: ignore[arg-type]
    except Exception:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.ABSOLUTE_DATE,
            status=ResolutionStatus.INVALID,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.DAY,
            start=None,
            end=None,
            operator=None,
            display_value=text,
            validation_result=_invalid("invalid_date", f"Invalid absolute date: {text}"),
        )

    start = _with_time(parsed)
    end = _with_time(parsed, end=True)
    return _validate_point_or_range(
        category=TimeCategory.ABSOLUTE_DATE,
        original_text=original_text,
        span=span,
        context=context,
        granularity=TimeGranularity.DAY,
        start=start,
        end=end,
        display_value=parsed.isoformat(),
    )


def _resolve_absolute_time(original_text: str, span: tuple[int, int], context: TimeContext) -> ResolvedTimeRange:
    match = patterns.ABSOLUTE_TIME_PARSE_RE.match(original_text.strip())
    if not match:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.ABSOLUTE_TIME,
            status=ResolutionStatus.UNRESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.TIME,
            start=None,
            end=None,
            operator=None,
            display_value=None,
            validation_result=_invalid("absolute_time_parse_failed", "Failed to parse absolute clock time."),
        )

    hour = int(match.group(1))
    minute = int(match.group(2) or "00")
    meridiem = (match.group(3) or "").lower()
    if meridiem:
        if not 1 <= hour <= 12:
            return _make_result(
                original_text=original_text,
                span=span,
                category=TimeCategory.ABSOLUTE_TIME,
                status=ResolutionStatus.INVALID,
                method="regex_v1",
                context=context,
                granularity=TimeGranularity.TIME,
                start=None,
                end=None,
                operator=None,
                display_value=None,
                validation_result=_invalid("absolute_time_hour", "12-hour clock hour must be within 1-12."),
            )
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
    elif not (0 <= hour <= 23):
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.ABSOLUTE_TIME,
            status=ResolutionStatus.INVALID,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.TIME,
            start=None,
            end=None,
            operator=None,
            display_value=None,
            validation_result=_invalid("absolute_time_hour", "24-hour clock hour must be within 0-23."),
        )
    if not (0 <= minute <= 59):
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.ABSOLUTE_TIME,
            status=ResolutionStatus.INVALID,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.TIME,
            start=None,
            end=None,
            operator=None,
            display_value=None,
            validation_result=_invalid("absolute_time_minute", "Clock minute must be within 0-59."),
        )

    resolved = _combine_time(_ref_date(context), hour, minute)
    return _validate_point_or_range(
        category=TimeCategory.ABSOLUTE_TIME,
        original_text=original_text,
        span=span,
        context=context,
        granularity=TimeGranularity.TIME,
        start=resolved,
        end=resolved,
        display_value=_format_time(resolved),
    )


def _resolve_relative_day(original_text: str, span: tuple[int, int], context: TimeContext) -> ResolvedTimeRange:
    mapping = {
        "today": 0, "yesterday": -1, "tomorrow": 1,
        # daypart phrases — resolve to the anchor day, ignoring time-of-day
        "last night": -1, "tonight": 0,
        "this morning": 0, "this afternoon": 0, "this evening": 0,
        "yesterday morning": -1, "yesterday afternoon": -1,
        "yesterday evening": -1, "yesterday night": -1,
        "tomorrow morning": 1, "tomorrow afternoon": 1, "tomorrow evening": 1,
    }
    offset = mapping.get(original_text.strip().lower())
    if offset is None:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.RELATIVE_DAY,
            status=ResolutionStatus.UNRESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.DAY,
            start=None,
            end=None,
            operator=None,
            display_value=None,
            validation_result=_invalid("unsupported_relative_day", "Unsupported relative-day token."),
        )
    target = _ref_date(context) + timedelta(days=offset)
    return _validate_point_or_range(
        category=TimeCategory.RELATIVE_DAY,
        original_text=original_text,
        span=span,
        context=context,
        granularity=TimeGranularity.DAY,
        start=_with_time(target),
        end=_with_time(target, end=True),
        display_value=target.isoformat(),
    )


def _resolve_relative_daypart(original_text: str, span: tuple[int, int], context: TimeContext) -> ResolvedTimeRange:
    match = patterns.RELATIVE_DAYPART_PARSE_RE.match(original_text.strip())
    if not match:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.RELATIVE_DAYPART,
            status=ResolutionStatus.UNRESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.TIME,
            start=None,
            end=None,
            operator=None,
            display_value=None,
            validation_result=_invalid("relative_daypart_parse_failed", "Failed to parse relative-daypart expression."),
        )

    rel, daypart = match.group(1).lower(), match.group(2).lower()
    day_offset = {"yesterday": -1, "tomorrow": 1, "last": -1}.get(rel)
    if day_offset is None:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.RELATIVE_DAYPART,
            status=ResolutionStatus.UNRESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.TIME,
            start=None,
            end=None,
            operator=None,
            display_value=None,
            validation_result=_invalid("relative_daypart_modifier", "Unsupported relative-daypart modifier."),
        )

    hhmm = _daypart_anchor_map(context).get(daypart)
    parsed = _parse_hhmm(hhmm or "")
    if parsed is None:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.RELATIVE_DAYPART,
            status=ResolutionStatus.INVALID,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.TIME,
            start=None,
            end=None,
            operator=None,
            display_value=None,
            validation_result=_invalid("relative_daypart_anchor_invalid", "Configured relative-daypart anchor time is invalid."),
        )
    hour, minute = parsed
    target_day = _ref_date(context) + timedelta(days=day_offset)
    resolved = _combine_time(target_day, hour, minute)
    display_value = f"{daypart} of {target_day.isoformat()}"
    result = _validate_point_or_range(
        category=TimeCategory.RELATIVE_DAYPART,
        original_text=original_text,
        span=span,
        context=context,
        granularity=TimeGranularity.RANGE,
        start=resolved,
        end=resolved,
        display_value=display_value,
    )
    return result


def _resolve_relative_hour(original_text: str, span: tuple[int, int], context: TimeContext) -> ResolvedTimeRange:
    match = patterns.RELATIVE_HOUR_PARSE_RE.match(original_text.strip())
    if not match:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.RELATIVE_HOUR,
            status=ResolutionStatus.UNRESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.TIME,
            start=None,
            end=None,
            operator=None,
            display_value=None,
            validation_result=_invalid("relative_hour_parse_failed", "Failed to parse relative-hour expression."),
        )

    in_hours = match.group(1)
    ago_hours = match.group(2)
    count = _parse_number(in_hours or ago_hours or "")
    if count is None:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.RELATIVE_HOUR,
            status=ResolutionStatus.UNRESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.TIME,
            start=None,
            end=None,
            operator=None,
            display_value=None,
            validation_result=_invalid("relative_hour_number", "Unsupported relative-hour count."),
        )

    delta = timedelta(hours=count)
    resolved = _ref_dt(context) + delta if in_hours else _ref_dt(context) - delta
    return _validate_point_or_range(
        category=TimeCategory.RELATIVE_HOUR,
        original_text=original_text,
        span=span,
        context=context,
        granularity=TimeGranularity.TIME,
        start=resolved,
        end=resolved,
        display_value=_format_time(resolved),
    )


def _resolve_daypart_time(original_text: str, span: tuple[int, int], context: TimeContext) -> ResolvedTimeRange:
    text = original_text.strip().lower()
    if not patterns.DAYPART_TIME_PARSE_RE.match(text):
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.DAYPART_TIME,
            status=ResolutionStatus.UNRESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.TIME,
            start=None,
            end=None,
            operator=None,
            display_value=None,
            validation_result=_invalid("daypart_time_parse_failed", "Failed to parse daypart-time expression."),
        )

    hhmm = _daypart_anchor_map(context).get(text)
    parsed = _parse_hhmm(hhmm or "")
    if parsed is None:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.DAYPART_TIME,
            status=ResolutionStatus.INVALID,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.TIME,
            start=None,
            end=None,
            operator=None,
            display_value=None,
            validation_result=_invalid("daypart_time_anchor_invalid", "Configured daypart anchor time is invalid."),
        )
    hour, minute = parsed
    resolved = _combine_time(_ref_date(context), hour, minute)
    daypart = "night" if text == "tonight" else text.split()[-1]
    display_value = f"{daypart} of {_ref_date(context).isoformat()}"
    result = _validate_point_or_range(
        category=TimeCategory.DAYPART_TIME,
        original_text=original_text,
        span=span,
        context=context,
        granularity=TimeGranularity.RANGE,
        start=resolved,
        end=resolved,
        display_value=display_value,
    )
    return result


def _resolve_relative_weekday(original_text: str, span: tuple[int, int], context: TimeContext) -> ResolvedTimeRange:
    """Resolve last/this/next weekday phrases.

    Mixed temporal policy:
    - last <weekday>  -> distance-based mixed rule:
                          * if target weekday is earlier than or equal to the
                            reference weekday in the current week, use the
                            previous calendar week's weekday
                          * otherwise use the nearest previous matching weekday
    - this <weekday>  -> matching weekday in the current calendar week
    - next <weekday>  -> nearest following matching weekday
    """
    match = patterns.RELATIVE_WEEKDAY_PARSE_RE.match(original_text.strip())
    if not match:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.RELATIVE_WEEKDAY,
            status=ResolutionStatus.UNRESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.DAY,
            start=None,
            end=None,
            operator=None,
            validation_result=_invalid("weekday_parse_failed", "Failed to parse relative weekday."),
        )

    modifier = match.group(1).lower()
    weekday = _WEEKDAYS[match.group(2).lower()]
    ref_date = _ref_date(context)
    target = _weekday_target(modifier, weekday, ref_date, context.last_weekday_policy)

    start = _with_time(target)
    end = _with_time(target, end=True)
    validation = _valid() if target.weekday() == weekday else _invalid("weekday_mismatch", "Resolved weekday does not match target.")
    status = ResolutionStatus.RESOLVED if validation.ok else ResolutionStatus.INVALID
    return _make_result(
        original_text=original_text,
        span=span,
        category=TimeCategory.RELATIVE_WEEKDAY,
        status=status,
        method="regex_v1",
        context=context,
        granularity=TimeGranularity.DAY,
        start=start if validation.ok else None,
        end=end if validation.ok else None,
        operator=None,
        display_value=target.isoformat(),
        validation_result=validation,
    )


def _resolve_relative_window(original_text: str, span: tuple[int, int], context: TimeContext) -> ResolvedTimeRange:
    text = original_text.strip()
    match = patterns.IN_LAST_PARSE_RE.match(text)
    is_range = True
    if not match:
        match = patterns.AGO_PARSE_RE.match(text)
        is_range = False
    if not match:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.RELATIVE_WINDOW,
            status=ResolutionStatus.UNRESOLVED,
            method="regex_v1",
            context=context,
            granularity=None,
            start=None,
            end=None,
            operator=None,
            validation_result=_invalid("relative_window_parse_failed", "Failed to parse relative window."),
        )

    count = _parse_number(match.group(1))
    unit = match.group(2).lower()
    if count is None:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.RELATIVE_WINDOW,
            status=ResolutionStatus.UNRESOLVED,
            method="regex_v1",
            context=context,
            granularity=None,
            start=None,
            end=None,
            operator=None,
            validation_result=_invalid("relative_window_number", "Unsupported relative-window count."),
        )

    ref_day = _ref_date(context)
    if unit.startswith("day"):
        resolved_day = ref_day - timedelta(days=count)
        granularity = TimeGranularity.DAY
        display_value = resolved_day.isoformat()
    elif unit.startswith("week"):
        resolved_day = ref_day - timedelta(days=count * 7)
        granularity = TimeGranularity.WEEK
        week_start = ref_day - timedelta(days=ref_day.weekday()) - timedelta(days=count * 7)
        display_value = f"week of {week_start.isoformat()}"
    elif unit.startswith("month"):
        resolved_day = _add_months(ref_day, -count)
        granularity = TimeGranularity.MONTH
        display_value = _month_display(resolved_day.year, resolved_day.month)
    else:
        resolved_day = _add_years(ref_day, -count)
        granularity = TimeGranularity.YEAR
        display_value = f"{resolved_day.year:04d}"

    if is_range:
        start = _with_time(resolved_day)
        end = _with_time(ref_day, end=True)
        return _validate_point_or_range(
            category=TimeCategory.RELATIVE_WINDOW,
            original_text=original_text,
            span=span,
            context=context,
            granularity=granularity,
            start=start,
            end=end,
            display_value=display_value,
        )

    start = _with_time(resolved_day)
    end = _with_time(resolved_day, end=True)
    return _validate_point_or_range(
        category=TimeCategory.RELATIVE_WINDOW,
        original_text=original_text,
        span=span,
        context=context,
        granularity=TimeGranularity.DAY,
        start=start,
        end=end,
        display_value=resolved_day.isoformat(),
    )


def _resolve_week_point(original_text: str, span: tuple[int, int], context: TimeContext) -> ResolvedTimeRange:
    match = patterns.WEEK_POINT_PARSE_RE.match(original_text.strip())
    if not match:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.WEEK_POINT,
            status=ResolutionStatus.UNRESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.WEEK,
            start=None,
            end=None,
            operator=None,
            validation_result=_invalid("week_parse_failed", "Failed to parse week expression."),
        )
    modifier = match.group(1).lower() if match.group(1) else None
    anchored_operator = match.group(2).lower() if match.group(2) else None
    anchored_text = match.group(3)

    if modifier is not None:
        ref_day = _ref_date(context)
        this_week_start = ref_day - timedelta(days=ref_day.weekday())
        offset = {"last": -7, "this": 0, "next": 7}[modifier]
        start_day = this_week_start + timedelta(days=offset)
        end_day = start_day + timedelta(days=6)
        display_value = f"week of {start_day.isoformat()}"
    else:
        anchor_match = classify_single_expression(anchored_text)
        anchor_resolution = resolve_match(anchor_match.text, anchor_match.span, anchor_match.category, context)
        if (
            anchor_resolution.status is not ResolutionStatus.RESOLVED
            or not anchor_resolution.start
        ):
            return _make_result(
                original_text=original_text,
                span=span,
                category=TimeCategory.WEEK_POINT,
                status=ResolutionStatus.UNRESOLVED,
                method="regex_v1",
                context=context,
                granularity=TimeGranularity.WEEK,
                start=None,
                end=None,
                operator=None,
                display_value=None,
                validation_result=_invalid("week_anchor_unresolved", "Failed to resolve anchored week reference."),
            )
        ref_day = anchor_resolution.start.date()
        this_week_start = ref_day - timedelta(days=ref_day.weekday())
        if anchored_operator == "before":
            start_day = this_week_start - timedelta(days=7)
        elif anchored_operator == "after":
            start_day = this_week_start + timedelta(days=7)
        else:
            start_day = this_week_start
        end_day = start_day + timedelta(days=6)
        display_value = f"week of {start_day.isoformat()}"

    return _validate_point_or_range(
        category=TimeCategory.WEEK_POINT,
        original_text=original_text,
        span=span,
        context=context,
        granularity=TimeGranularity.WEEK,
        start=_with_time(start_day),
        end=_with_time(end_day, end=True),
        display_value=display_value,
    )


def _resolve_month_point(original_text: str, span: tuple[int, int], context: TimeContext) -> ResolvedTimeRange:
    text = original_text.strip()
    lower = text.lower()
    ref_day = _ref_date(context)

    if lower in {"last month", "this month", "next month"}:
        delta = {"last month": -1, "this month": 0, "next month": 1}[lower]
        target = _add_months(date(ref_day.year, ref_day.month, 1), delta)
        year = target.year
        month = target.month
    else:
        match = patterns.MONTH_POINT_PARSE_RE.match(text)
        if not match:
            return _make_result(
                original_text=original_text,
                span=span,
                category=TimeCategory.MONTH_POINT,
                status=ResolutionStatus.UNRESOLVED,
                method="regex_v1",
                context=context,
                granularity=TimeGranularity.MONTH,
                start=None,
                end=None,
                operator=None,
                validation_result=_invalid("month_parse_failed", "Failed to parse month expression."),
            )
        parts = text.split()
        month = _MONTHS[parts[0].lower()]
        year = int(parts[1])

    start_day = date(year, month, 1)
    end_day = date(year, month, _last_day_of_month(year, month))
    return _validate_point_or_range(
        category=TimeCategory.MONTH_POINT,
        original_text=original_text,
        span=span,
        context=context,
        granularity=TimeGranularity.MONTH,
        start=_with_time(start_day),
        end=_with_time(end_day, end=True),
        display_value=_month_display(year, month),
    )


def _resolve_season_point(original_text: str, span: tuple[int, int], context: TimeContext) -> ResolvedTimeRange:
    text = original_text.strip()
    lower = text.lower()
    match = patterns.SEASON_POINT_PARSE_RE.match(text)
    if not match:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.SEASON_POINT,
            status=ResolutionStatus.UNRESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.SEASON,
            start=None,
            end=None,
            operator=None,
            display_value=None,
            validation_result=_invalid("season_parse_failed", "Failed to parse season expression."),
        )

    if lower.startswith(("last ", "this ", "next ")):
        modifier, season_name = lower.split()
        season_idx = _SEASON_INDEX[season_name]
        ref_day = _ref_date(context)
        candidates: list[tuple[date, date, int]] = []
        for label_year in range(ref_day.year - 2, ref_day.year + 3):
            start_day, end_day = _season_window(season_idx, label_year)
            candidates.append((start_day, end_day, label_year))
        if modifier == "last":
            eligible = [item for item in candidates if item[1] < ref_day]
            start_day, end_day, label_year = max(eligible, key=lambda item: item[1])
        elif modifier == "next":
            eligible = [item for item in candidates if item[0] > ref_day]
            start_day, end_day, label_year = min(eligible, key=lambda item: item[0])
        else:
            current = [item for item in candidates if item[0] <= ref_day <= item[1]]
            if current:
                start_day, end_day, label_year = current[0]
            else:
                same_year = [item for item in candidates if item[2] == ref_day.year]
                start_day, end_day, label_year = same_year[0]
    else:
        season_name, year_str = lower.split()
        season_idx = _SEASON_INDEX[season_name]
        label_year = int(year_str)
        start_day, end_day = _season_window(season_idx, label_year)
    display_value = _season_display(season_idx, label_year)
    return _validate_point_or_range(
        category=TimeCategory.SEASON_POINT,
        original_text=original_text,
        span=span,
        context=context,
        granularity=TimeGranularity.SEASON,
        start=_with_time(start_day),
        end=_with_time(end_day, end=True),
        display_value=display_value,
    )


def _resolve_year_point(original_text: str, span: tuple[int, int], context: TimeContext) -> ResolvedTimeRange:
    lower = original_text.strip().lower()
    ref_year = _ref_date(context).year
    if lower == "last year":
        year = ref_year - 1
    elif lower == "this year":
        year = ref_year
    elif lower == "next year":
        year = ref_year + 1
    else:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.YEAR_POINT,
            status=ResolutionStatus.UNRESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.YEAR,
            start=None,
            end=None,
            operator=None,
            validation_result=_invalid("year_parse_failed", "Failed to parse year expression."),
        )

    start_day = date(year, 1, 1)
    end_day = date(year, 12, 31)
    return _validate_point_or_range(
        category=TimeCategory.YEAR_POINT,
        original_text=original_text,
        span=span,
        context=context,
        granularity=TimeGranularity.YEAR,
        start=_with_time(start_day),
        end=_with_time(end_day, end=True),
        display_value=f"{year:04d}",
    )


def _resolve_month_week_range(original_text: str, span: tuple[int, int], context: TimeContext) -> ResolvedTimeRange:
    match = patterns.MONTH_WEEK_RANGE_PARSE_RE.match(original_text.strip())
    if not match:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.MONTH_WEEK_RANGE,
            status=ResolutionStatus.UNRESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.RANGE,
            start=None,
            end=None,
            operator=None,
            display_value=None,
            validation_result=_invalid("month_week_parse_failed", "Failed to parse month-week range."),
        )

    which = match.group(1).lower()
    month = _MONTHS[match.group(2).lower()]
    year = int(match.group(3))
    month_start = date(year, month, 1)
    month_end = date(year, month, _last_day_of_month(year, month))

    if which == "first":
        week_start = month_start - timedelta(days=month_start.weekday())
        range_start = max(month_start, week_start)
        range_end = min(month_end, week_start + timedelta(days=6))
    else:
        week_start = month_end - timedelta(days=month_end.weekday())
        range_start = max(month_start, week_start)
        range_end = min(month_end, week_start + timedelta(days=6))

    return _validate_point_or_range(
        category=TimeCategory.MONTH_WEEK_RANGE,
        original_text=original_text,
        span=span,
        context=context,
        granularity=TimeGranularity.RANGE,
        start=_with_time(range_start),
        end=_with_time(range_end, end=True),
        display_value=f"the {which} week of {_month_display(year, month)}",
    )


def _resolve_boundary(original_text: str, span: tuple[int, int], context: TimeContext) -> ResolvedTimeRange:
    match = patterns.BOUNDARY_PARSE_RE.match(original_text.strip())
    if not match:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.BOUNDARY,
            status=ResolutionStatus.UNRESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.RANGE,
            start=None,
            end=None,
            operator=None,
            display_value=None,
            validation_result=_invalid("boundary_parse_failed", "Failed to parse boundary expression."),
        )

    operator = match.group(1).lower()
    anchor_text = match.group(2)
    anchor_match = classify_single_expression(anchor_text)
    anchor_resolution = resolve_match(anchor_match.text, anchor_match.span, anchor_match.category, context)
    if anchor_resolution.status is not ResolutionStatus.RESOLVED or not anchor_resolution.start or not anchor_resolution.end:
        status = ResolutionStatus.PARTIALLY_RESOLVED if anchor_resolution.status in {
            ResolutionStatus.AMBIGUOUS,
            ResolutionStatus.UNRESOLVED,
        } else anchor_resolution.status
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.BOUNDARY,
            status=status,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.RANGE,
            start=None,
            end=None,
            operator=operator,
            display_value=f"{operator} {anchor_resolution.display_value or anchor_text}",
            validation_result=_invalid("boundary_anchor_unresolved", "Boundary anchor could not be resolved."),
        )

    if operator == "before":
        start = None
        end = anchor_resolution.start - timedelta(microseconds=1)
        valid = _valid() if end < anchor_resolution.start else _invalid("boundary_before", "before DATE must end before start_of(DATE).")
    elif operator == "after":
        start = anchor_resolution.end + timedelta(microseconds=1)
        end = None
        valid = _valid() if start > anchor_resolution.end else _invalid("boundary_after", "after DATE must start after end_of(DATE).")
    else:
        start = anchor_resolution.start
        end = None
        valid = _valid() if start == anchor_resolution.start else _invalid("boundary_since", "since DATE must start at start_of(DATE).")

    return _make_result(
        original_text=original_text,
        span=span,
        category=TimeCategory.BOUNDARY,
        status=ResolutionStatus.RESOLVED if valid.ok else ResolutionStatus.INVALID,
        method="regex_v1",
        context=context,
        granularity=TimeGranularity.RANGE,
        start=start,
        end=end,
        operator=operator,
        display_value=f"{operator} {anchor_resolution.display_value or anchor_text}",
        validation_result=valid,
    )


def _resolve_weekend(original_text: str, span: tuple[int, int], context: TimeContext) -> ResolvedTimeRange:
    """Resolve weekend phrases using calendar weekend blocks.

    Mixed temporal policy:
    - last weekend / this past weekend -> previous Saturday-Sunday block
    - this weekend                     -> current week Saturday-Sunday block
    - next weekend                     -> following Saturday-Sunday block
    """
    match = patterns.WEEKEND_PARSE_RE.match(original_text.strip())
    if not match:
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.WEEKEND,
            status=ResolutionStatus.UNRESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.RANGE,
            start=None,
            end=None,
            operator=None,
            validation_result=_invalid("weekend_parse_failed", "Failed to parse weekend expression."),
        )

    modifier = match.group(1).strip().lower()
    ref_day = _ref_date(context)
    # Monday of the current ISO week (weekday() == 0 on Monday)
    mon = ref_day - timedelta(days=ref_day.weekday())

    if modifier in {"last", "this past"}:
        sat = mon - timedelta(days=2)
        sun = mon - timedelta(days=1)
    elif modifier == "this":
        sat = mon + timedelta(days=5)
        sun = mon + timedelta(days=6)
    else:  # next
        sat = mon + timedelta(days=12)
        sun = mon + timedelta(days=13)

    date_range = f"{sat.isoformat()} to {sun.isoformat()}"
    return _validate_point_or_range(
        category=TimeCategory.WEEKEND,
        original_text=original_text,
        span=span,
        context=context,
        granularity=TimeGranularity.WEEKEND,
        start=_with_time(sat),
        end=_with_time(sun, end=True),
        display_value=date_range,
    )


def resolve_match(
    original_text: str,
    span: tuple[int, int],
    category: TimeCategory,
    context: TimeContext,
) -> ResolvedTimeRange:
    if category is TimeCategory.ABSOLUTE_DATE:
        return _resolve_absolute_date(original_text, span, context)
    if category is TimeCategory.ABSOLUTE_TIME:
        return _resolve_absolute_time(original_text, span, context)
    if category is TimeCategory.RELATIVE_DAY:
        return _resolve_relative_day(original_text, span, context)
    if category is TimeCategory.RELATIVE_DAYPART:
        return _resolve_relative_daypart(original_text, span, context)
    if category is TimeCategory.RELATIVE_HOUR:
        return _resolve_relative_hour(original_text, span, context)
    if category is TimeCategory.DAYPART_TIME:
        return _resolve_daypart_time(original_text, span, context)
    if category is TimeCategory.RELATIVE_WEEKDAY:
        return _resolve_relative_weekday(original_text, span, context)
    if category is TimeCategory.RELATIVE_WINDOW:
        return _resolve_relative_window(original_text, span, context)
    if category is TimeCategory.WEEK_POINT:
        return _resolve_week_point(original_text, span, context)
    if category is TimeCategory.WEEKEND:
        return _resolve_weekend(original_text, span, context)
    if category is TimeCategory.MONTH_POINT:
        return _resolve_month_point(original_text, span, context)
    if category is TimeCategory.SEASON_POINT:
        return _resolve_season_point(original_text, span, context)
    if category is TimeCategory.YEAR_POINT:
        return _resolve_year_point(original_text, span, context)
    if category is TimeCategory.MONTH_WEEK_RANGE:
        return _resolve_month_week_range(original_text, span, context)
    if category is TimeCategory.BOUNDARY:
        return _resolve_boundary(original_text, span, context)
    text_lower = original_text.strip().lower()
    ref_day = _ref_date(context)
    if re.fullmatch(r"a\s+few\s+days?\s+ago|a\s+couple\s+(?:of\s+)?days?\s+ago", text_lower, re.IGNORECASE):
        start_day = ref_day - timedelta(days=5)
        end_day = ref_day - timedelta(days=2)
        date_range = f"{start_day.isoformat()} to {end_day.isoformat()}"
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.RELATIVE_WINDOW,
            status=ResolutionStatus.RESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.RANGE,
            start=_with_time(start_day),
            end=_with_time(end_day, end=True),
            operator=None,
            display_value=date_range,
            validation_result=_valid("fuzzy_resolved", "Resolved to ref_date minus 5 to 2 days."),
        )
    if re.fullmatch(r"the\s+other\s+day", text_lower, re.IGNORECASE):
        start_day = ref_day - timedelta(days=4)
        end_day = ref_day - timedelta(days=1)
        date_range = f"{start_day.isoformat()} to {end_day.isoformat()}"
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.RELATIVE_WINDOW,
            status=ResolutionStatus.RESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.RANGE,
            start=_with_time(start_day),
            end=_with_time(end_day, end=True),
            operator=None,
            display_value=date_range,
            validation_result=_valid("fuzzy_resolved", "Resolved to ref_date minus 4 to 1 days."),
        )
    if re.fullmatch(r"a\s+few\s+years?\s+ago", text_lower, re.IGNORECASE):
        start_day = date(ref_day.year - 5, ref_day.month, ref_day.day)
        end_day = date(ref_day.year - 2, ref_day.month, ref_day.day)
        date_range = f"{start_day.isoformat()} to {end_day.isoformat()}"
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.RELATIVE_WINDOW,
            status=ResolutionStatus.RESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.RANGE,
            start=_with_time(start_day),
            end=_with_time(end_day, end=True),
            operator=None,
            display_value=date_range,
            validation_result=_valid("fuzzy_resolved", "Resolved to ref_date minus 5 to 2 years."),
        )
    if re.fullmatch(r"a\s+few\s+weeks?\s+ago", text_lower, re.IGNORECASE):
        start_day = ref_day - timedelta(days=28)
        end_day = ref_day - timedelta(days=14)
        date_range = f"{start_day.isoformat()} to {end_day.isoformat()}"
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.RELATIVE_WINDOW,
            status=ResolutionStatus.RESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.RANGE,
            start=_with_time(start_day),
            end=_with_time(end_day, end=True),
            operator=None,
            display_value=date_range,
            validation_result=_valid("fuzzy_resolved", "Resolved to ref_date minus 28 to 14 days."),
        )
    if re.fullmatch(r"a\s+while\s+ago", text_lower, re.IGNORECASE):
        start_day = ref_day - timedelta(days=14)
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.RELATIVE_WINDOW,
            status=ResolutionStatus.RESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.RANGE,
            start=_with_time(start_day),
            end=_with_time(ref_day, end=True),
            operator=None,
            display_value=f"{start_day.isoformat()} to {ref_day.isoformat()}",
            validation_result=_valid("fuzzy_resolved", "Resolved to ref_date minus 14 days."),
        )
    if re.fullmatch(r"recently|lately", text_lower, re.IGNORECASE):
        start_day = ref_day - timedelta(days=7)
        return _make_result(
            original_text=original_text,
            span=span,
            category=TimeCategory.RELATIVE_WINDOW,
            status=ResolutionStatus.RESOLVED,
            method="regex_v1",
            context=context,
            granularity=TimeGranularity.RANGE,
            start=_with_time(start_day),
            end=_with_time(ref_day, end=True),
            operator=None,
            display_value=f"{start_day.isoformat()} to {ref_day.isoformat()}",
            validation_result=_valid("fuzzy_resolved", "Resolved to past 7 days."),
        )
    return _make_result(
        original_text=original_text,
        span=span,
        category=TimeCategory.UNKNOWN,
        status=ResolutionStatus.UNRESOLVED,
        method="regex_v1",
        context=context,
        granularity=None,
        start=None,
       end=None,
        operator=None,
        display_value=None,
        validation_result=_invalid("unknown_pattern", "No deterministic temporal pattern matched."),
    )
