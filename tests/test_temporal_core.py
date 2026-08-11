from __future__ import annotations

from datetime import datetime

from KG.utils.temporal import build_time_context, extract_temporal_constraints


def _ctx(ref: datetime | None = None, *, last_weekday_policy: str = "nearest_previous"):
    return build_time_context(
        reference_dt=ref or datetime(2023, 4, 12, 12, 0, 0),
        reference_time_str="2023-04-12T12:00:00",
        source="tests",
        last_weekday_policy=last_weekday_policy,
    )


def _single(text: str, ref: datetime | None = None, *, last_weekday_policy: str = "nearest_previous"):
    return extract_temporal_constraints(text, _ctx(ref, last_weekday_policy=last_weekday_policy))[0]


def test_absolute_date_resolves_to_exact_date_entity_value():
    resolution = _single("April 10, 2023").resolution
    assert resolution.granularity.value == "day"
    assert resolution.display_value == "2023-04-10"
    assert resolution.display_value == "2023-04-10"


def test_relative_day_resolves_to_exact_date():
    assert _single("yesterday").resolution.display_value == "2023-04-11"


def test_absolute_time_parses_12h_and_24h_forms():
    three_pm = _single("at 3 pm").resolution
    three_thirty = _single("3:30pm").resolution
    fifteen_thirty = _single("15:30").resolution
    assert three_pm.granularity.value == "time"
    assert three_pm.display_value == "15:00"
    assert three_thirty.display_value == "15:30"
    assert fifteen_thirty.display_value == "15:30"


def test_relative_hours_parse_from_reference_datetime():
    in_two = _single("in 2 hours").resolution
    two_ago = _single("2 hours ago").resolution
    assert in_two.display_value == "14:00"
    assert two_ago.display_value == "10:00"


def test_daypart_time_uses_default_anchor_mapping():
    resolution = _single("this afternoon").resolution
    assert resolution.granularity.value == "range"
    assert resolution.display_value == "afternoon of 2023-04-12"


def test_bare_night_resolves_to_today_timespan():
    resolution = _single("night").resolution
    assert resolution.granularity.value == "range"
    assert resolution.display_value == "night of 2023-04-12"


def test_daypart_time_supports_configurable_anchor_mapping():
    ctx = build_time_context(
        reference_dt=datetime(2023, 4, 12, 12, 0, 0),
        reference_time_str="2023-04-12T12:00:00",
        source="tests",
        daypart_anchor_times={"this afternoon": "16:30", "tonight": "22:15"},
    )
    resolution = extract_temporal_constraints("tonight", ctx)[0].resolution
    assert resolution.display_value == "night of 2023-04-12"


def test_tomorrow_morning_resolves_with_both_date_and_time_semantics():
    resolution = _single("tomorrow morning").resolution
    assert resolution.granularity.value == "range"
    assert resolution.display_value == "morning of 2023-04-13"
    assert resolution.start.date().isoformat() == "2023-04-13"


def test_yesterday_evening_resolves_with_both_date_and_time_semantics():
    resolution = _single("yesterday evening").resolution
    assert resolution.display_value == "evening of 2023-04-11"
    assert resolution.start.date().isoformat() == "2023-04-11"


def test_last_weekday_default_policy_uses_nearest_previous():
    resolution = _single("last Friday", datetime(2023, 7, 15, 12, 0, 0)).resolution
    assert resolution.display_value == "2023-07-14"


def test_last_weekday_locomo_policy_uses_previous_calendar_week():
    resolution = _single(
        "last Friday",
        datetime(2023, 7, 15, 12, 0, 0),
        last_weekday_policy="previous_calendar_week",
    ).resolution
    assert resolution.display_value == "2023-07-07"


def test_last_week_uses_natural_week_display_with_normalized_bounds():
    resolution = _single("last week").resolution
    assert resolution.granularity.value == "week"
    assert resolution.display_value == "week of 2023-04-03"
    assert resolution.start.date().isoformat() == "2023-04-03"
    assert resolution.end.date().isoformat() == "2023-04-09"


def test_anchored_week_before_absolute_date_preserves_full_timespan_phrase():
    resolution = _single("the week before 2023-06-09").resolution
    assert resolution.granularity.value == "week"
    assert resolution.display_value == "week of 2023-05-29"
    assert resolution.start.date().isoformat() == "2023-05-29"
    assert resolution.end.date().isoformat() == "2023-06-04"


def test_last_weekend_display_and_normalized_use_date_range_format():
    resolution = _single("last weekend").resolution
    assert resolution.granularity.value == "weekend"
    assert resolution.display_value == "2023-04-08 to 2023-04-09"
    assert resolution.display_value == "2023-04-08 to 2023-04-09"
    assert resolution.start.date().isoformat() == "2023-04-08"
    assert resolution.end.date().isoformat() == "2023-04-09"


def test_last_and_next_month_use_natural_names():
    last_month = _single("last month").resolution
    next_month = _single("next month").resolution
    assert last_month.display_value == "March 2023"
    assert last_month.start.date().isoformat() == "2023-03-01"
    assert last_month.end.date().isoformat() == "2023-03-31"
    assert next_month.display_value == "May 2023"


def test_last_and_next_season_use_natural_names():
    last_summer = _single("last summer", datetime(2023, 5, 2, 9, 0, 0)).resolution
    next_winter = _single("next winter", datetime(2023, 5, 2, 9, 0, 0)).resolution
    assert last_summer.display_value == "Summer 2022"
    assert last_summer.start.date().isoformat() == "2022-06-01"
    assert last_summer.end.date().isoformat() == "2022-08-31"
    assert next_winter.display_value == "Winter 2023"
    assert next_winter.start.date().isoformat() == "2023-12-01"
    assert next_winter.end.date().isoformat() == "2024-02-29"


def test_last_and_next_year_use_year_display_not_iso_range():
    last_year = _single("last year").resolution
    next_year = _single("next year").resolution
    assert last_year.display_value == "2022"
    assert next_year.display_value == "2024"
    assert last_year.start.date().isoformat() == "2022-01-01"
    assert last_year.end.date().isoformat() == "2022-12-31"


def test_fuzzy_phrase_resolves_to_range():
    resolution = _single("a few days ago").resolution
    assert resolution.status.value == "resolved"
    assert resolution.granularity.value == "range"
    assert resolution.start.date().isoformat() == "2023-04-07"
    assert resolution.end.date().isoformat() == "2023-04-10"
    assert resolution.display_value == "2023-04-07 to 2023-04-10"
