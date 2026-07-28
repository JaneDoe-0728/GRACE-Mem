"""Regression tests for temporal coverage: weekday abbreviations, daypart phrases, weekends."""

from __future__ import annotations

from datetime import datetime

import pytest

from KG.utils.temporal import build_time_context, extract_temporal_constraints, rewrite_temporal_text


def _ctx(ref: datetime | None = None) -> object:
    return build_time_context(
        reference_dt=ref or datetime(2023, 4, 12, 12, 0, 0),  # Wednesday 2023-04-12
        reference_time_str="2023/04/12 12:00",
        source="test",
    )


def _single(text: str, ref: datetime | None = None):
    constraints = extract_temporal_constraints(text, _ctx(ref))
    assert constraints, f"No temporal constraints found in: {text!r}"
    return constraints[0]


# ---------------------------------------------------------------------------
# Weekday abbreviations
# ---------------------------------------------------------------------------

def test_last_fri_resolves():
    # ref: Wed 2023-04-12 → last Friday = 2023-04-07
    c = _single("last Fri")
    assert c.resolution.status.value == "resolved"
    assert c.resolution.display_value == "2023-04-07"


def test_last_fri_period_form_resolves():
    # "last Fri." — period is not captured but "last Fri" still matches at word boundary
    constraints = extract_temporal_constraints("We met last Fri.", _ctx())
    assert constraints
    assert constraints[0].resolution.display_value == "2023-04-07"


def test_last_tues_resolves():
    # default nearest_previous policy: ref Wed 2023-04-12 → last Tuesday = 2023-04-11
    c = _single("last Tues")
    assert c.resolution.status.value == "resolved"
    assert c.resolution.display_value == "2023-04-11"


def test_last_tue_resolves():
    c = _single("last Tue")
    assert c.resolution.display_value == "2023-04-11"


def test_next_mon_resolves():
    # ref: Wed 2023-04-12 → next Monday = 2023-04-17
    c = _single("next Mon")
    assert c.resolution.status.value == "resolved"
    assert c.resolution.display_value == "2023-04-17"


def test_this_wed_resolves():
    # ref: Wed 2023-04-12 → this Wednesday = 2023-04-12
    c = _single("this Wed")
    assert c.resolution.status.value == "resolved"
    assert c.resolution.display_value == "2023-04-12"


def test_last_thurs_resolves():
    # ref: Wed 2023-04-12 → last Thursday = 2023-04-06
    c = _single("last Thurs")
    assert c.resolution.status.value == "resolved"
    assert c.resolution.display_value == "2023-04-06"


def test_last_sat_resolves():
    # ref: Wed 2023-04-12 → last Saturday = 2023-04-08
    c = _single("last Sat")
    assert c.resolution.status.value == "resolved"
    assert c.resolution.display_value == "2023-04-08"


# ---------------------------------------------------------------------------
# Daypart phrases
# ---------------------------------------------------------------------------

def test_last_night_resolves_to_yesterday():
    c = _single("last night")
    assert c.resolution.status.value == "resolved"
    assert c.resolution.display_value == "night of 2023-04-11"
    assert c.resolution.start.date().isoformat() == "2023-04-11"


def test_tonight_resolves_to_today():
    c = _single("tonight")
    assert c.resolution.status.value == "resolved"
    assert c.resolution.display_value == "night of 2023-04-12"


def test_bare_night_resolves_to_today():
    c = _single("night")
    assert c.resolution.status.value == "resolved"
    assert c.resolution.display_value == "night of 2023-04-12"


def test_this_morning_resolves_to_today():
    c = _single("this morning")
    assert c.resolution.display_value == "morning of 2023-04-12"


def test_yesterday_morning_resolves_to_yesterday():
    c = _single("yesterday morning")
    assert c.resolution.display_value == "morning of 2023-04-11"
    assert c.resolution.start.date().isoformat() == "2023-04-11"


def test_tomorrow_afternoon_resolves_to_tomorrow():
    c = _single("tomorrow afternoon")
    assert c.resolution.display_value == "afternoon of 2023-04-13"
    assert c.resolution.start.date().isoformat() == "2023-04-13"


def test_yesterday_morning_prefers_compound_over_bare_yesterday():
    # The compound form must win; exactly one constraint covering the whole phrase.
    constraints = extract_temporal_constraints("yesterday morning", _ctx())
    assert len(constraints) == 1
    assert constraints[0].original_text.lower() == "yesterday morning"


# ---------------------------------------------------------------------------
# Weekend
# ---------------------------------------------------------------------------

def test_last_weekend_resolves():
    # ref: Wed 2023-04-12 → last weekend = Sat 2023-04-08 to Sun 2023-04-09
    c = _single("last weekend")
    assert c.resolution.status.value == "resolved"
    assert c.resolution.granularity.value == "weekend"
    assert c.resolution.display_value == "2023-04-08 to 2023-04-09"
    assert c.resolution.start.date().isoformat() == "2023-04-08"
    assert c.resolution.end.date().isoformat() == "2023-04-09"


def test_this_past_weekend_equals_last_weekend():
    last = _single("last weekend").resolution
    past = _single("this past weekend").resolution
    assert last.start == past.start
    assert last.end == past.end


def test_this_weekend_resolves_to_upcoming():
    # ref: Wed 2023-04-12 → this weekend = Sat 2023-04-15 to Sun 2023-04-16
    c = _single("this weekend")
    assert c.resolution.start.date().isoformat() == "2023-04-15"
    assert c.resolution.end.date().isoformat() == "2023-04-16"


def test_next_weekend_resolves():
    # ref: Wed 2023-04-12 → next weekend = Sat 2023-04-22 to Sun 2023-04-23
    c = _single("next weekend")
    assert c.resolution.start.date().isoformat() == "2023-04-22"
    assert c.resolution.end.date().isoformat() == "2023-04-23"


# ---------------------------------------------------------------------------
# Last year
# ---------------------------------------------------------------------------

def test_last_year_resolves_to_full_year_range():
    # ref: 2023 → last year = 2022-01-01 to 2022-12-31
    c = _single("last year")
    assert c.resolution.status.value == "resolved"
    assert c.resolution.display_value == "2022"
    assert c.resolution.start.date().isoformat() == "2022-01-01"
    assert c.resolution.end.date().isoformat() == "2022-12-31"


# ---------------------------------------------------------------------------
# Non-temporal queries must pass through unchanged
# ---------------------------------------------------------------------------

def test_non_temporal_query_unchanged():
    rewritten, meta = rewrite_temporal_text("What color was the car?", _ctx())
    assert rewritten == "What color was the car?"
    assert meta["constraints"] == []


def test_no_temporal_in_generic_sentence():
    constraints = extract_temporal_constraints("I like coffee in the morning.", _ctx())
    # "in the morning" is not a temporal expression the core resolves
    assert all(
        c.resolution.status.value not in ("resolved",) or c.original_text.lower() != "in the morning"
        for c in constraints
    )


# ---------------------------------------------------------------------------
# Fuzzy phrases must NOT be over-resolved
# ---------------------------------------------------------------------------

def test_recently_resolves_to_past_7_days():
    c = _single("I saw her recently")
    assert c.resolution.status.value == "resolved"
    assert c.resolution.start.date().isoformat() == "2023-04-05"
    assert c.resolution.display_value == "2023-04-05 to 2023-04-12"


def test_the_other_day_resolves_to_range():
    c = _single("the other day")
    assert c.resolution.status.value == "resolved"
    assert c.resolution.granularity.value == "range"
    assert c.resolution.start.date().isoformat() == "2023-04-08"
    assert c.resolution.end.date().isoformat() == "2023-04-11"


def test_a_while_ago_resolves_to_14_day_range():
    c = _single("a while ago")
    assert c.resolution.status.value == "resolved"
    assert c.resolution.start.date().isoformat() == "2023-03-29"
    assert c.resolution.end.date().isoformat() == "2023-04-12"


def test_fuzzy_phrase_is_rewritten():
    rewritten, _ = rewrite_temporal_text("I saw him recently.", _ctx())
    assert rewritten != "I saw him recently."
    assert "2023-04-05" in rewritten
