from __future__ import annotations

from datetime import datetime

from KG.utils.temporal import build_time_context, extract_temporal_constraints


def _single(text: str):
    ctx = build_time_context(
        reference_dt=datetime(2023, 4, 12, 12, 0, 0),
        reference_time_str="2023-04-12T12:00:00",
        source="test",
    )
    return extract_temporal_constraints(text, ctx)[0]


def test_before_is_exclusive_and_preserves_operator():
    constraint = _single("before April 10, 2023")
    assert constraint.operator == "before"
    assert constraint.resolution.status.value == "resolved"
    assert constraint.resolution.operator == "before"
    assert constraint.resolution.start is None
    assert constraint.resolution.end.isoformat() == "2023-04-09T23:59:59.999999"


def test_after_is_exclusive_and_preserves_operator():
    constraint = _single("after April 10, 2023")
    assert constraint.operator == "after"
    assert constraint.resolution.start.isoformat() == "2023-04-11T00:00:00"
    assert constraint.resolution.end is None


def test_since_is_inclusive_and_preserves_operator():
    constraint = _single("since April 10, 2023")
    assert constraint.operator == "since"
    assert constraint.resolution.start.isoformat() == "2023-04-10T00:00:00"
    assert constraint.resolution.end is None


def test_boundary_anchor_resolution_is_preserved():
    constraint = _single("before last Friday")
    assert constraint.operator == "before"
    assert constraint.anchor_text == "last Friday"
    assert constraint.anchor_resolution is not None
    assert constraint.anchor_resolution.display_value == "2023-04-07"
