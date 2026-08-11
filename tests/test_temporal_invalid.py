from __future__ import annotations

from datetime import datetime

from KG.utils.temporal import build_time_context, extract_temporal_constraints


def test_invalid_dates_return_invalid_status():
    ctx = build_time_context(
        reference_dt=datetime(2023, 4, 12, 12, 0, 0),
        reference_time_str="2023-04-12T12:00:00",
    )
    constraint = extract_temporal_constraints("February 30, 2023", ctx)[0]
    resolution = constraint.resolution
    assert resolution.status.value == "invalid"
    assert resolution.validation_result.ok is False
    assert resolution.start is None
    assert resolution.end is None
