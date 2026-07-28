from __future__ import annotations

from datetime import datetime

from KG.utils.temporal import build_time_context, extract_temporal_constraints, rewrite_temporal_text


def _ctx():
    return build_time_context(
        reference_dt=datetime(2023, 4, 12, 12, 0, 0),
        reference_time_str="2023-04-12T12:00:00",
    )


def test_recently_resolves_to_past_7_days():
    constraint = extract_temporal_constraints("I saw her recently", _ctx())[0]
    assert constraint.resolution.status.value == "resolved"
    assert constraint.resolution.start.date().isoformat() == "2023-04-05"
    assert constraint.resolution.end.date().isoformat() == "2023-04-12"


def test_non_temporal_query_is_unchanged():
    rewritten, metadata = rewrite_temporal_text("What color was the car?", _ctx())
    assert rewritten == "What color was the car?"
    assert metadata["constraints"] == []
