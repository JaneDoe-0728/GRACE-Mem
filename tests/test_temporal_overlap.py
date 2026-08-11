from __future__ import annotations

from datetime import datetime

from KG.utils.temporal import build_time_context, extract_temporal_constraints, rewrite_temporal_text


def _ctx():
    return build_time_context(
        reference_dt=datetime(2023, 4, 12, 12, 0, 0),
        reference_time_str="2023-04-12T12:00:00",
    )


def test_overlapping_boundary_expression_is_single_constraint():
    constraints = extract_temporal_constraints("What happened before last Friday?", _ctx())
    assert len(constraints) == 1
    assert constraints[0].operator == "before"
    assert constraints[0].original_text == "before last Friday"


def test_boundary_rewrite_does_not_double_rewrite_nested_expression():
    rewritten, metadata = rewrite_temporal_text("What happened before last Friday?", _ctx())
    assert rewritten == "What happened before 2023-04-07?"
    assert len(metadata["constraints"]) == 1
