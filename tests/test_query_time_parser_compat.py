from __future__ import annotations

from KG.utils.query_time_parser import detect_and_parse_time_expressions, parse_query_time


def test_parse_query_time_preserves_supported_formats():
    assert parse_query_time("2023/03/09 (Thu) 15:47").isoformat() == "2023-03-09T15:47:00"
    assert parse_query_time("8:18 pm on 6 July, 2023").isoformat() == "2023-07-06T20:18:00"


def test_detect_and_parse_time_expressions_preserves_tuple_shape_and_keys():
    rewritten, info = detect_and_parse_time_expressions(
        "What happened last Friday?",
        query_time="2023/04/12 (Wed) 12:00",
        rewrite_query=True,
    )
    assert rewritten == "What happened on 2023-04-07?"
    assert "detected_expressions" in info
    assert "reference_time" in info
    assert "reference_time_str" in info
    assert "expressions_count" in info
    assert "constraints" in info
    assert info["detected_expressions"][0]["absolute_date"] == "2023-04-07"


def test_detect_and_parse_time_expressions_leaves_non_temporal_query_unchanged():
    rewritten, info = detect_and_parse_time_expressions(
        "What color was the car?",
        query_time="2023/04/12 (Wed) 12:00",
        rewrite_query=True,
    )
    assert rewritten == "What color was the car?"
    assert info["detected_expressions"] == []
