import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from KG.utils.query_time_parser import detect_and_parse_time_expressions


def test_last_night_rewrites_to_previous_date():
    rewritten, info = detect_and_parse_time_expressions(
        "My mom and I made dinner together last night!",
        query_time="3:18 pm on 4 May, 2023",
        rewrite_query=True,
    )

    assert "on 2023-05-03" in rewritten
    assert info["detected_expressions"][0]["original"] == "last night"
    assert info["detected_expressions"][0]["absolute_date"] == "2023-05-03"


def test_yesterday_evening_rewrites_to_previous_date():
    rewritten, info = detect_and_parse_time_expressions(
        "We had drinks yesterday evening.",
        query_time="4:12 pm on 22 February, 2023",
        rewrite_query=True,
    )

    assert "on 2023-02-21" in rewritten
    assert info["detected_expressions"][0]["original"] == "yesterday evening"


def test_last_weekend_rewrites_to_weekend_range():
    rewritten, info = detect_and_parse_time_expressions(
        "Last weekend I joined a mentorship program.",
        query_time="2:31 pm on 17 July, 2023",
        rewrite_query=True,
    )

    assert "during 2023-07-15 to 2023-07-16" in rewritten
    assert info["detected_expressions"][0]["absolute_date"] == "2023-07-15 to 2023-07-16"


def test_two_weekends_ago_rewrites_to_weekend_range():
    rewritten, info = detect_and_parse_time_expressions(
        "We went camping two weekends ago.",
        query_time="2:31 pm on 17 July, 2023",
        rewrite_query=True,
    )

    assert "during 2023-07-08 to 2023-07-09" in rewritten
    assert info["detected_expressions"][0]["absolute_date"] == "2023-07-08 to 2023-07-09"


def test_existing_yesterday_rewrite_still_works():
    rewritten, info = detect_and_parse_time_expressions(
        "Who did I meet yesterday?",
        query_time="8:18 pm on 6 July, 2023",
        rewrite_query=True,
    )

    assert "on 2023-07-05" in rewritten
    assert info["detected_expressions"][0]["original"] == "yesterday"


def test_recently_rewrites_to_last_seven_days():
    rewritten, info = detect_and_parse_time_expressions(
        "What did I do recently?",
        query_time="2023/03/09 (Thu) 15:47",
        rewrite_query=True,
    )

    assert "during 2023-03-02 to 2023-03-09" in rewritten
    assert info["detected_expressions"][0]["absolute_date"] == "2023-03-02 to 2023-03-09"


def test_lately_rewrites_to_last_seven_days():
    rewritten, info = detect_and_parse_time_expressions(
        "What happened lately?",
        query_time="2023/03/09 (Thu) 15:47",
        rewrite_query=True,
    )

    assert "during 2023-03-02 to 2023-03-09" in rewritten
    assert info["detected_expressions"][0]["original"] == "lately"


def test_last_week_rewrites_to_calendar_week_range():
    rewritten, info = detect_and_parse_time_expressions(
        "What happened last week?",
        query_time="2023/03/09 (Thu) 15:47",
        rewrite_query=True,
    )

    assert "during 2023-02-27 to 2023-03-05" in rewritten
    assert info["detected_expressions"][0]["absolute_date"] == "2023-02-27 to 2023-03-05"


def test_two_weeks_ago_rewrites_to_calendar_week_range():
    rewritten, info = detect_and_parse_time_expressions(
        "What happened two weeks ago?",
        query_time="2023/03/09 (Thu) 15:47",
        rewrite_query=True,
    )

    assert "during 2023-02-20 to 2023-02-26" in rewritten
    assert info["detected_expressions"][0]["absolute_date"] == "2023-02-20 to 2023-02-26"


def test_last_month_rewrites_to_calendar_month_range():
    rewritten, info = detect_and_parse_time_expressions(
        "What happened last month?",
        query_time="2023/03/09 (Thu) 15:47",
        rewrite_query=True,
    )

    assert "during 2023-02-01 to 2023-02-28" in rewritten
    assert info["detected_expressions"][0]["absolute_date"] == "2023-02-01 to 2023-02-28"


def test_last_year_rewrites_to_calendar_year_range():
    rewritten, info = detect_and_parse_time_expressions(
        "What happened last year?",
        query_time="2023/03/09 (Thu) 15:47",
        rewrite_query=True,
    )

    assert "during 2022-01-01 to 2022-12-31" in rewritten
    assert info["detected_expressions"][0]["absolute_date"] == "2022-01-01 to 2022-12-31"


def test_fortnights_ago_rewrites_to_fourteen_day_offset():
    rewritten, info = detect_and_parse_time_expressions(
        "What happened two fortnights ago?",
        query_time="2023/03/09 (Thu) 15:47",
        rewrite_query=True,
    )

    assert "on 2023-02-09" in rewritten
    assert info["detected_expressions"][0]["absolute_date"] == "2023-02-09"


def test_last_quarter_rewrites_to_calendar_quarter_range():
    rewritten, info = detect_and_parse_time_expressions(
        "What happened last quarter?",
        query_time="2023/05/09 (Tue) 15:47",
        rewrite_query=True,
    )

    assert "during 2023-01-01 to 2023-03-31" in rewritten
    assert info["detected_expressions"][0]["absolute_date"] == "2023-01-01 to 2023-03-31"
