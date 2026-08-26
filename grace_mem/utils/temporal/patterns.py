"""Compiled English-only temporal regex patterns."""

from __future__ import annotations

import re

NUMBER_TOKEN = (
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
)
MONTH_NAME = (
    r"(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
)
# Full names must precede abbreviations so longer alternatives match first.
WEEKDAY_NAME = (
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|mon|tues?|wed|thu(?:rs?)?|fri|sat|sun)"
)

ABSOLUTE_DATE_TEXT = (
    rf"(?:\d{{4}}[-/]\d{{1,2}}[-/]\d{{1,2}}|"
    rf"{MONTH_NAME}\s+\d{{1,2}},?\s+\d{{4}}|"
    rf"\d{{1,2}}\s+{MONTH_NAME}\s+\d{{4}})"
)
ABSOLUTE_TIME_TEXT = r"(?:at\s+)?(?:\d{1,2}:\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm))"
# Compound daypart phrases must precede bare day tokens so they win the alternation.
RELATIVE_DAYPART_TEXT = r"(?:yesterday\s+(?:morning|afternoon|evening|night)|tomorrow\s+(?:morning|afternoon|evening)|last\s+night)"
RELATIVE_DAY_TEXT = (
    rf"(?:this\s+(?:morning|afternoon|evening)"
    r"|last\s+night"
    r"|tonight"
    r"|today|yesterday|tomorrow)"
)
DAYPART_TIME_TEXT = r"(?:this\s+morning|this\s+afternoon|this\s+evening|tonight|morning|afternoon|evening|night)"
RELATIVE_WEEKDAY_TEXT = rf"(?:last|this|next)\s+{WEEKDAY_NAME}"
AGO_TEXT = rf"{NUMBER_TOKEN}\s+(?:day|week|month|year)s?\s+ago"
RELATIVE_HOUR_TEXT = rf"(?:in\s+{NUMBER_TOKEN}\s+hours?|{NUMBER_TOKEN}\s+hours?\s+ago)"
IN_LAST_TEXT = rf"in\s+the\s+last\s+{NUMBER_TOKEN}\s+(?:day|week|month|year)s?"
WEEK_POINT_TEXT = (
    rf"(?:"
    rf"(?:last|this|next)\s+week"
    rf"|the\s+week\s+(?:before|after|of)\s+"
    rf"(?:\d{{4}}[-/]\d{{1,2}}[-/]\d{{1,2}}|{MONTH_NAME}\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}\s+{MONTH_NAME}\s+\d{{4}})"
    rf")"
)
WEEKEND_TEXT = r"(?:(?:this\s+past|last|this|next)\s+weekend)"
MONTH_POINT_TEXT = rf"(?:last|this|next)\s+month|{MONTH_NAME}\s+\d{{4}}"
SEASON_NAME = r"(?:spring|summer|fall|autumn|winter)"
SEASON_POINT_TEXT = rf"(?:last|this|next)\s+{SEASON_NAME}|{SEASON_NAME}\s+\d{{4}}"
YEAR_POINT_TEXT = r"(?:last|this|next)\s+year"
MONTH_WEEK_RANGE_TEXT = rf"(?:the\s+)?(?:first|last)\s+week\s+of\s+{MONTH_NAME}\s+\d{{4}}"
FUZZY_TEXT = r"(?:recently|lately|the\s+other\s+day|a\s+while\s+ago|a\s+few\s+days?\s+ago|a\s+few\s+weeks?\s+ago|a\s+few\s+years?\s+ago)"

TIME_EXPR_INNER_TEXT = (
    rf"(?:{MONTH_WEEK_RANGE_TEXT}|{ABSOLUTE_DATE_TEXT}|{ABSOLUTE_TIME_TEXT}|{RELATIVE_HOUR_TEXT}|{RELATIVE_DAYPART_TEXT}|{DAYPART_TIME_TEXT}|{IN_LAST_TEXT}|{AGO_TEXT}|"
    rf"{RELATIVE_WEEKDAY_TEXT}|{RELATIVE_DAY_TEXT}|{WEEKEND_TEXT}|{WEEK_POINT_TEXT}|"
    rf"{MONTH_POINT_TEXT}|{SEASON_POINT_TEXT}|{YEAR_POINT_TEXT})"
)
BOUNDARY_TEXT = rf"(?:before|after|since)\s+{TIME_EXPR_INNER_TEXT}"


def _compile(text: str) -> re.Pattern[str]:
    return re.compile(rf"\b{text}\b", re.IGNORECASE)


BOUNDARY_RE = _compile(BOUNDARY_TEXT)
MONTH_WEEK_RANGE_RE = _compile(MONTH_WEEK_RANGE_TEXT)
ABSOLUTE_DATE_RE = _compile(ABSOLUTE_DATE_TEXT)
ABSOLUTE_TIME_RE = _compile(ABSOLUTE_TIME_TEXT)
RELATIVE_DAYPART_RE = _compile(RELATIVE_DAYPART_TEXT)
RELATIVE_DAY_RE = _compile(RELATIVE_DAY_TEXT)
DAYPART_TIME_RE = _compile(DAYPART_TIME_TEXT)
RELATIVE_WEEKDAY_RE = _compile(RELATIVE_WEEKDAY_TEXT)
RELATIVE_HOUR_RE = _compile(RELATIVE_HOUR_TEXT)
IN_LAST_RE = _compile(IN_LAST_TEXT)
AGO_RE = _compile(AGO_TEXT)
WEEK_POINT_RE = _compile(WEEK_POINT_TEXT)
WEEKEND_RE = _compile(WEEKEND_TEXT)
MONTH_POINT_RE = _compile(MONTH_POINT_TEXT)
SEASON_POINT_RE = _compile(SEASON_POINT_TEXT)
YEAR_POINT_RE = _compile(YEAR_POINT_TEXT)
FUZZY_RE = _compile(FUZZY_TEXT)

BOUNDARY_PARSE_RE = re.compile(rf"^(before|after|since)\s+({TIME_EXPR_INNER_TEXT})$", re.IGNORECASE)
RELATIVE_DAYPART_PARSE_RE = re.compile(r"^(yesterday|tomorrow|last)\s+(morning|afternoon|evening|night)$", re.IGNORECASE)
RELATIVE_WEEKDAY_PARSE_RE = re.compile(rf"^(last|this|next)\s+({WEEKDAY_NAME})$", re.IGNORECASE)
AGO_PARSE_RE = re.compile(rf"^({NUMBER_TOKEN})\s+(day|week|month|year)s?\s+ago$", re.IGNORECASE)
RELATIVE_HOUR_PARSE_RE = re.compile(rf"^(?:in\s+({NUMBER_TOKEN})\s+hours?|({NUMBER_TOKEN})\s+hours?\s+ago)$", re.IGNORECASE)
DAYPART_TIME_PARSE_RE = re.compile(
    r"^(this\s+morning|this\s+afternoon|this\s+evening|tonight|morning|afternoon|evening|night)$",
    re.IGNORECASE,
)
IN_LAST_PARSE_RE = re.compile(
    rf"^in\s+the\s+last\s+({NUMBER_TOKEN})\s+(day|week|month|year)s?$",
    re.IGNORECASE,
)
WEEK_POINT_PARSE_RE = re.compile(
    rf"^(?:(last|this|next)\s+week|the\s+week\s+(before|after|of)\s+({ABSOLUTE_DATE_TEXT}))$",
    re.IGNORECASE,
)
WEEKEND_PARSE_RE = re.compile(r"^(this\s+past|last|this|next)\s+weekend$", re.IGNORECASE)
MONTH_POINT_PARSE_RE = re.compile(rf"^((?:last|this|next)\s+month|{MONTH_NAME}\s+\d{{4}})$", re.IGNORECASE)
SEASON_POINT_PARSE_RE = re.compile(rf"^((?:last|this|next)\s+{SEASON_NAME}|{SEASON_NAME}\s+\d{{4}})$", re.IGNORECASE)
MONTH_WEEK_RANGE_PARSE_RE = re.compile(
    rf"^(?:the\s+)?(first|last)\s+week\s+of\s+({MONTH_NAME})\s+(\d{{4}})$",
    re.IGNORECASE,
)
ABSOLUTE_DATE_ISO_PARSE_RE = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$")
ABSOLUTE_DATE_MONTH_FIRST_PARSE_RE = re.compile(
    rf"^({MONTH_NAME})\s+(\d{{1,2}}),?\s+(\d{{4}})$",
    re.IGNORECASE,
)
ABSOLUTE_DATE_DAY_FIRST_PARSE_RE = re.compile(
    rf"^(\d{{1,2}})\s+({MONTH_NAME})\s+(\d{{4}})$",
    re.IGNORECASE,
)
ABSOLUTE_TIME_PARSE_RE = re.compile(r"^(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", re.IGNORECASE)
