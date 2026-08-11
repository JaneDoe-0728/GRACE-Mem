from __future__ import annotations

from datetime import datetime
import sys
import types

import pytest

if "nltk" not in sys.modules:
    nltk_stub = types.ModuleType("nltk")
    nltk_stub.word_tokenize = lambda text: text.split()
    nltk_stub.pos_tag = lambda toks: [(tok, "NN") for tok in toks]
    sys.modules["nltk"] = nltk_stub

from KG.pipeline.ingestor import _repair_temporal_entities
from KG.utils.common import Entity, EntityType
from KG.utils.temporal import augment_temporal_text, build_time_context


def _ctx():
    return build_time_context(
        reference_dt=datetime(2023, 7, 6, 10, 0, 0),
        reference_time_str="2023/07/06 10:00",
        source="tests",
    )


def test_augment_rewrites_relative_date_in_place():
    aug, hints, _ = augment_temporal_text("Yesterday I took my kids to the museum", _ctx())
    assert "[DATE: 2023-07-05]" in aug
    assert "Yesterday" not in aug
    assert hints[0]["markers"] == [{"entity_type": "Date", "marker_type": "DATE", "entity_name": "2023-07-05"}]
    assert hints[0]["resolved_to"] == "2023-07-05"


def test_augment_rewrites_relative_hour_to_hhmm():
    aug, hints, _ = augment_temporal_text("Let's talk in 2 hours", _ctx())
    assert "[TIME: 2023-07-06T12:00]" in aug
    assert hints[0]["resolved_to"] == "12:00"
    assert hints[0]["normalized_time"] == "12:00"


def test_augment_rewrites_daypart_to_configured_hhmm():
    ctx = build_time_context(
        reference_dt=datetime(2023, 7, 6, 10, 0, 0),
        reference_time_str="2023/07/06 10:00",
        source="tests",
        daypart_anchor_times={"tonight": "22:30"},
    )
    aug, hints, _ = augment_temporal_text("Dinner is tonight", ctx)
    assert "[TIMESPAN: night of 2023-07-06]" in aug
    assert hints[0]["resolved_to"] == "night of 2023-07-06"


def test_augment_rewrites_bare_night_to_timespan_marker():
    aug, hints, _ = augment_temporal_text("Dinner is night", _ctx())
    assert "[TIMESPAN: night of 2023-07-06]" in aug
    assert hints[0]["resolved_to"] == "night of 2023-07-06"


def test_augment_rewrites_tomorrow_morning_with_date_and_time():
    aug, hints, _ = augment_temporal_text("Let's meet tomorrow morning", _ctx())
    assert "[TIMESPAN: morning of 2023-07-07]" in aug
    assert hints[0]["resolved_to"] == "morning of 2023-07-07"
    assert hints[0]["normalized_start"] == "2023-07-07"
    assert hints[0]["normalized_time"] == "09:00"


def test_augment_rewrites_timespan_to_natural_display_value():
    aug, hints, _ = augment_temporal_text("last year we traveled a lot", _ctx())
    assert "2022-01-01 to 2022-12-31" not in aug
    assert "[TIMESPAN: 2022]" in aug
    assert hints[0]["resolved_to"] == "2022"
    assert hints[0]["normalized_start"] == "2022-01-01"
    assert hints[0]["normalized_end"] == "2022-12-31"


def test_augment_preserves_full_anchored_week_phrase():
    aug, hints, _ = augment_temporal_text("the week before 2023-06-09 was busy", _ctx())
    assert "[TIMESPAN: week of 2023-05-29]" in aug
    assert "[TIMESPAN: before 2023-06-09]" not in aug
    assert hints[0]["resolved_to"] == "week of 2023-05-29"
    assert hints[0]["normalized_start"] == "2023-05-29"
    assert hints[0]["normalized_end"] == "2023-06-04"


def test_augment_fuzzy_phrase_resolves():
    aug, hints, _ = augment_temporal_text("I saw him recently", _ctx())
    assert aug != "I saw him recently"
    assert len(hints) == 1
    assert hints[0]["original"] == "recently"
    assert hints[0]["resolved_to"] == "2023-06-29 to 2023-07-06"


def test_repair_date_entity_keeps_llm_description_when_present():
    hints = [{
        "original": "Yesterday",
        "resolved_to": "2023-07-05",
        "display_value": "2023-07-05",
        "normalized_start": "2023-07-05",
        "normalized_end": "2023-07-05",
        "granularity": "day",
        "reference_time": "2023/07/06 10:00",
        "status": "resolved",
        "confidence": "high",
    }]
    ents, _ = _repair_temporal_entities(
        [Entity(entity_name="Yesterday", entity_type=EntityType.Date, entity_description="Day user visited museum")],
        [],
        hints,
        _ctx(),
    )
    assert ents[0].entity_name == "2023-07-05"
    assert ents[0].entity_description == "Day user visited museum"
    assert ents[0].entity_metadata["temporal"]["normalized_start"] == "2023-07-05"


def test_repair_date_entity_falls_back_to_generic_description_when_empty():
    hints = [{
        "original": "Yesterday",
        "resolved_to": "2023-07-05",
        "display_value": "2023-07-05",
        "normalized_start": "2023-07-05",
        "normalized_end": "2023-07-05",
        "granularity": "day",
        "reference_time": "2023/07/06 10:00",
        "status": "resolved",
        "confidence": "high",
    }]
    ents, _ = _repair_temporal_entities(
        [Entity(entity_name="Yesterday", entity_type=EntityType.Date, entity_description=" ")],
        [],
        hints,
        _ctx(),
    )
    assert ents[0].entity_name == "2023-07-05"
    assert ents[0].entity_description == "The calendar date 2023-07-05."


def test_repair_time_entity_keeps_llm_description_when_present():
    hints = [{
        "original": "at 9 pm",
        "resolved_to": "21:00",
        "display_value": "21:00",
        "normalized_time": "21:00",
        "normalized_start": "2023-07-06",
        "normalized_end": "2023-07-06",
        "granularity": "time",
        "reference_time": "2023/07/06 10:00",
        "status": "resolved",
        "confidence": "high",
    }]
    ents, _ = _repair_temporal_entities(
        [Entity(entity_name="at 9 pm", entity_type=EntityType.Time, entity_description="When dinner happens")],
        [],
        hints,
        _ctx(),
    )
    assert ents[0].entity_name == "21:00"
    assert ents[0].entity_description == "When dinner happens"
    assert ents[0].entity_metadata["temporal"]["normalized_time"] == "21:00"


def test_repair_daypart_timespan_keeps_llm_description_when_present():
    hints = [{
        "original": "tomorrow morning",
        "resolved_to": "morning of 2023-07-07",
        "display_value": "morning of 2023-07-07",
        "normalized_time": "09:00",
        "normalized_start": "2023-07-07",
        "normalized_end": "2023-07-07",
        "granularity": "range",
        "reference_time": "2023/07/06 10:00",
        "status": "resolved",
        "confidence": "high",
    }]
    ents, _ = _repair_temporal_entities(
        [Entity(entity_name="tomorrow morning", entity_type=EntityType.Timespan, entity_description="When the meeting happens")],
        [],
        hints,
        _ctx(),
    )
    assert ents[0].entity_name == "morning of 2023-07-07"
    assert ents[0].entity_description == "When the meeting happens"


def test_repair_timespan_entity_keeps_llm_description_and_range_in_metadata_only():
    hints = [{
        "original": "last year",
        "resolved_to": "2022",
        "display_value": "2022",
        "normalized_start": "2022-01-01",
        "normalized_end": "2022-12-31",
        "granularity": "year",
        "reference_time": "2023/07/06 10:00",
        "status": "resolved",
        "confidence": "high",
    }]
    ents, _ = _repair_temporal_entities(
        [Entity(entity_name="last year", entity_type=EntityType.Timespan, entity_description="Period discussed")],
        [],
        hints,
        _ctx(),
    )
    assert ents[0].entity_name == "2022"
    assert ents[0].entity_description == "Period discussed"
    assert "2022-01-01 to 2022-12-31" not in ents[0].entity_description
    assert ents[0].entity_metadata["temporal"]["normalized_end"] == "2022-12-31"


def test_repair_event_strips_temporal_phrase_and_rewrites_description_date():
    hints = [{"original": "last Friday", "resolved_to": "2023-06-30"}]
    ents, _ = _repair_temporal_entities(
        [Entity(entity_name="last Friday pottery workshop", entity_type=EntityType.Event, entity_description="A pottery class last Friday")],
        [],
        hints,
        _ctx(),
    )
    assert ents[0].entity_name == "pottery workshop"
    assert "2023-06-30" in ents[0].entity_description


@pytest.mark.parametrize("phrase,resolved,etype", [
    ("Yesterday", "2023-07-05", EntityType.Date),
    ("last year", "2022", EntityType.Timespan),
])
def test_no_canonical_entity_named_relative_phrase(phrase, resolved, etype):
    hints = [{"original": phrase, "resolved_to": resolved, "display_value": resolved}]
    ents, _ = _repair_temporal_entities(
        [Entity(entity_name=phrase, entity_type=etype, entity_description="Some description")],
        [],
        hints,
        _ctx(),
    )
    assert ents[0].entity_name == resolved


def test_prompt_instructs_natural_timespan_names():
    from KG.llm.prompts.extraction.two_step import entity_extraction_only

    prompt = entity_extraction_only["entity_extraction"]
    assert "Marker tags are authoritative temporal anchors" in prompt
    assert "Do not compute, rename, normalize, or paraphrase marker payloads." in prompt
    assert "Never replace an Event with a temporal entity. If an event and marker appear together, output BOTH." in prompt
    assert "[DATE: ...]" in prompt
    assert "[TIME: ...]" in prompt
    assert "[TIMESPAN: ...]" in prompt
    assert "Timespan entity whose entity_name is the marker payload exactly as written." in prompt
    assert "[RESOLVED_DATE:" not in prompt
