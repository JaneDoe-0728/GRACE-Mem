from __future__ import annotations

import sys
import types
from datetime import datetime


if "nltk" not in sys.modules:
    nltk_stub = types.ModuleType("nltk")
    nltk_stub.word_tokenize = lambda text: text.split()
    nltk_stub.pos_tag = lambda toks: [(tok, "NN") for tok in toks]
    sys.modules["nltk"] = nltk_stub

from KG.pipeline.ingestor import _repair_temporal_entities
from KG.utils.common import Entity, EntityType
from KG.utils.temporal import build_time_context


def _ctx():
    return build_time_context(
        reference_dt=datetime(2023, 5, 8, 13, 56, 0),
        reference_time_str="1:56 pm on 8 May, 2023",
        source="test",
    )


def test_date_entity_keeps_name_level_fallback_resolution():
    ents = [
        Entity(
            entity_name="Yesterday",
            entity_type=EntityType.Date,
            entity_description="The date when Caroline attended the LGBTQ support group.",
        )
    ]

    repaired, _ = _repair_temporal_entities(ents, [], [], _ctx())

    assert repaired[0].entity_name == "2023-05-07"


def test_timespan_entity_does_not_reparse_canonical_range_name():
    ents = [
        Entity(
            entity_name="2022",
            entity_type=EntityType.Timespan,
            entity_description="The period when Melanie painted the lake sunrise painting.",
        )
    ]

    repaired, _ = _repair_temporal_entities(ents, [], [], _ctx())

    assert repaired[0].entity_name == "2022"
