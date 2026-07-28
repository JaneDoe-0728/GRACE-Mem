# llm/prompts/config.py
"""
Common configuration for LLM prompts.
"""
from typing import Dict

EXTRA_KWARGS: Dict[str, str] = {
    # "language": "English",
    # "entity_types": "person, organization, location, geo, event, animal, food, product, category, unknown",
    "entity_types": "Person, Event, Date, Time, Timespan, Location, Organization, Product, Service, Activity, Topic, Concept",
    "tuple_delimiter": "<|>",
    "record_delimiter": "<|RECORD|>",
    "completion_delimiter": "<|COMPLETE|>",
    "dialogue_datetime": "{dialogue_datetime}",
    "temporal_hints": "(none)",
    "raw_conversation": "{raw_conversation}",
}
