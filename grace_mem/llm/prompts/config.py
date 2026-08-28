"""
Common configuration for LLM prompts.
"""

from grace_mem.utils.common import EntityType

EXTRA_KWARGS: dict[str, str] = {
    # "language": "English",
    # "entity_types": "person, organization, location, geo, event, animal, food, product, category, unknown",
    # Rendered from EntityType so the enum is the only place a type is declared.
    # Enum members keep declaration order, which is the order the prompt has
    # always used; tests/test_prompt_entity_types.py pins the rendered string so
    # reordering the enum cannot silently rewrite every extraction prompt.
    "entity_types": ", ".join(entity_type.value for entity_type in EntityType),
    "tuple_delimiter": "<|>",
    "record_delimiter": "<|RECORD|>",
    "completion_delimiter": "<|COMPLETE|>",
    "dialogue_datetime": "{dialogue_datetime}",
    "temporal_hints": "(none)",
    "raw_conversation": "{raw_conversation}",
}
