"""The extraction prompt's type list must stay identical to EntityType.

`EXTRA_KWARGS["entity_types"]` is rendered from the enum rather than written out
by hand, so the enum is the only place an entity type is declared. That makes the
enum's *declaration order* part of the prompt text: reordering the members --
alphabetising them, say -- would rewrite every extraction prompt and silently
change model behaviour, which would invalidate comparisons with historical runs.

The literal below is the string as it stood before the value was derived. It is
pinned deliberately: this test exists to fail when the enum is reordered.
"""

from grace_mem.llm.prompts.config import EXTRA_KWARGS
from grace_mem.utils.common import EntityType

# The exact string the prompt used before entity_types became a derived value.
FROZEN_ENTITY_TYPES = (
    "Person, Event, Date, Time, Timespan, Location, "
    "Organization, Product, Service, Activity, Topic, Concept"
)


def test_prompt_entity_types_match_the_frozen_string():
    assert EXTRA_KWARGS["entity_types"] == FROZEN_ENTITY_TYPES


def test_prompt_entity_types_are_derived_from_the_enum():
    assert EXTRA_KWARGS["entity_types"] == ", ".join(t.value for t in EntityType)


def test_every_entity_type_appears_exactly_once():
    rendered = [part.strip() for part in EXTRA_KWARGS["entity_types"].split(",")]

    assert rendered == [t.value for t in EntityType]
    assert len(set(rendered)) == len(rendered)
