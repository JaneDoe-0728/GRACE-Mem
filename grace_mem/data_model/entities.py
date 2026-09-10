"""The Entity: one node in the knowledge graph, and the rules that identify it.

Pure data. Nothing here reaches for a graph, a vector store or an LLM -- a model
that needed infrastructure to exist could not be constructed in a test.
"""

import re
import unicodedata
from enum import Enum
from typing import Any

from pydantic import BaseModel


def _slugify(text: str) -> str:
    """Reduce an id fragment to a lowercase, separator-free key.

    `/` and `::` are replaced rather than kept because ids end up in graph node
    keys and filesystem paths, where both are structural characters.
    """
    return (text.strip().lower().replace(" ", "_").replace("/", "_").replace("::", "_"))

def canonical_entity_id(name: str, etype: str) -> str:
    """Derive an entity's stable id from its type and name.

    Deterministic rather than generated, which is what lets ingestion recognise
    a re-mention across turns and sessions without a lookup: the same person
    named the same way yields the same id in any process, in any run.
    """
    return _slugify(f"{etype}::{name}")

def _norm_name(s: str) -> str:
    """Normalize an entity name for identity comparison.

    NFKC folds the compatibility variants that arrive from copied text --
    fullwidth forms, ligatures -- onto their ASCII equivalents, so a name that
    looks identical on screen compares identical here. Whitespace runs collapse
    for the same reason.
    """
    s = unicodedata.normalize("NFKC", s).strip().lower()
    return re.sub(r"\s+", " ", s)

def _entity_key(name: str, etype: str) -> str:
    """Build the in-memory cache key for an entity.

    Distinct from `canonical_entity_id`: this one keeps the "::" separator and
    skips slugification, so it stays reversible for debugging. Do not persist
    it -- ids are the durable form.
    """
    return f"{_norm_name(name)}::{etype.lower()}"


class EntityType(str, Enum):
    """The closed set of entity types extraction may emit.

    Closed on purpose. A free-text type field let the model invent near
    synonyms ("person", "individual", "human") that split one real entity
    across several graph nodes. Inheriting from `str` keeps members usable
    directly as dict keys and in serialized metadata.

    Date/Time/Timespan are separate members rather than one "temporal" type
    because retrieval filters on granularity -- a query bounded to a day must
    not match a node spanning a year.
    """

    Person      = "Person"
    Event       = "Event"
    Date        = "Date"
    Time        = "Time"
    Timespan    = "Timespan"
    Location    = "Location"
    Organization= "Organization"
    Product     = "Product"
    Service     = "Service"
    Activity    = "Activity"
    Topic       = "Topic"
    Concept     = "Concept"

class Entity(BaseModel):
    """One node in the knowledge graph.

    Attributes:
        entity_description: Free text, and the field that is embedded for dense
            retrieval -- the name alone is too short to encode usefully. An
            entity with an empty description is effectively unsearchable.
        entity_metadata: Type-specific extras. Temporal entities carry their
            resolved value here; see `pipeline/ingestor.py`.
    """

    entity_name: str
    entity_type: EntityType
    entity_description: str
    entity_metadata: dict[str, Any] | None = None
