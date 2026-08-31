"""The data concepts GRACE-Mem stores, independent of how they are stored.

This package imports nothing else from grace_mem.
"""

from grace_mem.domain.entities import Entity, EntityType, canonical_entity_id
from grace_mem.domain.extraction import (
    ExtractionResult,
    KeywordExtractionResult,
    SCHEMA_keyword,
)
from grace_mem.domain.relationships import Relationship, canonical_rel_id

__all__ = [
    "SCHEMA_keyword",
    "Entity",
    "EntityType",
    "ExtractionResult",
    "KeywordExtractionResult",
    "Relationship",
    "canonical_entity_id",
    "canonical_rel_id",
]
