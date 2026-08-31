"""Resolving, merging and persisting what extraction produced.

Exports resolve lazily through `__getattr__`: importing these eagerly pulls in
the embedding model and the graph client, which is why a CLI that only wanted
--help used to wait on CUDA.
"""

from typing import TYPE_CHECKING, Any

__all__ = ["EntityManager", "RelationshipManager"]

if TYPE_CHECKING:
    from grace_mem.ingestion.managers.entity_manager import EntityManager
    from grace_mem.ingestion.managers.relationship_manager import RelationshipManager


def __getattr__(name: str) -> Any:
    """Lazily import manager classes to avoid eager dependency loading."""
    if name == "EntityManager":
        from grace_mem.ingestion.managers.entity_manager import EntityManager

        return EntityManager
    if name == "RelationshipManager":
        from grace_mem.ingestion.managers.relationship_manager import RelationshipManager

        return RelationshipManager
    raise AttributeError(f"module 'grace_mem.ingestion.managers' has no attribute {name!r}")
