"""Service layer: entity, relationship, and provenance management.

Exports are resolved lazily through the module-level `__getattr__` below.
Importing them eagerly would pull in the embedding model and the graph client
as a side effect of `import grace_mem.services`, which is why a CLI that only
wanted to print `--help` used to wait on CUDA initialization. The TYPE_CHECKING
block keeps type checkers and IDEs seeing the real names despite the
indirection.
"""

from typing import TYPE_CHECKING, Any

__all__ = ["EntityManager", "Provenance", "RelationshipManager"]

if TYPE_CHECKING:
    from grace_mem.services.entity_manager import EntityManager
    from grace_mem.services.provenance import Provenance
    from grace_mem.services.relationship_manager import RelationshipManager


def __getattr__(name: str) -> Any:
    """Lazily import service-layer classes to avoid eager dependency loading."""
    if name == "EntityManager":
        from grace_mem.services.entity_manager import EntityManager

        return EntityManager
    if name == "RelationshipManager":
        from grace_mem.services.relationship_manager import RelationshipManager

        return RelationshipManager
    if name == "Provenance":
        from grace_mem.services.provenance import Provenance

        return Provenance
    raise AttributeError(f"module 'grace_mem.services' has no attribute {name!r}")
