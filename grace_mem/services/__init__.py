from typing import TYPE_CHECKING, Any

__all__ = ["EntityManager", "RelationshipManager", "Provenance"]

if TYPE_CHECKING:
    from grace_mem.services.entity_manager import EntityManager
    from grace_mem.services.relationship_manager import RelationshipManager
    from grace_mem.services.provenance import Provenance


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
