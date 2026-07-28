from typing import TYPE_CHECKING, Any

__all__ = ["EntityManager", "RelationshipManager", "Provenance"]

if TYPE_CHECKING:
    from KG.services.entity_manager import EntityManager
    from KG.services.relationship_manager import RelationshipManager
    from KG.services.provenance import Provenance


def __getattr__(name: str) -> Any:
    """Lazily import service-layer classes to avoid eager dependency loading."""
    if name == "EntityManager":
        from KG.services.entity_manager import EntityManager

        return EntityManager
    if name == "RelationshipManager":
        from KG.services.relationship_manager import RelationshipManager

        return RelationshipManager
    if name == "Provenance":
        from KG.services.provenance import Provenance

        return Provenance
    raise AttributeError(f"module 'KG.services' has no attribute {name!r}")
