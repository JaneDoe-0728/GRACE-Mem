from typing import TYPE_CHECKING, Any

from grace_mem.storage.cache import CacheStore, build_id_to_meta_maps

__all__ = ["ART_DIR", "MGR", "VDBManager", "CacheStore", "build_id_to_meta_maps"]

if TYPE_CHECKING:
    from grace_mem.storage.chroma_manager import ART_DIR, MGR, VDBManager


def __getattr__(name: str) -> Any:
    """Lazily expose storage singletons and manager types when requested."""
    if name in {"ART_DIR", "MGR", "VDBManager"}:
        from grace_mem.storage.chroma_manager import ART_DIR, MGR, VDBManager

        exports = {"ART_DIR": ART_DIR, "MGR": MGR, "VDBManager": VDBManager}
        return exports[name]
    raise AttributeError(f"module 'grace_mem.storage' has no attribute {name!r}")
