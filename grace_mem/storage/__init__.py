"""Storage layer: vector stores, the extraction cache, and artifact paths.

The split between the two import styles below is deliberate. `cache` is
imported eagerly because it is pure stdlib and pickle. The chroma_manager
exports go through the lazy `__getattr__` because importing that module
constructs the MGR singleton, which opens a Chroma client -- a cost no caller
should pay just for touching `grace_mem.storage`.
"""

from typing import TYPE_CHECKING, Any

from grace_mem.storage.cache import CacheStore, build_id_to_meta_maps

__all__ = ["ARTIFACTS_DIR", "MGR", "CacheStore", "VDBManager", "build_id_to_meta_maps"]

if TYPE_CHECKING:
    from grace_mem.storage.chroma_manager import ARTIFACTS_DIR, MGR, VDBManager


def __getattr__(name: str) -> Any:
    """Lazily expose storage singletons and manager types when requested."""
    if name in {"ARTIFACTS_DIR", "MGR", "VDBManager"}:
        from grace_mem.storage.chroma_manager import ARTIFACTS_DIR, MGR, VDBManager

        exports = {"ARTIFACTS_DIR": ARTIFACTS_DIR, "MGR": MGR, "VDBManager": VDBManager}
        return exports[name]
    raise AttributeError(f"module 'grace_mem.storage' has no attribute {name!r}")
