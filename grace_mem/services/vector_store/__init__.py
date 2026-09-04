"""Chroma vector stores and the manager that owns them for one run.

Exports resolve lazily through `__getattr__`: importing chroma_manager
constructs the MGR singleton, which opens a Chroma client -- a cost no caller
should pay just for touching this package.
"""

from typing import TYPE_CHECKING, Any

__all__ = ["ARTIFACTS_DIR", "MGR", "VDBManager"]

if TYPE_CHECKING:
    from grace_mem.services.vector_store.chroma_manager import ARTIFACTS_DIR, MGR, VDBManager


def __getattr__(name: str) -> Any:
    """Lazily expose the vector-store singletons and manager type."""
    if name in {"ARTIFACTS_DIR", "MGR", "VDBManager"}:
        from grace_mem.services.vector_store.chroma_manager import ARTIFACTS_DIR, MGR, VDBManager

        exports = {"ARTIFACTS_DIR": ARTIFACTS_DIR, "MGR": MGR, "VDBManager": VDBManager}
        return exports[name]
    raise AttributeError(f"module 'grace_mem.services.vector_store' has no attribute {name!r}")
