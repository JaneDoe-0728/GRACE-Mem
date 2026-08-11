"""Load the bundled NocoDB client without adding its directory to sys.path."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any


UPLOADER_ROOT = Path(__file__).resolve().parents[2] / "noco-db-uploader"
_PACKAGE_NAME = "_grace_mem_noco_uploader"


@lru_cache(maxsize=1)
def _load_src_package() -> ModuleType:
    package_name = f"{_PACKAGE_NAME}.src"
    package_dir = UPLOADER_ROOT / "src"
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load bundled NocoDB client from {package_dir}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(package_name, None)
        raise
    return module


def load_noco_client_class() -> type[Any]:
    """Return the bundled NocoDBClient class from an isolated namespace."""
    return _load_src_package().NocoDBClient
