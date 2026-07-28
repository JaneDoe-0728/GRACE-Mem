"""Single source of truth for the working artifacts directory.

The KG storage layer (Chroma indices, BM25, cache) and the experiment
snapshot/backup helpers all resolve the working artifacts dir through here.
This is what makes parallel ingest safe: two processes that share the default
``KG/storage/artifacts`` path would interleave their Chroma writes and cross-
contaminate each other's VDB (e.g. sample_0's vectors leaking into sample_5).

To isolate a process, set ``KG_ARTIFACTS_DIR`` to a per-process path before the
process starts, e.g. ``KG_ARTIFACTS_DIR=KG/storage/artifacts_sample5``. Keep the
value stable across the ingest/retrieval stages of the *same* logical run so the
later stage can read what the earlier stage wrote — key it by sample/worker,
not by PID (PID changes between stages).
"""

from __future__ import annotations

import os
from pathlib import Path

# Default working dir: KG/storage/artifacts (this file lives in KG/storage/).
_DEFAULT_ART_DIR = Path(__file__).resolve().parent / "artifacts"

# Env var that overrides the working artifacts directory (per-process isolation).
ARTIFACTS_DIR_ENV = "KG_ARTIFACTS_DIR"


def resolve_artifacts_dir(*, create: bool = False) -> Path:
    """Resolve the working artifacts directory, honoring ``KG_ARTIFACTS_DIR``.

    Returns an absolute path. When ``create`` is True the directory (and its
    parents) is created if missing.
    """
    override = os.environ.get(ARTIFACTS_DIR_ENV, "").strip()
    art_dir = (Path(override).expanduser() if override else _DEFAULT_ART_DIR).resolve()
    if create:
        art_dir.mkdir(parents=True, exist_ok=True)
    return art_dir
