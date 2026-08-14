"""Vector top-up search from the gap description -- the second weapon in the
sufficiency repair arm.

The grep repair arm comes back empty about 87% of the time, and the root cause is
the paraphrase gap: the information the verifier flags as missing is often not a
literal span anywhere in the corpus. So here the question plus the missing
description is embedded and used to search that question's summaries VDB (the
:u/:a split produced by rebuild_split_summaries.py), pulling back the semantic
neighbours grep cannot reach and handing them to the agent to confirm.

Both the VDB client and the embedder are lazily cached globally (the embedder
takes 2-3GB of GPU and is loaded only once).
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

_lock = threading.Lock()
_vdb_cache: dict[str, object] = {}
_embed_fn = None


def _get_embed():
    global _embed_fn
    if _embed_fn is None:
        from grace_mem.embeddings import embedder  # lazy: loads qwen3-0.6b onto the GPU
        _embed_fn = embedder.embed
    return _embed_fn


def _get_vdb(artifact_dir: Path):
    key = str(artifact_dir)
    with _lock:
        if key not in _vdb_cache:
            from grace_mem.storage.chroma_vdb import SummariesVDB
            _vdb_cache[key] = SummariesVDB(
                dim=1024,
                path=str(Path(artifact_dir) / "summaries_chroma"),
                collection_name="summaries",
            )
        return _vdb_cache[key]


def vector_gap_candidates(
    artifact_dir: str | Path,
    query_text: str,
    *,
    exclude: set[str],
    topn: int = 6,
    min_score: float = 0.30,
) -> list[tuple[str, float]]:
    """Return [(sid, score)] with the sids in `exclude` removed. Any failure returns
    an empty list."""
    try:
        artifact_dir = Path(artifact_dir)
        if not (artifact_dir / "summaries_chroma").exists():
            return []
        vec = _get_embed()([query_text])
        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim == 2:
            vec = vec[0]
        hits = _get_vdb(artifact_dir).search(vec, top_k=topn * 3, threshold=min_score)
        out: list[tuple[str, float]] = []
        for meta, score in hits:
            sid = str(meta.get("id") or "").strip()
            if not sid or sid in exclude:
                continue
            out.append((sid, float(score)))
            if len(out) >= topn:
                break
        return out
    except Exception:
        return []
