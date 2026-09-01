"""Semantic search over a question's summaries VDB.

It exists for the paraphrase gap: what the question asks for is often not a
literal span anywhere in the corpus, so GREP cannot reach it. The VECTOR
command, which the agent drives itself, is its one caller.

The VDB holds the :u/:a split summaries produced by rebuild_split_summaries.py.
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
        from grace_mem.adapters.embedding.embeddings import (
            embedder,  # lazy: loads qwen3-0.6b onto the GPU
        )
        _embed_fn = embedder.embed
    return _embed_fn


def _get_vdb(artifact_dir: Path):
    key = str(artifact_dir)
    with _lock:
        if key not in _vdb_cache:
            from grace_mem.adapters.vector_store.chroma_vdb import SummariesVDB
            _vdb_cache[key] = SummariesVDB(
                dim=1024,
                path=str(Path(artifact_dir) / "summaries_chroma"),
                collection_name="summaries",
            )
        return _vdb_cache[key]


def search_summaries(
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


def render_hits(corpus, query: str, hits: list[tuple[str, float]]) -> str:
    """Render VECTOR hits as the inline candidate list the agent reads back.

    Same shape as a GREP result, so the agent needs no second format, and each
    hit still carries its score.
    """
    if not hits:
        return (f"vector {query!r}: 0 hits above threshold. "
                "Try rephrasing the query, or fall back to GREP with rare literal words.")
    lines = [(f"vector {query!r}: {len(hits)} semantically similar turns "
             "(NOT verified — check with READ/GREP before including)")]
    for sid, score in hits:
        turns = corpus.resolve(sid)
        entry = corpus.display_entry(sid, max_chars=200) or "(text unavailable)"
        dt = f"[{turns[0].date}] " if turns and turns[0].date else ""
        lines.append(f"[sid={sid}] (score={score:.2f}) {dt}{entry}")
    return "\n".join(lines)
