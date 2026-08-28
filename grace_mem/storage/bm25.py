"""Lexical (BM25) half of the hybrid entity search.

Dense retrieval alone misses exact-match queries -- a rare proper noun the
encoder never saw well gets a mediocre cosine score, while BM25 ranks it first
because it is a rare term. This index supplies that lexical signal, and the
retriever fuses the two rankings with RRF.

Names and descriptions are indexed separately rather than as one concatenated
document. BM25 normalizes by document length, so folding a long description in
with a two-word name would let description terms dominate the score for a query
that was actually naming an entity.
"""

import pickle
import threading
from typing import Any

from rank_bm25 import BM25Okapi


class EntitiesBM25:
    """Parallel BM25 indexes over entity names and descriptions.

    Three lists are maintained in lockstep -- name tokens, description tokens,
    and metadata -- and share a positional index with the vector store. That
    shared ordering is the contract this class exists to keep: a BM25 score at
    position i and a vector score at position i must describe the same entity,
    or fusion silently blends unrelated candidates. Every mutation therefore
    appends to all three under one lock.
    """
    def __init__(self) -> None:
        """Initialize BM25 indexes for entity names, descriptions, and metadata."""
        self._lock = threading.Lock()
        self._docs_name: list[list[str]] = []
        self._docs_desc: list[list[str]] = []
        self._bm25_name: BM25Okapi | None = None
        self._bm25_desc: BM25Okapi | None = None
        self._metas: list[dict[str, Any]] = []

    @property
    def size(self) -> int:
        """Return the number of indexed entity metadata rows."""
        return len(self._metas)

    @property
    def metas(self) -> list[dict[str, Any]]:
        """Expose the metadata list aligned with the BM25 document order."""
        return self._metas

    def build(self) -> None:
        """Rebuild both BM25 indexes from the stored token lists.

        The empty-corpus fallback of `BM25Okapi([[]])` is load-bearing:
        BM25Okapi rejects a genuinely empty document list, and without a usable
        index `get_scores` would raise on a query against a cold store rather
        than returning zeros.
        """
        self._bm25_name = BM25Okapi(self._docs_name) if self._docs_name else BM25Okapi([[]])
        self._bm25_desc = BM25Okapi(self._docs_desc) if self._docs_desc else BM25Okapi([[]])

    def add(self, tokens_name: list[str], tokens_desc: list[str], meta: dict[str, Any]) -> None:
        """Append one entity and rebuild both indexes.

        Rebuilding on every insert is O(n) per add, quadratic over a full
        ingest. It is accepted because BM25 scores depend on corpus-wide IDF
        and average document length, and rank_bm25 exposes no incremental
        update -- appending a document without recomputing those would leave
        every previously indexed term scored against a stale corpus. Ingestion
        is the batch phase and retrieval is the hot one, so the cost lands
        where latency is not measured.
        """
        with self._lock:
            self._docs_name.append(tokens_name)
            self._docs_desc.append(tokens_desc)
            self._metas.append(meta)
            self.build()

    def get_scores(self, q_tokens: list[str]) -> tuple[list[float], list[float]]:
        """Score a tokenized query against both indexes.

        Args:
            q_tokens: Query tokens, tokenized the same way the documents were.
                A mismatch here degrades silently -- no error, just no hits.

        Returns:
            (name_scores, desc_scores), each positionally aligned with `metas`.
            Raw BM25 scores, not normalized: they are only ever compared within
            one ranking, and the fusion step consumes ranks rather than values.
        """
        if not self._bm25_name or not self._bm25_desc:
            self.build()
        assert self._bm25_name is not None
        assert self._bm25_desc is not None
        name_scores = self._bm25_name.get_scores(q_tokens)
        desc_scores = self._bm25_desc.get_scores(q_tokens)
        return name_scores, desc_scores

    # ---- persistence ----
    def save(self, path: str) -> None:
        """Pickle the token lists and metadata to `path`.

        The fitted BM25Okapi objects are deliberately not persisted -- they are
        derived state, and `load` reconstructs them via `build`. Storing only
        the inputs keeps the file readable by a different rank_bm25 version.
        """
        with self._lock, open(path, "wb") as f:
            pickle.dump({
                "docs_name": self._docs_name,
                "docs_desc": self._docs_desc,
                "metas": self._metas,
            }, f)

    def load(self, path: str) -> None:
        """Restore token lists and metadata from `path`, then refit the indexes.

        Replaces current state outright rather than merging, and takes no lock:
        loading into a store that is already serving queries would race. Load
        before the index is shared.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self._docs_name = obj.get("docs_name", [])
        self._docs_desc = obj.get("docs_desc", [])
        self._metas     = obj.get("metas", [])
        self.build()
