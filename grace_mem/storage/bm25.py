# storage/bm25.py
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import pickle, threading

class EntitiesBM25:
    """
    Keeps the name and desc BM25 indexes in step with each other, plus a metas
    list ordered identically to the VDB.
    """
    def __init__(self) -> None:
        """Initialize BM25 indexes for entity names, descriptions, and metadata."""
        self._lock = threading.Lock()
        self._docs_name: List[List[str]] = []
        self._docs_desc: List[List[str]] = []
        self._bm25_name: Optional[BM25Okapi] = None
        self._bm25_desc: Optional[BM25Okapi] = None
        self._metas: List[Dict[str, Any]] = []

    @property
    def size(self) -> int:
        """Return the number of indexed entity metadata rows."""
        return len(self._metas)

    @property
    def metas(self) -> List[Dict[str, Any]]:
        """Expose the metadata list aligned with the BM25 document order."""
        return self._metas

    def build(self) -> None:
        """Rebuild both BM25 indexes from the currently stored token lists."""
        # Called on first initialization, or on a bulk rebuild
        self._bm25_name = BM25Okapi(self._docs_name) if self._docs_name else BM25Okapi([[]])
        self._bm25_desc = BM25Okapi(self._docs_desc) if self._docs_desc else BM25Okapi([[]])

    def add(self, tokens_name: List[str], tokens_desc: List[str], meta: Dict[str, Any]) -> None:
        """Append one entity document and rebuild the BM25 indexes."""
        with self._lock:
            self._docs_name.append(tokens_name)
            self._docs_desc.append(tokens_desc)
            self._metas.append(meta)
            # rank_bm25 has no clean incremental IDF; the simple answer is to
            # rebuild after every add
            self.build()

    def get_scores(self, q_tokens: List[str]) -> tuple[list[float], list[float]]:
        """Score a tokenized query against name and description indexes."""
        if not self._bm25_name or not self._bm25_desc:
            self.build()
        name_scores = self._bm25_name.get_scores(q_tokens)
        desc_scores = self._bm25_desc.get_scores(q_tokens)
        return name_scores, desc_scores

    # ---- persistence ----
    def save(self, path: str) -> None:
        """Persist the BM25 token lists and metadata to disk."""
        with self._lock, open(path, "wb") as f:
            pickle.dump({
                "docs_name": self._docs_name,
                "docs_desc": self._docs_desc,
                "metas": self._metas,
            }, f)

    def load(self, path: str) -> None:
        """Restore BM25 token lists and metadata from disk, then rebuild indexes."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self._docs_name = obj.get("docs_name", [])
        self._docs_desc = obj.get("docs_desc", [])
        self._metas     = obj.get("metas", [])
        self.build()
