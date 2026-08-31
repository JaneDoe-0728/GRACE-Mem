"""On-disk cache for the keyword-extraction LLM call.

The keyword-extraction model (served via vLLM/LM Studio) is not deterministic
even at temperature=0 with a fixed seed, which makes retrieval non-reproducible
across runs. Caching the result on disk, keyed by a hash of the exact prompt,
pins that variation down: the second run of a question replays the first run's
keywords rather than asking again.

That is the point of it. The cost saving -- one LLM call per question, and
evaluation re-asks the same questions across every ablation -- is a side
benefit.

This is storage, not retrieval. The Retriever asks for keywords and stores
them; it has no business knowing that they live in a JSON file, that the file
is written through a temporary and renamed, or that a corrupt file is treated
as a cold start.

Environment:
    KG_KEYWORD_CACHE_PATH     override the cache file; point several runs at one
                              shared file to reuse keywords across them
    KG_KEYWORD_CACHE_DISABLE  set to "1" to bypass the cache entirely

Both are read once, at import time, exactly as they were when this lived in
retriever.py.
"""

import hashlib
import json
import os
import threading

from grace_mem.domain.extraction import KeywordExtractionResult

# Repo root, three levels up from grace_mem/pipeline/. Kept as the original
# expression so the resolved path is unchanged; note that it depends on this
# module's depth, so moving it up or down a level moves the default cache file.
_DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".keyword_cache.json"
)


class KeywordCache:
    """Prompt hash -> extracted keywords, persisted as one JSON object.

    Entries are never evicted. The prompt hash covers the question and the
    guidance text, so a changed prompt template simply misses rather than
    returning stale keywords, and the file grows by one entry per distinct
    question the system has ever been asked.
    """

    def __init__(self, path: str | None = None, disabled: bool | None = None) -> None:
        self._path = path if path is not None else os.environ.get(
            "KG_KEYWORD_CACHE_PATH", _DEFAULT_CACHE_PATH
        )
        self._disabled = (
            disabled if disabled is not None
            else os.environ.get("KG_KEYWORD_CACHE_DISABLE", "") == "1"
        )
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, list[str]]] | None = None

    @staticmethod
    def _key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _load(self) -> dict[str, dict[str, list[str]]]:
        """Read the file on first use; treat absent or corrupt as a cold start.

        A corrupt cache is not an error worth raising: re-asking the LLM is
        always correct, just slower, while a raise here would take down a run
        over a file that exists only to make it faster.

        Caller holds the lock.
        """
        if self._entries is None:
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._entries = json.load(f)
            except (FileNotFoundError, ValueError):
                self._entries = {}
        return self._entries

    def get(self, prompt: str) -> KeywordExtractionResult | None:
        """Return the cached keywords for this prompt, or None on a miss."""
        if self._disabled:
            return None
        with self._lock:
            hit = self._load().get(self._key(prompt))
        if hit is None:
            return None
        return KeywordExtractionResult(
            high_level_keywords=list(hit.get("high_level_keywords", [])),
            low_level_keywords=list(hit.get("low_level_keywords", [])),
        )

    def put(self, prompt: str, result: KeywordExtractionResult) -> None:
        """Store this prompt's keywords and persist the whole cache.

        Written through a temporary file and renamed, because concurrent sample
        workers share one cache file and a half-written JSON object would be
        read back as a cold start by every one of them.

        A failed write is swallowed: the cache is an optimisation, and losing it
        must not fail the run that was only trying to fill it.
        """
        if self._disabled:
            return
        with self._lock:
            entries = self._load()
            entries[self._key(prompt)] = {
                "high_level_keywords": list(result.high_level_keywords),
                "low_level_keywords": list(result.low_level_keywords),
            }
            tmp = f"{self._path}.tmp"
            try:
                os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(entries, f, ensure_ascii=False)
                os.replace(tmp, self._path)
            except OSError:
                pass


# Process-wide instance. The cache was module-level state before this class
# existed, and retrieval depends on that sharing: every Retriever in a process
# must see the keywords the others already paid for.
keyword_cache = KeywordCache()
