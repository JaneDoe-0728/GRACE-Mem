"""Extracting the search keywords a question should be looked up by, and the
on-disk cache that pins them down.

One LLM call turns the question into two lists: high-level words for abstract
reasoning and low-level anchors naming concrete entities and topics. Retrieval
uses them differently -- the low-level list drives lexical relationship search,
the high-level list is the baseline -- which is why they are produced together
and kept apart.

The call is cached on disk. That is about reproducibility before cost: the
extraction model is nondeterministic even at temperature 0 with a fixed seed,
so an uncached second run of the same question would retrieve differently for
no reason. The cost saving -- one LLM call per question, and evaluation re-asks
the same questions across every ablation -- is a side benefit.

Constrained decoding does the schema enforcement; the retries here are for the
model returning something unparseable anyway.

The cache half is storage, not retrieval: the caller asks for keywords and gets
them, with no business knowing that they live in a JSON file, that the file is
written through a temporary and renamed, or that a corrupt file is treated as a
cold start. It lives here rather than in its own module because
`generate_query_keywords` is its only caller.

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

from grace_mem.data_model.extraction import KeywordExtractionResult
from grace_mem.retrieval.prompts.keyword.extraction import KEYWORD_EXTRACTION_PROMPT
from grace_mem.utils.logger_config import _StepTimer, make_module_jlog, setup_logger

_jlog = make_module_jlog(name="grace_mem.Retriever", filename="kg_retriever.jsonl")
logger = setup_logger("grace_mem.Retriever")

# Repo root. This module sits at grace_mem/retrieval/steps/, so the root is four
# levels up -- the count is the module's own depth and has to be corrected
# whenever the file moves, or the default cache file moves with it and every
# previously cached question turns into a miss.
_DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
    ".keyword_cache.json",
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


def generate_query_keywords(*, llm, question: str, request_id: str | None = None,
                            max_retries: int = 5,
                            retrieval_guidance: str | None = None) -> KeywordExtractionResult:
    """
    Extract local/global keywords from query.
    Retries up to max_retries times only if the LLM output is unparseable.
    Empty or partial keyword lists are allowed so retrieval can continue
    with whichever signals are available.
    """
    import re as _re
    timer = _StepTimer()
    guidance_section = ""
    if retrieval_guidance:
        guidance_section = f"\nRetrieval guidance:\n{retrieval_guidance}\n"
    keyword_prompt = KEYWORD_EXTRACTION_PROMPT.format(
        query=question, guidance_section=guidance_section
    )
    last_error = ""
    js = ""

    _jlog(
        "generate_query_keywords_start",
        request_id,
        step="1",
        question=question,
        max_retries=max_retries,
    )

    # Reproducibility: return cached keywords if this exact prompt was seen.
    cached = keyword_cache.get(keyword_prompt)
    if cached is not None:
        _jlog(
            "generate_keywords_cache_hit",
            request_id,
            step="1",
            high_level_count=len(cached.high_level_keywords),
            low_level_count=len(cached.low_level_keywords),
            high_level_keywords=cached.high_level_keywords,
            low_level_keywords=cached.low_level_keywords,
            elapsed_sec=timer.sec(),
        )
        return cached

    for attempt in range(1, max_retries + 1):
        try:
            _jlog(
                "generate_keywords_attempt_start",
                request_id,
                step="1",
                attempt=attempt,
            )
            js, sec = llm.generate_llm_keyword(keyword_prompt)
            _jlog(
                "generate_keywords_llm_done",
                request_id,
                step="1",
                attempt=attempt,
                latency_sec=sec,
            )

            # Strip <think>...</think> or any prose before the JSON object
            m = _re.search(r'\{.*\}', js, _re.DOTALL)
            if m:
                js = m.group(0)

            res = KeywordExtractionResult.model_validate_json(js)

            if not res.high_level_keywords and not res.low_level_keywords:
                _jlog(
                    "generate_keywords_empty",
                    request_id,
                    step="1",
                    attempt=attempt,
                    high_level_count=len(res.high_level_keywords),
                    low_level_count=len(res.low_level_keywords),
                )
                if attempt < max_retries:
                    continue
            elif not res.high_level_keywords or not res.low_level_keywords:
                _jlog(
                    "generate_keywords_partial",
                    request_id,
                    step="1",
                    attempt=attempt,
                    high_level_count=len(res.high_level_keywords),
                    low_level_count=len(res.low_level_keywords),
                )

            _jlog(
                "generate_query_keywords_result",
                request_id,
                step="1",
                attempt=attempt,
                high_level_count=len(res.high_level_keywords),
                low_level_count=len(res.low_level_keywords),
                high_level_keywords=res.high_level_keywords,
                low_level_keywords=res.low_level_keywords,
                elapsed_sec=timer.sec(),
            )
            # Cache non-empty results so reruns are reproducible.
            if res.low_level_keywords or res.high_level_keywords:
                keyword_cache.put(keyword_prompt, res)
            return res

        except Exception as e:
            last_error = str(e)
            _jlog(
                "generate_keywords_attempt_failed",
                request_id,
                step="1",
                attempt=attempt,
                error=last_error,
            )

    # All retries exhausted
    _jlog(
        "generate_keywords_give_up",
        request_id,
        step="1",
        max_retries=max_retries,
        last_error=last_error,
        raw_output_preview=repr(js[:500]) if js else "",
        elapsed_sec=timer.sec(),
    )
    return KeywordExtractionResult()
