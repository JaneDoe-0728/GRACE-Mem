"""
KG Evidence Narrowing Module (reductive agent filter)
=====================================================
Post-retrieval narrowing: given the assembled evidence block (pool of ~16
candidate summaries), KEEP only the question-relevant subset and drop the rest.

Motivation
----------
Baseline retrieval already pulls back a large evidence pool (~16 summaries, ~39k
chars) — that is why golden-answer coverage is high (~86%). But too much context
dilutes the QA model: it has to find the answer among many off-topic summaries.
This module is the reductive "agent filter": it greps each retrieved summary for
question/entity keywords, scores by overlap, and keeps the top-N most relevant.

Success metric: keep golden-answer coverage high while shrinking context, so the
QA model focuses on answer-bearing evidence instead of noise.

Offline result (sample 0, baseline evidence): keep top-8 by lexical overlap →
context 39k→22k chars (56%), golden coverage 0.86→0.82, gold preserved on 90% of
questions. The bet: less noise → the QA model answers better.

Integration point in build_kg_context():
    Step 3:   evidence_block = evidence_builder.build_evidence_block(...)  ← pool
    Step 3.5: evidence_block = narrowing_module.narrow(...)                ← HERE
    Step 4:   kg_context = base_text + evidence_block                      ← to LLM

Interface contract:
    narrow(question, evidence_block, *, request_id) -> str
    - Output is a SUBSET of input snippets, same format.
    - Never adds snippets not present in the input.
    - Never raises; on failure returns input unchanged.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from grace_mem.runtime.logger_config import make_module_jlog

_jlog = make_module_jlog(name="grace_mem.Retrieval.Narrowing", filename="kg_retrieval_narrowing.jsonl")
logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "is",
    "was", "were", "are", "be", "did", "do", "does", "what", "when", "where",
    "who", "why", "how", "which", "that", "this", "with", "as", "by", "it",
    "she", "he", "they", "her", "his", "their", "you", "we", "about", "from",
    "has", "have", "had", "not", "would", "could", "will", "been", "than",
    "into", "out", "up", "down", "over", "after", "before",
}


def _content_terms(text: str, min_len: int = 3) -> list[str]:
    """Lowercase content tokens (length-filtered, stopword-filtered)."""
    return [
        t for t in re.findall(r"[a-z0-9']+", (text or "").lower())
        if len(t) >= min_len and t not in _STOPWORDS
    ]


class NarrowingModule:
    """
    Reductive agent filter: narrows the evidence block to the most
    question-relevant snippets by keyword/entity overlap (a "grep" over the
    retrieved summaries), keeping the top-N.

    Config (constructor kwargs):
        keep_top_n:    max snippets to keep (default 8; offline sweet spot)
        min_overlap:   drop snippets sharing fewer than this many query terms
                       (default 1; 0 disables the drop and keeps top-N by score)
        enabled:       master switch (default True). False = identity passthrough.
    """

    # Header lines in the evidence block (kept verbatim, not scored as snippets).
    _HEADER_RE = re.compile(r"^\s*#{1,6}\s")

    def __init__(self, **kwargs: Any) -> None:
        self._cfg = kwargs
        self.keep_top_n: int = int(kwargs.get("keep_top_n", 8))
        self.min_overlap: int = int(kwargs.get("min_overlap", 1))
        self.enabled: bool = bool(kwargs.get("enabled", True))

    def narrow(
        self,
        question: str,
        evidence_block: str,
        *,
        request_id: str | None = None,
        entity_names: list[str] | None = None,
    ) -> str:
        """Keep the top-N question-relevant snippets; drop the rest."""
        if not evidence_block or not self.enabled:
            _jlog("narrowing_passthrough", request_id, step="3.5",
                  strategy="disabled" if not self.enabled else "empty",
                  input_length=len(evidence_block or ""))
            return evidence_block

        try:
            lines = evidence_block.split("\n")
            headers: list[str] = []
            snippets: list[str] = []
            for ln in lines:
                if not ln.strip():
                    continue
                if self._HEADER_RE.match(ln):
                    headers.append(ln)
                else:
                    snippets.append(ln)

            if len(snippets) <= self.keep_top_n and self.min_overlap <= 0:
                _jlog("narrowing_passthrough", request_id, step="3.5",
                      strategy="under_budget", snippet_count=len(snippets),
                      keep_top_n=self.keep_top_n)
                return evidence_block

            # "grep" the question/entity terms across each snippet (lexical overlap).
            qterms = set(_content_terms(question))
            for name in (entity_names or []):
                qterms.update(_content_terms(name))

            scored: list[tuple[int, int, str]] = []  # (overlap, orig_index, snippet)
            for idx, snip in enumerate(snippets):
                low = snip.lower()
                overlap = sum(1 for t in qterms if t in low)
                scored.append((overlap, idx, snip))

            # Drop snippets below the overlap floor (pure noise), then rank by
            # overlap desc, keep top-N, and restore original order for readability.
            kept = [s for s in scored if s[0] >= self.min_overlap]
            if not kept:
                # Nothing matched — fall back to top-N by score so we never empty out.
                kept = scored
            kept.sort(key=lambda x: x[0], reverse=True)
            kept = kept[: self.keep_top_n]
            kept.sort(key=lambda x: x[1])  # restore original order

            new_block = "\n".join(headers + [s[2] for s in kept])
            _jlog("narrowing_applied", request_id, step="3.5",
                  strategy="lexical_topn", keep_top_n=self.keep_top_n,
                  min_overlap=self.min_overlap,
                  snippet_count=len(snippets), kept=len(kept),
                  input_length=len(evidence_block), output_length=len(new_block))
            return new_block
        except Exception as exc:  # narrowing must never break retrieval
            _jlog("narrowing_error", request_id, step="3.5", error=str(exc))
            return evidence_block
