"""HyDE: search with a hypothetical answer instead of the question.

A question and the summary that answers it often share few words, so the
question's own embedding can miss it. HyDE asks the LLM to write what an
answering summary would look like, embeds that, and searches with it -- the
hypothetical text lives in the same space as the corpus in a way the question
does not.

Off by default (`summary_hyde_enable`). It costs one extra LLM call per query.

The client and the embedder are passed in: this generates a vector, it does not
own the things that produce one.
"""


import numpy as np

from grace_mem.retrieval.prompts.hyde import HYDE_SYSTEM, HYDE_USER
from grace_mem.utils.logger_config import _StepTimer, make_module_jlog

_jlog = make_module_jlog(name="grace_mem.Retriever", filename="kg_retriever.jsonl")

def generate_hyde_vector(*, llm, searcher, question: str, request_id: str | None = None):
    """
    HyDE: generate hypothetical answer-summary sentences for the question,
    embed them, and return a single normalized vector (mean of sentence
    embeddings). Returns None on failure so the caller falls back to the
    plain query vector.
    """
    timer = _StepTimer()
    try:
        user_prompt = HYDE_USER.format(question=question)
        raw, sec = llm.generate_llm_hyde(HYDE_SYSTEM, user_prompt)
        sentences = [s.strip(" -•\t") for s in (raw or "").splitlines() if s.strip()]
        if not sentences:
            _jlog("hyde_empty", request_id, step="0d", latency_sec=sec)
            return None
        vecs = searcher.embed(sentences)
        vec = np.asarray(vecs, dtype=np.float32).mean(axis=0)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        _jlog(
            "hyde_done",
            request_id,
            step="0d",
            latency_sec=sec,
            sentence_count=len(sentences),
            sentences=sentences,
            elapsed_sec=timer.sec(),
        )
        return vec
    except Exception as exc:
        _jlog("hyde_error", request_id, step="0d", error=str(exc))
        return None
