"""Extracting the search keywords a question should be looked up by.

One LLM call turns the question into two lists: high-level words for abstract
reasoning and low-level anchors naming concrete entities and topics. Retrieval
uses them differently -- the low-level list drives lexical relationship search,
the high-level list is the baseline -- which is why they are produced together
and kept apart.

The call is cached on disk (`keyword_cache`). That is about reproducibility
before cost: the extraction model is nondeterministic even at temperature 0
with a fixed seed, so an uncached second run of the same question would retrieve
differently for no reason.

Constrained decoding does the schema enforcement; the retries here are for the
model returning something unparseable anyway.
"""


from grace_mem.llm.prompts.keyword.extraction import KEYWORD_EXTRACTION_PROMPT
from grace_mem.pipeline.keyword_cache import keyword_cache
from grace_mem.utils.common import KeywordExtractionResult
from grace_mem.utils.logger_config import _StepTimer, make_module_jlog, setup_logger

_jlog = make_module_jlog(name="grace_mem.Retriever", filename="kg_retriever.jsonl")
logger = setup_logger("grace_mem.Retriever")

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
