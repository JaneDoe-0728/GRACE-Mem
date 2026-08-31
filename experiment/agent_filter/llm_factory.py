"""Which endpoint each Agent Filter call talks to.

The agent loop, the sufficiency verifier and the answering model are three
different jobs, and an experiment routinely wants them on different endpoints
(a 120B verifier against a 20B agent, say). Each override follows the JUDGE_*
convention: set it and that call moves; leave it unset and the call shares the
answering client it was handed.

Both clients are cached module-wide: they are stateless and expensive to build.
"""
from __future__ import annotations

import os

_agent_llm_cache = None
_verify_llm_cache = None


def agent_llm(default_llm):
    """GREP_AGENT_LLM_API / GREP_AGENT_MODEL_NAME can point at a different endpoint
    (following the JUDGE_* convention); when unset, the answering LLM passed in is
    shared."""
    global _agent_llm_cache
    base = os.getenv("GREP_AGENT_LLM_API")
    name = os.getenv("GREP_AGENT_MODEL_NAME")
    if not (base or name):
        return default_llm
    if _agent_llm_cache is None:
        from grace_mem.adapters.llm import LLMClient
        _agent_llm_cache = LLMClient(base_url=base or None, model_name=name or None)
    return _agent_llm_cache


def verify_llm(default_llm):
    """GREP_AGENT_VERIFY_LLM_API / GREP_AGENT_VERIFY_MODEL_NAME can point the
    sufficiency verifier alone at a different endpoint (120B, say) without
    affecting the agent loop or answering. One of the main reasons v3-v6 were
    closed out was the oss-20b verifier's 43% misjudgement rate; this hook exists
    to retest with a stronger verifier."""
    global _verify_llm_cache
    base = os.getenv("GREP_AGENT_VERIFY_LLM_API")
    name = os.getenv("GREP_AGENT_VERIFY_MODEL_NAME")
    if not (base or name):
        return default_llm
    if _verify_llm_cache is None:
        from grace_mem.adapters.llm import LLMClient
        _verify_llm_cache = LLMClient(base_url=base or None, model_name=name or None, timeout=300.0)
    return _verify_llm_cache
