"""Which endpoint each Agent Filter call talks to.

The agent loop and the answering model are two different jobs, and an experiment
routinely wants them on different endpoints (a 20B agent against a 120B answerer,
say). The override follows the JUDGE_* convention: set it and the agent loop
moves; leave it unset and it shares the answering client it was handed.

The client is cached module-wide: it is stateless and expensive to build.
"""
from __future__ import annotations

import os

_agent_llm_cache = None


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
