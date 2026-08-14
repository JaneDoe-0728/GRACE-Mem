"""LLM layer: the chat client and the process-wide token tracker.

`token_tracker` is imported eagerly and `LLMClient` lazily, because the tracker
is a plain in-memory counter while the client drags in the OpenAI SDK, httpx,
and the entity-ops processor. Code that only wants to read token totals -- the
analysis scripts, mostly -- should not pay for a transport stack.
"""

from typing import Any

from grace_mem.llm.token_tracking import token_tracker

__all__ = ["LLMClient", "token_tracker"]


def __getattr__(name: str) -> Any:
    """Lazily import public LLM exports on first attribute access."""
    if name == "LLMClient":
        from grace_mem.llm.client import LLMClient

        return LLMClient
    raise AttributeError(f"module 'grace_mem.llm' has no attribute {name!r}")
