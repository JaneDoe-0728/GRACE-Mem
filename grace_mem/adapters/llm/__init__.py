"""The OpenAI chat client and the process-wide token tracker.

`token_tracker` is eager and `LLMClient` lazy: the tracker is a plain in-memory
counter while the client drags in the OpenAI SDK, httpx and the entity-ops
processor. Code that only wants token totals should not pay for a transport
stack.
"""

from typing import Any

from grace_mem.adapters.llm.token_tracking import token_tracker

__all__ = ["LLMClient", "token_tracker"]


def __getattr__(name: str) -> Any:
    """Lazily import the LLM client on first attribute access."""
    if name == "LLMClient":
        from grace_mem.adapters.llm.client import LLMClient

        return LLMClient
    raise AttributeError(f"module 'grace_mem.adapters.llm' has no attribute {name!r}")
