from typing import Any

from grace_mem.llm.token_tracking import token_tracker

__all__ = ["LLMClient", "token_tracker"]


def __getattr__(name: str) -> Any:
    """Lazily import public LLM exports on first attribute access."""
    if name == "LLMClient":
        from grace_mem.llm.client import LLMClient

        return LLMClient
    raise AttributeError(f"module 'grace_mem.llm' has no attribute {name!r}")
