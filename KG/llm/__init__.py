from typing import Any

from KG.llm.token_tracking import token_tracker

__all__ = ["LLMClient", "token_tracker"]


def __getattr__(name: str) -> Any:
    """Lazily import public LLM exports on first attribute access."""
    if name == "LLMClient":
        from KG.llm.client import LLMClient

        return LLMClient
    raise AttributeError(f"module 'KG.llm' has no attribute {name!r}")
