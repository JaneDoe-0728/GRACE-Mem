from typing import TYPE_CHECKING, Any

__all__ = ["LLMClient", "token_tracker"]

if TYPE_CHECKING:
    from KG.llm.client import LLMClient, token_tracker


def __getattr__(name: str) -> Any:
    """Lazily import public LLM exports on first attribute access."""
    if name in {"LLMClient", "token_tracker"}:
        from KG.llm.client import LLMClient, token_tracker

        exports = {"LLMClient": LLMClient, "token_tracker": token_tracker}
        return exports[name]
    raise AttributeError(f"module 'KG.llm' has no attribute {name!r}")
