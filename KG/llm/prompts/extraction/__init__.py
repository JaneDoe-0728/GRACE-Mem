# llm/prompts/extraction/__init__.py
"""
Entity and relationship extraction prompts organized by purpose.
"""
from .two_step import entity_extraction_only, relationship_extraction_only

__all__ = [
    "entity_extraction_only",
    "relationship_extraction_only",
]
