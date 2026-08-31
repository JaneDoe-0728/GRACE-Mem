"""
LLM Prompts - Organized by purpose.

This module provides all prompts used by the KG system, organized into logical subfolders:
- config: Common configuration (EXTRA_KWARGS)
- extraction: Entity/relationship extraction prompts (basic, two-step, context-aware, etc.)
- keyword: Keyword extraction for hybrid retrieval
- entity_ops: Entity operation decision rules and examples (ADD vs UPDATE)

All exports maintain backward compatibility with the original flat prompts.py structure.
"""

# Configuration
# Adaptive retrieval
from grace_mem.retrieval.prompts.adaptive import (
    ADAPTIVE_REWRITE_SYSTEM,
    ADAPTIVE_REWRITE_SYSTEM_MULTIHOP,
)

# Keyword extraction
from grace_mem.retrieval.prompts.keyword import KEYWORD_EXTRACTION_PROMPT

from .config import EXTRA_KWARGS

# Entity operations
from .entity_ops import (
    ENTITY_OPS_FEW_SHOT,
    ENTITY_OPS_RULES_V2,
)

# Extraction prompts - most commonly used
from .extraction import (
    entity_extraction_only,
    relationship_extraction_only,
)

__all__ = [
    # Adaptive retrieval
    "ADAPTIVE_REWRITE_SYSTEM",
    "ADAPTIVE_REWRITE_SYSTEM_MULTIHOP",
    "ENTITY_OPS_FEW_SHOT",
    # Entity Ops
    "ENTITY_OPS_RULES_V2",
    # Config
    "EXTRA_KWARGS",
    # Keyword
    "KEYWORD_EXTRACTION_PROMPT",
    # Extraction
    "entity_extraction_only",
    "relationship_extraction_only",
]
