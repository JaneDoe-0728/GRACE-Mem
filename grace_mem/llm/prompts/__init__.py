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
from .config import EXTRA_KWARGS

# Extraction prompts - most commonly used
from .extraction import (
    entity_extraction_only,
    relationship_extraction_only,
)

# Keyword extraction
from .keyword import KEYWORD_EXTRACTION_PROMPT

# Adaptive retrieval
from .adaptive import (
    ADAPTIVE_REWRITE_SYSTEM,
    ADAPTIVE_REWRITE_SYSTEM_MULTIHOP,
)

# Entity operations
from .entity_ops import (
    ENTITY_OPS_RULES_V2,
    ENTITY_OPS_FEW_SHOT,
)

__all__ = [
    # Config
    "EXTRA_KWARGS",

    # Extraction
    "entity_extraction_only",
    "relationship_extraction_only",

    # Keyword
    "KEYWORD_EXTRACTION_PROMPT",

    # Adaptive retrieval
    "ADAPTIVE_REWRITE_SYSTEM",
    "ADAPTIVE_REWRITE_SYSTEM_MULTIHOP",

    # Entity Ops
    "ENTITY_OPS_RULES_V2",
    "ENTITY_OPS_FEW_SHOT",
]
