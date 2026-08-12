"""
Adaptive retrieval query rewrite prompts.
"""
from .rewrite import ADAPTIVE_REWRITE_SYSTEM
from .multihop import ADAPTIVE_REWRITE_SYSTEM_MULTIHOP

__all__ = [
    "ADAPTIVE_REWRITE_SYSTEM",
    "ADAPTIVE_REWRITE_SYSTEM_MULTIHOP",
]
