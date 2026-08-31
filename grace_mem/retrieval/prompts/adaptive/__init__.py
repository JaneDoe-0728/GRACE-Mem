"""
Adaptive retrieval query rewrite prompts.
"""
from .multihop import ADAPTIVE_REWRITE_SYSTEM_MULTIHOP
from .rewrite import ADAPTIVE_REWRITE_SYSTEM

__all__ = [
    "ADAPTIVE_REWRITE_SYSTEM",
    "ADAPTIVE_REWRITE_SYSTEM_MULTIHOP",
]
