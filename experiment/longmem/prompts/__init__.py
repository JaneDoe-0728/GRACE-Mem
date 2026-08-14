"""LongMemEval judge prompt and its message builder."""

from .judge import SYSTEM_PROMPT as JUDGE_SYSTEM_PROMPT, build_messages as build_judge_messages

__all__ = ["JUDGE_SYSTEM_PROMPT", "build_judge_messages"]

