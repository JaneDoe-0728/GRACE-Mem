"""Shared module-level adapters for LongMemEval QA helpers.

The processor uses these two helpers as plain functions while sharing one
QAEvalStage instance.
"""

from __future__ import annotations

from experiment.longmem.stages import QAEvalStage

_QA_STAGE = QAEvalStage()

def rewrite_temporal_question(question: str, query_time: str | None = None) -> str:
    return _QA_STAGE.rewrite_temporal_question(question, query_time=query_time)


def single_result_frame(*, question: str, question_date: str | None, context: str, answer: str, gold: str, correctness: str = ""):
    """Module-level forwarder to `QAEvalStage.single_result_frame`."""
    return _QA_STAGE.single_result_frame(
        question=question,
        question_date=question_date,
        context=context,
        answer=answer,
        gold=gold,
        correctness=correctness,
    )
