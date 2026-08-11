from __future__ import annotations

from experiment.longmem.stages import IngestStage, JudgeStage, QAEvalStage

_INGEST_STAGE = IngestStage()
_JUDGE_STAGE = JudgeStage()
_QA_STAGE = QAEvalStage()

def normalize_sessions(df):
    return _INGEST_STAGE.normalize_sessions(df)


def ingest_by_turn_pairs(*, ingestor, df, prev_k=None, entity_sim_topk=None, entity_sim_threshold=None, ignore_trailing_user_without_reply=True):
    return _INGEST_STAGE.ingest_by_turn_pairs(
        ingestor,
        df,
        prev_k=prev_k,
        entity_sim_topk=entity_sim_topk,
        entity_sim_threshold=entity_sim_threshold,
        ignore_trailing_user_without_reply=ignore_trailing_user_without_reply,
    )


def ingest_by_session(*, ingestor, df, prev_k=None, entity_sim_topk=None, entity_sim_threshold=None):
    return _INGEST_STAGE.ingest_by_session(
        ingestor,
        df,
        prev_k=prev_k,
        entity_sim_topk=entity_sim_topk,
        entity_sim_threshold=entity_sim_threshold,
    )


def rewrite_temporal_question(question: str, query_time: str | None = None) -> str:
    return _QA_STAGE.rewrite_temporal_question(question, query_time=query_time)


def load_question_from_csv(path):
    return _QA_STAGE.load_question_from_csv(path)


def build_context(*, retriever, question: str, retrieval_params: dict, query_time: str | None) -> str:
    return _QA_STAGE.build_context(
        retriever,
        question=question,
        retrieval_params=retrieval_params,
        query_time=query_time,
    )


def ask_llm(*, llm, question: str, context: str, question_date: str | None) -> str:
    return _QA_STAGE.ask_llm(
        llm,
        question=question,
        context=context,
        question_date=question_date,
    )


def judge_single(
    *,
    llm,
    question: str,
    gold: str,
    generated: str,
    category: str | None = None,
    is_abstention: bool = False,
) -> int:
    return _JUDGE_STAGE.judge_single(
        llm,
        question=question,
        gold=gold,
        generated=generated,
        category=category,
        is_abstention=is_abstention,
    )


def single_result_frame(*, question: str, question_date: str | None, context: str, answer: str, gold: str, correctness: str = ""):
    return _QA_STAGE.single_result_frame(
        question=question,
        question_date=question_date,
        context=context,
        answer=answer,
        gold=gold,
        correctness=correctness,
    )
