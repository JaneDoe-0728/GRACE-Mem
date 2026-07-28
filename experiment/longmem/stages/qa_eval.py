from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiment.longmem.utils.io import read_csv_frame
from KG.utils.query_time_parser import parse_query_time
from KG.utils.temporal import build_time_context, rewrite_temporal_text, time_rewrite_ablation_enabled


class QAEvalStage:
    """Retrieval and answer-generation stage for LongMem."""

    SYSTEM_PROMPT = (
        "You are a concise and accurate assistant. "
        "Use the Retrieved Context. If context is insufficient, use general knowledge, "
        "but prefer retrieved facts. Answer directly."
    )

    def __init__(self, *, retriever=None):
        self.retriever = retriever

    def load_question_from_csv(self, path: str | Path) -> tuple[str, str | None]:
        df = read_csv_frame(Path(path))
        if "question" not in df.columns:
            raise ValueError("CSV 缺少 question 欄位")

        question = next((str(x) for x in df["question"].dropna().tolist() if str(x).strip()), None)
        if not question:
            raise ValueError("question 欄位全為空")

        question_date = None
        if "question_date" in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row.get("question")) and str(row["question"]).strip() == question:
                    if pd.notna(row.get("question_date")):
                        question_date = str(row["question_date"]).strip()
                        break

        return question.strip(), question_date

    def rewrite_temporal_question(self, question: str, query_time: str | None = None) -> str:
        # Ablation G: query 端時間改寫全關(ingest 端已烙進 artifacts,不在範圍)
        if time_rewrite_ablation_enabled():
            print("⏰ [ablation] time rewrite skipped (KG_ABLATION_NO_TIME_REWRITE=1)")
            return question

        if not query_time:
            return question

        reference_dt = parse_query_time(query_time)
        if reference_dt is None:
            return question

        rewritten_question, metadata = rewrite_temporal_text(
            question,
            build_time_context(
                reference_dt=reference_dt,
                reference_time_str=query_time,
                source="longmem",
            ),
        )

        constraints = metadata.get("constraints", [])
        resolved_constraints = [
            constraint for constraint in constraints
            if ((constraint.get("resolution") or {}).get("status") == "resolved")
        ]
        if resolved_constraints:
            print("⏰ Time expressions detected and rewritten:")
            print(f"   Original:  {question}")
            print(f"   Rewritten: {rewritten_question}")
            for constraint in resolved_constraints:
                resolution = constraint.get("resolution") or {}
                print(f"   • '{constraint.get('original_text', '')}' -> {resolution.get('normalized_text')}")

        return rewritten_question

    def build_context(
        self,
        retriever=None,
        *,
        question: str,
        retrieval_params: dict,
        query_time: str | None = None,
    ) -> str:
        resolved_retriever = retriever or self.retriever
        if resolved_retriever is None:
            raise ValueError("Retriever is required for QAEvalStage.build_context")
        return resolved_retriever.build_kg_context(
            question=question,
            query_time=query_time,
            **retrieval_params,
        )

    def ask_llm(self, llm, *, question: str, context: str, question_date: str | None = None) -> str:
        system_content = self.SYSTEM_PROMPT
        if question_date:
            system_content += f"\n\nCurrent Date/Time: {question_date}"
            system_content += "\nNote: When answering temporal questions (e.g., 'how long ago', 'how many months'), calculate based on this date."
        system_content += f"\n\n---Retrieved Context---\n{context}\n------------------"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\n\nAnswer:"},
        ]
        resp = llm.chat(messages=messages, temperature=0.0, max_tokens=1024)
        return resp.choices[0].message.content.strip()

    def debug_messages(self, *, question: str, context: str, question_date: str | None = None) -> list[dict]:
        system_content = self.SYSTEM_PROMPT
        if question_date:
            system_content += f"\n\nCurrent Date/Time: {question_date}"
            system_content += "\nNote: When answering temporal questions (e.g., 'how long ago', 'how many months'), calculate based on this date."
        system_content += f"\n\n---Retrieved Context---\n{context}\n------------------"
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\n\nAnswer:"},
        ]

    def single_result_frame(
        self,
        *,
        question: str,
        question_date: str | None,
        context: str,
        answer: str,
        gold: str,
        correctness: str = "",
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "question": question,
                    "question_date": question_date or "",
                    "Retrieved_Context": context,
                    "Generated_Answer": answer,
                    "answer": gold,
                    "correctness": correctness,
                }
            ]
        )

    def run_single_csv(
        self,
        *,
        csv_path: str | Path,
        llm,
        retriever=None,
        retrieval_params: dict,
    ) -> tuple[str, str, str | None]:
        question, question_date = self.load_question_from_csv(csv_path)
        rewritten = self.rewrite_temporal_question(question, query_time=question_date)
        context = self.build_context(
            retriever,
            question=rewritten,
            retrieval_params=retrieval_params,
            query_time=question_date,
        )
        print(
            "=== MESSAGES TO LLM ===\n",
            json.dumps(
                self.debug_messages(question=rewritten, context=context, question_date=question_date),
                ensure_ascii=False,
                indent=2,
            ),
        )
        answer = self.ask_llm(llm, question=rewritten, context=context, question_date=question_date)
        return answer, context, question_date


def rewrite_temporal_question(question: str, query_time: str | None = None) -> str:
    return QAEvalStage().rewrite_temporal_question(question, query_time=query_time)


def build_context(retriever, *, question: str, retrieval_params: dict, query_time: str | None = None) -> str:
    return QAEvalStage(retriever=retriever).build_context(
        question=question,
        retrieval_params=retrieval_params,
        query_time=query_time,
    )


def ask_llm(llm, *, question: str, context: str, question_date: str | None = None) -> str:
    return QAEvalStage().ask_llm(
        llm,
        question=question,
        context=context,
        question_date=question_date,
    )


def single_result_frame(
    *,
    question: str,
    question_date: str | None,
    context: str,
    answer: str,
    gold: str,
    correctness: str = "",
) -> pd.DataFrame:
    return QAEvalStage().single_result_frame(
        question=question,
        question_date=question_date,
        context=context,
        answer=answer,
        gold=gold,
        correctness=correctness,
    )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from KG.llm import LLMClient
    from KG.pipeline.factory import build_pipeline as _build_pipeline
    from experiment_config import RETRIEVAL_PARAMS

    CSV_PATH = "./experiment/longmem/script_data/temporal_reasoning/2ebe6c92.csv"
    retriever = _build_pipeline()["retriever"]
    stage = QAEvalStage(retriever=retriever)
    answer, _, _ = stage.run_single_csv(
        csv_path=CSV_PATH,
        llm=LLMClient(),
        retrieval_params=RETRIEVAL_PARAMS,
    )
    print("answer:", answer)
