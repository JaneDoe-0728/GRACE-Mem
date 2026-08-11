from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment.common.evaluation.judge import JudgeEngine, parse_binary_judge
from experiment.longmem.utils.io import read_csv_dict_rows, write_csv_frame


class JudgeStage:
    """LLM judge stage for LongMem evaluation outputs."""

    DEFAULT_INPUT_CSV = "./experiment/longmem/output/default/temporal_reasoning/all_answers.csv"
    DEFAULT_OUTPUT_CSV = "./experiment/longmem/output/default/temporal_reasoning/all_answers_judged_0316.csv"

    def parse_binary_judge(self, text: str) -> int:
        return parse_binary_judge(text)

    def judge_single(
        self,
        llm,
        *,
        question: str,
        gold: str,
        generated: str,
        category: str | None = None,
        is_abstention: bool = False,
    ) -> int:
        return JudgeEngine(llm, "longmem").judge(
            question=question,
            gold=gold,
            generated=generated,
            category=category,
            is_abstention=is_abstention,
        )

    def llm_as_judge_singlemode(
        self,
        *,
        llm,
        input_csv: str | Path = DEFAULT_INPUT_CSV,
        output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    ) -> None:
        input_path = Path(input_csv)
        output_path = Path(output_csv)
        source = output_path if output_path.exists() else input_path
        print(f"Reading from: {source}")

        _, rows = read_csv_dict_rows(source)
        df = pd.DataFrame(rows)

        q_col = next((c for c in df.columns if c.lower() == "question"), None)
        g_col = next((c for c in df.columns if c.lower() in ["answer", "gold_answer"]), None)
        gen_col = next((c for c in df.columns if c.lower() in ["generated_answer", "model_answer"]), None)

        if not all([q_col, g_col, gen_col]):
            raise ValueError("找不到必要欄位 (question, answer/gold_answer, generated_answer/model_answer)")

        if "correctness" not in df.columns:
            df["correctness"] = ""

        category = input_path.parent.name.replace("_", "-")
        is_abstention = input_path.stem.endswith("_abs")

        for i, row in df.iterrows():
            question = str(row[q_col]).strip()
            gold = str(row[g_col]).strip()
            generated = str(row[gen_col]).strip()
            if not generated:
                df.at[i, "correctness"] = ""
                continue

            existing = str(row.get("correctness", "")).strip()
            if existing in ("0", "1"):
                print(f"Skipping row {i} (already judged: {existing})")
                continue

            print(f"Judging row {i}: {question[:50]}...")
            value = self.judge_single(
                llm,
                question=question,
                gold=gold,
                generated=generated,
                category=category,
                is_abstention=is_abstention,
            )
            df.at[i, "correctness"] = value

        write_csv_frame(df, output_path)
        print(f"Saved to {output_path}")


def parse_binary_judge(text: str) -> int:
    return JudgeStage().parse_binary_judge(text)


def judge_single(
    llm,
    *,
    question: str,
    gold: str,
    generated: str,
    category: str | None = None,
    is_abstention: bool = False,
) -> int:
    return JudgeStage().judge_single(
        llm,
        question=question,
        gold=gold,
        generated=generated,
        category=category,
        is_abstention=is_abstention,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from KG.llm import LLMClient

    JudgeStage().llm_as_judge_singlemode(llm=LLMClient())
