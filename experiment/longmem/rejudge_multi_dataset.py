"""
Re-judge all CSVs in multi_dataset_output with the new category-aware judge prompt.
Adds a `correctness_new` column to each CSV (skips rows already filled).

Usage:
    python experiment/longmem/rejudge_multi_dataset.py
    python experiment/longmem/rejudge_multi_dataset.py --dry-run
    python experiment/longmem/rejudge_multi_dataset.py --col correctness_v2
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import pandas as pd

from KG.llm import LLMClient
from experiment.longmem.stages.judge import JudgeStage

OUTPUT_DIR = _ROOT / "experiment" / "longmem" / "multi_dataset_output"

_DIR_TO_CATEGORY: dict[str, str] = {
    "single_session_user": "single-session-user",
    "single_session_assistant": "single-session-assistant",
    "multi_session": "multi-session",
    "single_session_preference": "single-session-preference",
    "temporal_reasoning": "temporal-reasoning",
    "knowledge_update": "knowledge-update",
}


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def rejudge_csv(
    path: Path,
    *,
    category: str,
    judge: JudgeStage,
    llm,
    col: str,
    dry_run: bool,
) -> tuple[int, int]:
    """Returns (judged, skipped) counts."""
    df = pd.read_csv(path)

    q_col = _find_col(df, ["question"])
    g_col = _find_col(df, ["answer", "gold_answer"])
    gen_col = _find_col(df, ["Generated_Answer", "generated_answer", "model_answer"])

    if not all([q_col, g_col, gen_col]):
        print(f"  [SKIP] {path.name}: missing columns (got {df.columns.tolist()})")
        return 0, 0

    if col not in df.columns:
        df[col] = ""

    judged = skipped = 0
    for i, row in df.iterrows():
        existing = str(row.get(col, "")).strip()
        if existing in ("0", "1"):
            skipped += 1
            continue

        question = str(row[q_col]).strip()
        gold = str(row[g_col]).strip()
        generated = str(row[gen_col]).strip()
        if not generated or not question:
            df.at[i, col] = ""
            continue

        if dry_run:
            print(f"  [DRY] [{category}] {question[:60]}")
            df.at[i, col] = -1
            judged += 1
            continue

        value = judge.judge_single(llm, question=question, gold=gold, generated=generated, category=category)
        df.at[i, col] = value
        judged += 1

    if not dry_run:
        df.to_csv(path, index=False)

    return judged, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--col", default="correctness_new", help="Column name for new judge results")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be judged without calling LLM")
    parser.add_argument("--category", default="", help="Only process this category directory")
    args = parser.parse_args()

    llm = None if args.dry_run else LLMClient()
    judge = JudgeStage()

    total_judged = total_skipped = total_files = 0

    for dir_name, category in _DIR_TO_CATEGORY.items():
        if args.category and dir_name != args.category:
            continue
        cat_dir = OUTPUT_DIR / dir_name
        if not cat_dir.exists():
            continue
        csvs = sorted(cat_dir.glob("*.csv"))
        print(f"\n[{dir_name}] {len(csvs)} files, category={category}")
        for csv_path in csvs:
            judged, skipped = rejudge_csv(
                csv_path,
                category=category,
                judge=judge,
                llm=llm,
                col=args.col,
                dry_run=args.dry_run,
            )
            total_files += 1
            total_judged += judged
            total_skipped += skipped
            status = f"judged={judged} skipped={skipped}"
            print(f"  {csv_path.name}: {status}")

    print(f"\nDone. files={total_files} judged={total_judged} skipped={total_skipped}")


if __name__ == "__main__":
    main()
