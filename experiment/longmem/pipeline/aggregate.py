"""Merge per-category results into the run-level answer table and progress rows.

Called as each category finishes rather than once at the end, so a run
interrupted part-way still leaves a valid partial aggregate. Both writers are
read-modify-write against files other workers also touch, which is why the
progress update goes through the locked helper in `helpers/progress.py`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment.longmem.helpers.progress import save_progress_row
from experiment.longmem.utils.io import read_csv_frame, write_csv_frame


def update_all_answers_csv(output_dir: Path, results: list[dict]) -> None:
    all_csv = output_dir / "all_answers.csv"
    if all_csv.exists():
        df = read_csv_frame(all_csv, dtype=str, on_bad_lines="skip", engine="python")
    else:
        df = pd.DataFrame(
            columns=[
                "dataset",
                "question",
                "question_date",
                "answer",
                "Generated_Answer",
                "correctness",
                "Retrieved_Context",
            ]
        )
    updated = 0
    for result in results:
        dataset = result["dataset"]
        mask = df["dataset"].astype(str) == dataset
        row = {
            "Retrieved_Context": result.get("context", ""),
            "Generated_Answer": result.get("answer", ""),
            "correctness": result.get("correctness", ""),
        }
        if mask.any():
            for key, value in row.items():
                df.loc[mask, key] = value
        else:
            new_row = pd.DataFrame(
                [
                    {
                        "dataset": dataset,
                        "question": result.get("question", ""),
                        "question_date": result.get("question_date", ""),
                        "answer": result.get("gold", ""),
                        "Generated_Answer": result.get("answer", ""),
                        "correctness": result.get("correctness", ""),
                        "Retrieved_Context": result.get("context", ""),
                    }
                ]
            )
            df = pd.concat([df, new_row], ignore_index=True)
        updated += 1

    write_csv_frame(df, all_csv)
    print(f"[MERGED] Updated {updated} rows in {all_csv}")


def update_progress_rows(output_dir: Path, results: list[dict], *, filename: str = "progress_rerun.csv") -> None:
    for result in results:
        save_progress_row(
            output_dir,
            dataset=result["dataset"],
            status="judged" if result.get("correctness") in ("0", "1") else "qa_complete",
            correctness=result.get("correctness", ""),
            question=result.get("question", ""),
            gold_answer=result.get("gold", ""),
            generated_answer=result.get("answer", ""),
            filename=filename,
        )
    print(f"[PROGRESS] {len(results)} rows → {output_dir / filename}")
