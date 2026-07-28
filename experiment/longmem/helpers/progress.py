from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from experiment.longmem.utils.io import ensure_dir, file_lock, read_csv_frame, write_csv_frame


PROGRESS_COLUMNS = [
    "dataset",
    "status",
    "correctness",
    "question",
    "gold_answer",
    "generated_answer",
    "updated_at",
    "stuck_history",
]


def build_noco_table_name(*, run_tag: str | None, target_name: str) -> str:
    """Build the canonical LongMem NocoDB table name."""

    normalized_run_tag = (run_tag or "").strip()
    return f"{normalized_run_tag}_{target_name}" if normalized_run_tag else target_name


def progress_path(base_output_dir: Path, filename: str = "progress.csv") -> Path:
    return base_output_dir / filename


def progress_lock_path(base_output_dir: Path, filename: str = "progress.csv") -> Path:
    return progress_path(base_output_dir, filename).with_name(f"{filename}.lock")


def load_progress(base_output_dir: Path, filename: str = "progress.csv") -> pd.DataFrame:
    path = progress_path(base_output_dir, filename)
    if path.exists():
        try:
            df = read_csv_frame(path, dtype=str)
            for column in PROGRESS_COLUMNS:
                if column not in df.columns:
                    df[column] = ""
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=PROGRESS_COLUMNS)


def _mutate_progress(
    base_output_dir: Path,
    filename: str,
    updater: Callable[[pd.DataFrame], pd.DataFrame | None],
) -> None:
    ensure_dir(base_output_dir)
    with file_lock(progress_lock_path(base_output_dir, filename)):
        df = load_progress(base_output_dir, filename)
        updated = updater(df)
        write_csv_frame(updated if updated is not None else df, progress_path(base_output_dir, filename))


def init_progress_rows(base_output_dir: Path, dataset_names: list[str], filename: str = "progress.csv") -> None:
    def updater(df: pd.DataFrame) -> pd.DataFrame:
        existing = set(df["dataset"].astype(str).tolist())
        new_rows = []
        for dataset_name in dataset_names:
            if dataset_name in existing:
                continue
            new_rows.append(
                {
                    "dataset": dataset_name,
                    "status": "not_started",
                    "correctness": "",
                    "question": "",
                    "gold_answer": "",
                    "generated_answer": "",
                    "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "stuck_history": "",
                }
            )
        if new_rows:
            return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        return df

    _mutate_progress(base_output_dir, filename, updater)


def save_progress_row(
    base_output_dir: Path,
    *,
    dataset: str,
    status: str,
    correctness: str = "",
    question: str = "",
    gold_answer: str = "",
    generated_answer: str = "",
    filename: str = "progress.csv",
) -> None:
    def updater(df: pd.DataFrame) -> pd.DataFrame:
        row = {
            "dataset": dataset,
            "status": status,
            "correctness": correctness,
            "question": question,
            "gold_answer": gold_answer,
            "generated_answer": generated_answer,
            "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        mask = df["dataset"] == dataset
        if mask.any():
            for key, value in row.items():
                df.loc[mask, key] = value
            return df
        row["stuck_history"] = ""
        return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    _mutate_progress(base_output_dir, filename, updater)


def append_stuck_history(base_output_dir: Path, *, dataset: str, processed: int, total, filename: str = "progress.csv") -> None:
    entry = f"{datetime.utcnow().strftime('%Y-%m-%dT%H:%M')} (processed {processed}/{total} sessions)"
    append_stuck_history_entry(base_output_dir, dataset=dataset, entry=entry, filename=filename)


def append_stuck_history_entry(
    base_output_dir: Path,
    *,
    dataset: str,
    entry: str,
    filename: str = "progress.csv",
) -> None:
    def updater(df: pd.DataFrame) -> pd.DataFrame:
        if "stuck_history" not in df.columns:
            df["stuck_history"] = ""
        mask = df["dataset"] == dataset
        if mask.any():
            existing = str(df.loc[mask, "stuck_history"].iloc[0])
            existing = "" if existing in ("nan", "None") else existing
            df.loc[mask, "stuck_history"] = (existing + "; " + entry).strip("; ")
            return df
        return pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "dataset": dataset,
                            "status": "",
                            "correctness": "",
                            "question": "",
                            "gold_answer": "",
                            "generated_answer": "",
                            "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "stuck_history": entry,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    _mutate_progress(base_output_dir, filename, updater)
