"""Shared progress table, written concurrently by every worker.

A LongMemEval run spans many categories evaluated in parallel, and progress.csv
is the one place that records where each stands. Because several processes
append to it at once, every mutation goes through `file_lock` and the
read-modify-write in `_mutate_progress` -- an unlocked update loses whichever
worker wrote first.

The file is also what makes a run resumable: on restart the runner reads it to
decide what still needs doing, so a row that was never written looks like work
that was never done.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd

from experiment.longmem.utils.io import (
    ensure_dir,
    file_lock,
    read_csv_frame,
    write_csv_frame,
)

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


def progress_path(base_output_dir: Path, filename: str = "progress.csv") -> Path:
    return base_output_dir / filename


def progress_lock_path(base_output_dir: Path, filename: str = "progress.csv") -> Path:
    return progress_path(base_output_dir, filename).with_name(f"{filename}.lock")


def load_progress(base_output_dir: Path, filename: str = "progress.csv") -> pd.DataFrame:
    """Read the progress table, returning an empty frame if it is unreadable.

    Read as strings throughout: dataset names and statuses are categorical, and
    letting pandas infer types turns a numeric-looking dataset name into a
    float and breaks every subsequent match against it.

    Missing columns are backfilled so callers can index them unconditionally,
    which is what lets an older progress file be read by newer code.

    A corrupt file yields an empty frame rather than raising -- this is called
    while other processes are mid-write, so a transiently unparseable read is
    expected. The caller holds the lock before any read-modify-write.
    """
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
    """Apply `updater` to the progress table under an exclusive file lock.

    Every mutation goes through here. Several worker processes update this file
    concurrently, and an unlocked read-modify-write loses whichever writer
    finished first -- silently, and in a way that looks afterwards like the run
    simply never did that work.

    The lock is advisory (fcntl), so it holds only because all writers use this
    function.
    """
    ensure_dir(base_output_dir)
    with file_lock(progress_lock_path(base_output_dir, filename)):
        df = load_progress(base_output_dir, filename)
        updated = updater(df)
        write_csv_frame(updated if updated is not None else df, progress_path(base_output_dir, filename))


def init_progress_rows(base_output_dir: Path, dataset_names: list[str], filename: str = "progress.csv") -> None:
    """Seed a not_started row for each dataset, leaving existing rows alone.

    Skipping datasets already present is what makes this safe to call on a
    resumed run: re-seeding would reset completed datasets to not_started and
    the run would redo all of them.
    """
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
    """Upsert one dataset's progress row.

    Carries the question, gold, and generated answer alongside the status so the
    progress table doubles as a live view of results -- a run can be inspected
    without opening the per-category outputs.
    """
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
    """Append a stall record to a dataset's stuck_history.

    Appends rather than overwrites because repeated stalls are the signal worth
    having: one is a transient backend hiccup, three on the same dataset is a
    reproducible problem with that data.
    """
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
