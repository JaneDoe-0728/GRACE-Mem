#!/usr/bin/env python3
"""
Upload progress.csv files from multi_dataset_output subfolders to NocoDB.

- Scans all subdirectories under multi_dataset_output/ for progress.csv
- Excludes the 'stuck_history' column before uploading
- Each subfolder becomes a separate NocoDB table named after the folder

Requires in .env (root):
  NOCO_URL, API_TOKEN, PROJECT_ID
  ORG  (optional, defaults to "noco")

Usage:
  python experiment/upload_progress_to_noco.py
"""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if __package__ in (None, "") and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.noco.client_loader import load_noco_client_class

EXCLUDE_COLS = {"stuck_history"}
OUTPUT_DIR = REPO_ROOT / "experiment" / "multi_dataset_output"


def _upload_file(file_path: str, *, table_title: str, table_name: str) -> str:
    client_class = load_noco_client_class()
    client = client_class(os.environ["NOCO_URL"], os.environ["API_TOKEN"])
    project_id = os.environ["PROJECT_ID"]
    source_id = os.getenv("SOURCE_ID") or client.get_first_source_id(project_id)
    return client.upload_file(
        file_path=file_path,
        project_id=project_id,
        source_id=source_id,
        org=os.getenv("ORG", "noco"),
        table_title=table_title,
        table_name=table_name,
    )


def upload_progress(csv_path: Path, table_name: str) -> None:
    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")

    cols_to_drop = [c for c in df.columns if c.lower() in EXCLUDE_COLS]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  Excluded columns: {cols_to_drop}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8-sig"
    ) as tmp:
        df.to_csv(tmp, index=False)
        tmp_path = tmp.name

    try:
        table_id = _upload_file(tmp_path, table_title=table_name, table_name=table_name)
        print(f"  Uploaded '{table_name}' → table ID: {table_id}")
    finally:
        os.unlink(tmp_path)


def main():
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(REPO_ROOT / "noco-db-uploader" / ".env")
    os.environ.setdefault(
        "NOCO_TARGETS_PATH",
        str(REPO_ROOT / "experiment" / "noco" / "noco_targets.yaml"),
    )

    for required in ("NOCO_URL", "API_TOKEN", "PROJECT_ID"):
        if not os.getenv(required):
            sys.exit(f"Error: {required} is not set in .env")

    progress_files = sorted(OUTPUT_DIR.glob("*/progress.csv"))
    if not progress_files:
        print(f"No progress.csv files found under {OUTPUT_DIR}")
        return

    print(f"Found {len(progress_files)} progress file(s):\n")
    for csv_path in progress_files:
        table_name = csv_path.parent.name
        print(f"[{table_name}] {csv_path}")
        upload_progress(csv_path, table_name)
        print()


if __name__ == "__main__":
    main()
