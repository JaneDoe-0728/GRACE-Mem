"""Per-dataset checkpoints, so an interrupted category resumes mid-way.

Coarser than the progress table: progress.csv tracks whether a category is
done, while a checkpoint records how far into one category the run got. A
category can hold hundreds of questions, and losing all of them to a crash
near the end is the cost this avoids.

Keyed by DatasetConfig so two categories never share a checkpoint file.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from experiment.longmem.models import DatasetConfig
from experiment.longmem.utils.io import read_json_file, write_json_file


def checkpoint_path(base_output_dir: Path, config: DatasetConfig) -> Path:
    if config.checkpoint_path:
        return Path(config.checkpoint_path)
    return base_output_dir / f"checkpoint_{config.name}.json"


def load_checkpoint(base_output_dir: Path, config: DatasetConfig) -> dict:
    if not config.resume:
        return {"processed_session_ids": [], "stage": "new"}
    path = checkpoint_path(base_output_dir, config)
    data = read_json_file(path, default=None)
    if not isinstance(data, dict):
        return {"processed_session_ids": [], "stage": "new"}
    if "processed_session_ids" not in data:
        data["processed_session_ids"] = []
    return data


def save_checkpoint(
    base_output_dir: Path,
    config: DatasetConfig,
    processed: set[str],
    *,
    total_sessions: int | None = None,
    stage: str = "ingest_in_progress",
) -> None:
    if not config.resume:
        return
    payload = {
        "dataset": config.name,
        "stage": stage,
        "processed_session_ids": sorted(processed),
        "total_sessions": total_sessions,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    path = checkpoint_path(base_output_dir, config)
    write_json_file(path, payload)
