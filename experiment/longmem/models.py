from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DatasetConfig:
    """Configuration for one LongMem dataset run."""

    name: str
    csv_path: str

    ingest_mode: str
    prev_k: int
    entity_sim_topk: int
    entity_sim_threshold: float

    ent_topk: int
    rel_topk: int
    ent_threshold: float
    rel_threshold: float
    filter_ent_topk: int
    filter_rel_topk: int
    filter_ent_threshold: float
    filter_rel_threshold: float
    summary_topk_per_item: int
    summary_vec_threshold: float

    question_column: str = "question"
    output_path: Optional[str] = None
    artifacts_dir: Optional[str] = None

    resume: bool = True
    checkpoint_every_n_sessions: int = 5
    checkpoint_path: Optional[str] = None


@dataclass(frozen=True)
class DatasetPaths:
    base_output_dir: Path
    dataset_name: str

    @property
    def output_csv(self) -> Path:
        return self.base_output_dir / f"{self.dataset_name}.csv"

    @property
    def artifacts_dir(self) -> Path:
        return self.base_output_dir / f"artifacts_{self.dataset_name}"

    @property
    def log_dir(self) -> Path:
        return self.base_output_dir / f"logs_{self.dataset_name}"

    @property
    def checkpoint_json(self) -> Path:
        return self.base_output_dir / f"checkpoint_{self.dataset_name}.json"

