from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
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
    use_split_summary: bool = True

    question_column: str = "question"
    output_path: Optional[str] = None
    artifacts_dir: Optional[str] = None

    resume: bool = True
    checkpoint_every_n_sessions: int = 5
    checkpoint_path: Optional[str] = None

    @classmethod
    def from_params(
        cls,
        *,
        name: str,
        csv_path: str,
        ingest_params: Mapping[str, Any],
        retrieval_params: Mapping[str, Any],
        **overrides: Any,
    ) -> "DatasetConfig":
        """Build a LongMem config from shared experiment parameter mappings."""
        values = {
            "ingest_mode": ingest_params["ingest_mode"],
            "prev_k": ingest_params["prev_k"],
            "entity_sim_topk": ingest_params["entity_sim_topk"],
            "entity_sim_threshold": ingest_params["entity_sim_threshold"],
            "use_split_summary": ingest_params.get("use_split_summary", True),
            "ent_topk": retrieval_params["ent_topk"],
            "rel_topk": retrieval_params["rel_topk"],
            "ent_threshold": retrieval_params["ent_threshold"],
            "rel_threshold": retrieval_params["rel_threshold"],
            "filter_ent_topk": retrieval_params["filter_ent_topk"],
            "filter_rel_topk": retrieval_params["filter_rel_topk"],
            "filter_ent_threshold": retrieval_params["filter_ent_threshold"],
            "filter_rel_threshold": retrieval_params["filter_rel_threshold"],
            "summary_topk_per_item": retrieval_params["summary_topk_per_item"],
            "summary_vec_threshold": retrieval_params["summary_vec_threshold"],
            **overrides,
        }
        return cls(name=name, csv_path=csv_path, **values)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Dataset name must not be empty")
        if not self.csv_path.strip():
            raise ValueError("Dataset csv_path must not be empty")
        if self.ingest_mode not in {"turn_pairs", "session"}:
            raise ValueError(f"Unsupported ingest_mode: {self.ingest_mode}")

        non_negative = {
            "prev_k": self.prev_k,
            "checkpoint_every_n_sessions": self.checkpoint_every_n_sessions,
        }
        positive = {
            "entity_sim_topk": self.entity_sim_topk,
            "ent_topk": self.ent_topk,
            "rel_topk": self.rel_topk,
            "filter_ent_topk": self.filter_ent_topk,
            "filter_rel_topk": self.filter_rel_topk,
            "summary_topk_per_item": self.summary_topk_per_item,
        }
        for field_name, value in non_negative.items():
            if int(value) < 0:
                raise ValueError(f"{field_name} must be >= 0")
        for field_name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{field_name} must be > 0")

        thresholds = {
            "entity_sim_threshold": self.entity_sim_threshold,
            "ent_threshold": self.ent_threshold,
            "rel_threshold": self.rel_threshold,
            "filter_ent_threshold": self.filter_ent_threshold,
            "filter_rel_threshold": self.filter_rel_threshold,
            "summary_vec_threshold": self.summary_vec_threshold,
        }
        for field_name, value in thresholds.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")

    def retrieval_kwargs(self) -> dict[str, int | float]:
        return {
            "ent_topk": self.ent_topk,
            "rel_topk": self.rel_topk,
            "ent_threshold": self.ent_threshold,
            "rel_threshold": self.rel_threshold,
            "filter_ent_topk": self.filter_ent_topk,
            "filter_rel_topk": self.filter_rel_topk,
            "filter_ent_threshold": self.filter_ent_threshold,
            "filter_rel_threshold": self.filter_rel_threshold,
            "summary_topk_per_item": self.summary_topk_per_item,
            "summary_vec_threshold": self.summary_vec_threshold,
        }


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
