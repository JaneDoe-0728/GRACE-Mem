from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    dataset_json_path: Path
    sessions_jsonl_path: Path | None
    run_root: Path
    run_tag: str
    sample_ids: tuple[int, ...]
    upload_nocodb: bool
    no_judge: bool
    include_adversarial: bool
    post_refresh_sleep: float

    @classmethod
    def from_args(
        cls,
        *,
        args,
        dataset: str,
        dataset_json_path: Path,
        sessions_jsonl_path: Path | None,
        run_root: Path,
        sample_ids: Sequence[int],
    ) -> "RunConfig":
        return cls(
            dataset=dataset,
            dataset_json_path=dataset_json_path,
            sessions_jsonl_path=sessions_jsonl_path,
            run_root=run_root,
            run_tag=args.run_tag,
            sample_ids=tuple(sample_ids),
            upload_nocodb=bool(args.nocodb),
            no_judge=bool(args.no_judge),
            include_adversarial=bool(args.adv),
            post_refresh_sleep=float(args.post_refresh_sleep),
        )


@dataclass(frozen=True)
class WorkerPaths:
    sample_index: int
    sample_dir: Path | None
    eval_csv: Path
    judge_csv: Path
    stats_json: Path


@dataclass(frozen=True)
class SamplePlan:
    sample_index: int
    worker_paths: WorkerPaths
    skip_graph_restore: bool
    conv_id: str | None = None
    is_cognitive: bool = True


@dataclass(frozen=True)
class AggregateResult:
    output_json: Path
    merged_csv: Path | None = None


@dataclass(frozen=True)
class PreviousSampleState:
    conv_id: str | None = None
    was_cognitive: bool = True
    success: bool = False


@dataclass(frozen=True)
class PlusSampleContext:
    sample_index: int
    conv_id: str | None
    is_cognitive: bool


@dataclass
class RunState:
    previous: PreviousSampleState = field(default_factory=PreviousSampleState)

    def update(self, *, conv_id: str | None, is_cognitive: bool, success: bool) -> None:
        self.previous = PreviousSampleState(
            conv_id=conv_id if success else None,
            was_cognitive=is_cognitive,
            success=success,
        )


@dataclass
class RunRuntime:
    config: RunConfig
    run_summary_json: Path
    per_sample_stats: dict[str, dict]
    run_state: RunState = field(default_factory=RunState)
    all_samples_plus: list[Any] | None = None


@dataclass(frozen=True)
class DatasetStrategy:
    uses_run_dirs: bool
    sync_logs_after_worker: bool
    track_plus_context: bool
