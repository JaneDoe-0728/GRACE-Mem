"""Value types describing a standard LoCoMo run and its sample workers.

The values are frozen because the run configuration is written into metadata;
changing it while samples execute would make that record inaccurate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    """The immutable configuration for one run, as parsed from the CLI.

    Attributes:
        sample_ids: A tuple, not a list, so the config stays hashable and
            genuinely immutable.
        no_judge: Skip LLM judging. Retrieval traces are still produced, which
            makes this the cheap mode for iterating on retrieval.
        post_refresh_sleep: Seconds to wait after refreshing the graph backend.
            Non-zero because the backend accepts connections before it is ready
            to serve, and a query issued in that window fails in a way that
            looks like a retrieval bug.
    """

    dataset_json_path: Path
    sessions_jsonl_path: Path | None
    run_root: Path
    run_tag: str
    sample_ids: tuple[int, ...]
    no_judge: bool
    include_adversarial: bool
    post_refresh_sleep: float

    @classmethod
    def from_args(
        cls,
        *,
        args,
        dataset_json_path: Path,
        sessions_jsonl_path: Path | None,
        run_root: Path,
        sample_ids: Sequence[int],
    ) -> RunConfig:
        """Build a config from parsed args plus the paths the caller resolved.

        Path resolution stays outside this method because it depends on dataset
        layout on disk, which argument parsing cannot know. Values taken from
        `args` are coerced here rather than trusted, since argparse hands back
        whatever the type= said and several of these come through as strings.
        """
        return cls(
            dataset_json_path=dataset_json_path,
            sessions_jsonl_path=sessions_jsonl_path,
            run_root=run_root,
            run_tag=args.run_tag,
            sample_ids=tuple(sample_ids),
            no_judge=bool(args.no_judge),
            include_adversarial=bool(args.adv),
            post_refresh_sleep=float(args.post_refresh_sleep),
        )


@dataclass(frozen=True)
class WorkerPaths:
    """Where one sample's worker writes its outputs.

    Computed up front rather than derived inside the worker so the parent can
    check for existing artifacts and skip a completed sample on resume.

    Attributes:
        sample_dir: Directory containing this sample's artifacts and outputs.
    """

    sample_index: int
    sample_dir: Path
    eval_csv: Path
    judge_csv: Path
    stats_json: Path


@dataclass(frozen=True)
class SamplePlan:
    """One sample index paired with the paths its worker should use."""

    sample_index: int
    worker_paths: WorkerPaths


@dataclass(frozen=True)
class AggregateResult:
    """Paths produced by aggregating a run's per-sample results.

    Attributes:
        merged_csv: None when the run produced no per-sample CSVs to merge.
    """

    output_json: Path
    merged_csv: Path | None = None
