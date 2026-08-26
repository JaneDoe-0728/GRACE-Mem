"""Value types describing a LoCoMo run: its config, plan, and live state.

Almost everything is frozen. A run's configuration is written into its metadata
and used to decide what to skip on resume, so a value that could change
mid-flight would make the recorded config disagree with what actually ran.
`RunState` and `RunRuntime` are the exceptions -- they exist precisely to
accumulate as samples complete.

The distinction worth holding onto: `RunConfig` is what the user asked for,
`SamplePlan` is what one sample will therefore do, and `RunRuntime` is what has
happened so far.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


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

    dataset: str
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
        dataset: str,
        dataset_json_path: Path,
        sessions_jsonl_path: Path | None,
        run_root: Path,
        sample_ids: Sequence[int],
    ) -> "RunConfig":
        """Build a config from parsed args plus the paths the caller resolved.

        Path resolution stays outside this method because it depends on dataset
        layout on disk, which argument parsing cannot know. Values taken from
        `args` are coerced here rather than trusted, since argparse hands back
        whatever the type= said and several of these come through as strings.
        """
        return cls(
            dataset=dataset,
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
        sample_dir: None for datasets that write flat rather than per-sample;
            see `DatasetStrategy.uses_run_dirs`.
    """

    sample_index: int
    sample_dir: Path | None
    eval_csv: Path
    judge_csv: Path
    stats_json: Path


@dataclass(frozen=True)
class SamplePlan:
    """What one sample is going to do, decided before the worker starts.

    Attributes:
        skip_graph_restore: Reuse the graph left by the previous sample instead
            of rebuilding it. Only valid when that sample used the same
            conversation and succeeded -- restoring is the expensive part of a
            run, and skipping it wrongly evaluates against someone else's KG.
    """

    sample_index: int
    worker_paths: WorkerPaths
    skip_graph_restore: bool
    conv_id: str | None = None


@dataclass(frozen=True)
class AggregateResult:
    """Paths produced by aggregating a run's per-sample results.

    Attributes:
        merged_csv: None when the run produced no per-sample CSVs to merge.
    """

    output_json: Path
    merged_csv: Path | None = None


@dataclass(frozen=True)
class PreviousSampleState:
    """What the previously executed sample did, used to plan the next one.

    Only meaningful in a sequential run. `success` gates graph reuse: a failed
    sample may have left the graph half-built, so the next one must rebuild
    rather than inherit it.
    """

    conv_id: str | None = None
    success: bool = False


@dataclass(frozen=True)
class PlusSampleContext:
    """Sample identity for the "plus" datasets, which index by conversation.

    Those variants can place several samples on one conversation, so the sample
    index alone no longer identifies the underlying data and conv_id has to be
    tracked alongside it.
    """

    sample_index: int
    conv_id: str | None


@dataclass
class RunState:
    """Mutable carry-over between samples in a sequential run.

    Deliberately holds only the immediately preceding sample. Planning never
    looks further back than one step, and keeping the whole history here would
    invite decisions that quietly depend on it.
    """

    previous: PreviousSampleState = field(default_factory=PreviousSampleState)

    def update(self, *, conv_id: str | None, success: bool) -> None:
        """Record the sample that just finished.

        A failed sample stores `conv_id=None`, which is what prevents the next
        sample from reusing its graph: planning compares conv_ids, and None
        matches nothing.
        """
        self.previous = PreviousSampleState(
            conv_id=conv_id if success else None,
            success=success,
        )


@dataclass
class RunRuntime:
    """Everything a run accumulates while executing.

    The mutable counterpart to `RunConfig`: config is what was asked for, this
    is what has happened. `per_sample_stats` is filled in as workers finish and
    is what the aggregate step reads.
    """

    config: RunConfig
    run_summary_json: Path
    per_sample_stats: dict[str, dict]
    run_state: RunState = field(default_factory=RunState)
    all_samples_plus: list[Any] | None = None


@dataclass(frozen=True)
class DatasetStrategy:
    """Per-dataset behaviour switches for the shared run loop.

    LoCoMo and the "plus" variants differ in bookkeeping but not in pipeline,
    so the differences are data here rather than branches scattered through the
    runner -- adding a dataset means adding a strategy, not editing the loop.

    Attributes:
        uses_run_dirs: Write per-sample subdirectories instead of flat output.
        sync_logs_after_worker: Copy worker logs into the run directory once
            the worker exits.
        track_plus_context: Maintain `PlusSampleContext` across samples.
    """

    uses_run_dirs: bool
    sync_logs_after_worker: bool
    track_plus_context: bool
