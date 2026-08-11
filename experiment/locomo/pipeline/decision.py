from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from experiment.locomo.models import (
    DatasetStrategy,
    PlusSampleContext,
    PreviousSampleState,
    RunRuntime,
    RunState,
    SamplePlan,
    WorkerPaths,
)


def _select_strategy(dataset: str) -> DatasetStrategy:
    if dataset == "locomo":
        return DatasetStrategy(
            uses_run_dirs=False,
            sync_logs_after_worker=False,
            track_plus_context=False,
        )
    return DatasetStrategy(
        uses_run_dirs=True,
        sync_logs_after_worker=True,
        track_plus_context=True,
    )


def _current_context(
    plan: SamplePlan,
    strategy: DatasetStrategy,
) -> PlusSampleContext | None:
    if not strategy.track_plus_context:
        return None
    return PlusSampleContext(
        sample_index=plan.sample_index,
        conv_id=plan.conv_id,
        is_cognitive=plan.is_cognitive,
    )


def _next_context(
    runtime: RunRuntime,
    next_sample_index: int | None,
    is_cognitive_item,
    strategy: DatasetStrategy,
) -> PlusSampleContext | None:
    if not strategy.track_plus_context or next_sample_index is None or runtime.all_samples_plus is None:
        return None
    return sample_context_for_index(
        next_sample_index,
        runtime.all_samples_plus,
        is_cognitive_item=is_cognitive_item,
    )


def _update_run_state(
    runtime: RunRuntime,
    plan: SamplePlan,
    success: bool,
    strategy: DatasetStrategy,
) -> None:
    if not strategy.track_plus_context:
        return
    runtime.run_state.update(
        conv_id=plan.conv_id,
        is_cognitive=plan.is_cognitive,
        success=success,
    )


def _judge_dir_for_aggregate(runtime: RunRuntime, strategy: DatasetStrategy) -> Optional[Path]:
    if not strategy.uses_run_dirs or runtime.config.no_judge:
        return None
    return runtime.config.run_root / "judge"


def sample_context_for_index(
    sample_index: int,
    all_samples_plus: list[Any],
    *,
    is_cognitive_item,
) -> PlusSampleContext | None:
    if sample_index >= len(all_samples_plus):
        return None
    sample = all_samples_plus[sample_index]
    return PlusSampleContext(
        sample_index=sample_index,
        conv_id=str(sample.get("conversation_id", "")).strip() or None,
        is_cognitive=is_cognitive_item(sample),
    )


def should_skip_graph_restore(
    *,
    current: PlusSampleContext | None,
    previous: PreviousSampleState,
) -> bool:
    if current is None:
        return False
    return (
        not current.is_cognitive
        and not previous.was_cognitive
        and current.conv_id is not None
        and current.conv_id == previous.conv_id
        and previous.success
    )


def should_skip_refresh(
    *,
    current: PlusSampleContext | None,
    next_sample: PlusSampleContext | None,
    current_success: bool,
) -> bool:
    if current is None or next_sample is None or not current_success:
        return False
    return (
        not current.is_cognitive
        and not next_sample.is_cognitive
        and current.conv_id is not None
        and current.conv_id == next_sample.conv_id
    )


def build_sample_plan(
    *,
    dataset: str,
    sample_index: int,
    worker_paths: WorkerPaths,
    run_state: RunState,
    all_samples_plus: list[Any] | None,
    is_cognitive_item,
) -> SamplePlan:
    current_plus = None
    if dataset == "locomo-plus" and all_samples_plus is not None:
        current_plus = sample_context_for_index(
            sample_index,
            all_samples_plus,
            is_cognitive_item=is_cognitive_item,
        )
    return SamplePlan(
        sample_index=sample_index,
        worker_paths=worker_paths,
        skip_graph_restore=should_skip_graph_restore(
            current=current_plus,
            previous=run_state.previous,
        ),
        conv_id=current_plus.conv_id if current_plus else None,
        is_cognitive=current_plus.is_cognitive if current_plus else True,
    )
