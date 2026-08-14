"""Planning: decides what each sample will do before it is executed.

Kept apart from the runner so the decisions are testable without launching a
subprocess, and so the expensive skips are stated in one place. The two that
matter:

`should_skip_graph_restore` -- reuse the previous sample's graph. Valid only
when that sample covered the same conversation and succeeded. Skipping wrongly
means evaluating against another conversation's KG, which produces plausible
but meaningless results.

`should_skip_refresh` -- leave the backend alone between samples. Saves the
restart and its settling delay, at the cost of carrying over any state the
previous sample left behind.
"""

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
    """Pick the bookkeeping strategy for a dataset.

    locomo writes flat output and has no conversation-level context to carry;
    the plus variants place several samples on one conversation and therefore
    need run directories, log syncing, and context tracking. Anything not
    named locomo gets the plus behaviour.
    """
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
    """Describe the sample about to run, or None if this dataset does not track it.

    Returning None for locomo is what makes the skip predicates below
    unconditionally false there -- locomo samples never share a conversation, so
    there is nothing to reuse between them.
    """
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
    """Describe the sample that will run next, for the refresh decision.

    Looking one sample ahead is what allows the backend refresh to be skipped:
    the decision depends on whether the next sample wants the state this one is
    about to leave behind.
    """
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
    """Record how this sample ended, for the next sample's planning.

    A no-op for datasets that do not track context, which keeps `RunState` empty
    rather than filled with values nothing will consult.
    """
    if not strategy.track_plus_context:
        return
    runtime.run_state.update(
        conv_id=plan.conv_id,
        is_cognitive=plan.is_cognitive,
        success=success,
    )


def _judge_dir_for_aggregate(runtime: RunRuntime, strategy: DatasetStrategy) -> Optional[Path]:
    """Locate the run's judge directory, or None when there is nothing to aggregate.

    None both when judging was disabled and when the dataset writes flat output
    and so has no judge subdirectory to collect from.
    """
    if not strategy.uses_run_dirs or runtime.config.no_judge:
        return None
    return runtime.config.run_root / "judge"


def sample_context_for_index(
    sample_index: int,
    all_samples_plus: list[Any],
    *,
    is_cognitive_item,
) -> PlusSampleContext | None:
    """Build the context for a sample by index, or None if the index is past the end.

    The bounds check is what lets callers ask about "the next sample" without
    first checking whether one exists.
    """
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
    """Whether this sample can reuse the graph the previous one left behind.

    Restoring the graph dominates a run's wall clock, so skipping it is the
    largest single saving available -- and also the easiest way to silently
    invalidate a run, since a wrong reuse evaluates against another
    conversation's knowledge graph and still produces plausible answers.

    Every one of the five conditions is load-bearing. Both samples must be
    non-cognitive (the cognitive path mutates the graph, so its leftovers are
    not reusable), they must name the same non-null conversation, and the
    previous sample must have succeeded -- a failed one may have left the graph
    partially built.
    """
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
    """Whether the backend can be left running between this sample and the next.

    The forward-looking counterpart to `should_skip_graph_restore`: that one
    asks whether to inherit state, this asks whether to preserve it. Same
    conditions, applied to the next sample instead of the previous, plus this
    sample having succeeded.
    """
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
    """Decide everything about one sample before its worker is launched.

    The single entry point for planning: it resolves the dataset strategy, the
    sample's context, and whether the graph restore can be skipped, and returns
    them as one immutable plan. Deciding up front rather than inside the worker
    is what makes these choices testable without spawning a process.
    """
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
