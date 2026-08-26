"""Adapters between the runner's arguments and the stage entry points.

Thin by design: each function translates the runner's flat argument namespace
into a stage call and hands back a plain dict. Keeping the translation here
means the stages take explicit parameters rather than reaching into `args`,
which is what lets them be called from tests and from the LongMem runner
without an argparse namespace.

`skipped_judge_stats` returns the same keys as a real judge result with None
values, so downstream aggregation never has to special-case a skipped stage.
"""

import dataclasses
from pathlib import Path
from typing import Any, Sequence


def skipped_judge_stats(*, exclude_adversarial: bool) -> dict[str, Any]:
    """Return a judge-stats shape with None values, for a run that skipped judging.

    Same keys as a real result so aggregation never has to special-case a
    skipped stage. None rather than 0, so "not judged" stays distinguishable
    from "judged and scored zero".
    """
    return {
        "avg_correctness": None,
        "avg_correctness_percent": None,
        "avg_f1": None,
        "avg_bleu1": None,
        "by_category": {},
        "exclude_adversarial": exclude_adversarial,
        "skipped_due_to_adversarial_filter": True,
    }


def configure_retriever(retriever: Any, *, adaptive: bool, tau: float) -> None:
    """Apply the run's adaptive-retrieval settings to a retriever instance.

    Args:
        tau: Confidence threshold below which the adaptive second pass triggers.
            Only consulted when `adaptive` is on.
    """
    if not adaptive:
        return
    retriever.cfg = dataclasses.replace(
        retriever.cfg,
        enable_adaptive_search=True,
        tau_confidence=tau,
    )


def build_eval_rows(
    *,
    qa_eval_module: Any,
    qa_items: Sequence[dict[str, Any]],
    simplify_gold_evidence: bool,
) -> list[dict[str, Any]]:
    """Run QA evaluation over a sample's questions and return the result rows."""
    return qa_eval_module.evaluate_items(
        list(qa_items),
        simplify_evidence=simplify_gold_evidence,
    )


def run_judge_stage(
    *,
    judge_module: Any,
    input_csv: str | Path,
    output_csv: str | Path,
    sample_index: int | None,
    dataset_json: str | Path,
    exclude_adversarial: bool,
) -> dict[str, Any]:
    """Judge a sample's evaluation rows and return the accuracy statistics."""
    return judge_module.llm_as_judge_singlemode(
        input_csv=str(input_csv),
        output_csv=str(output_csv),
        sample_index=sample_index,
        dataset_json=str(dataset_json),
        exclude_adversarial=exclude_adversarial,
    )
