"""Argument groups shared across the LongMemEval entry points.

The runner, the child worker, and the rerun tool accept overlapping but not
identical flags. Defining each group once and composing them keeps the three
from drifting -- a flag added to the parent but missing from the child is a
setting that silently applies to only half the run.

`resolve_stages` centralises stage selection so "which stages will execute" has
one answer rather than one per entry point.
"""

from __future__ import annotations

import argparse

VALID_STAGES = ("ingest", "qa_eval", "judge")
DEFAULT_STAGES = VALID_STAGES
RETRIEVAL_ONLY_STAGES = tuple(stage for stage in VALID_STAGES if stage != "ingest")


def add_data_args(parser: argparse.ArgumentParser) -> None:
    """--output-root, --data-folder, --file-pattern"""
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory root",
    )
    parser.add_argument(
        "--data-folder",
        default=None,
        help="Base data folder containing question CSVs",
    )
    parser.add_argument(
        "--file-pattern",
        default="*.csv",
        help="Glob pattern for CSV discovery (default: *.csv)",
    )


def add_child_args(parser: argparse.ArgumentParser) -> None:
    """--child, --child-file, --type"""
    parser.add_argument(
        "--child",
        action="store_true",
        help="Use child_qa manifest to select datasets by dataset_id,category",
    )
    parser.add_argument(
        "--child-file",
        default="./experiment/longmem/script_data/child_qa.txt",
        help="Child dataset manifest path (dataset_id,category format)",
    )
    parser.add_argument(
        "--type",
        nargs="+",
        default=None,
        metavar="TYPE",
        help=(
            "One or more question categories under the data folder, e.g. "
            "single_session_user single_session_assistant "
            "single_session_preference multi_session temporal_reasoning. "
            "In --child mode this is an optional category filter."
        ),
    )


def add_run_args(parser: argparse.ArgumentParser) -> None:
    """--stage, --no-judge, --run-tag, --num, --dataset-id"""
    parser.add_argument(
        "--stage",
        dest="stages",
        nargs="+",
        choices=VALID_STAGES,
        default=None,
        metavar="STAGE",
        help=(
            "Stages to run. Default: full pipeline "
            "(ingest qa_eval judge). "
            "Examples: --stage ingest qa_eval, --stage judge"
        ),
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the judge step",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Run identifier used in output paths",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=None,
        metavar="N",
        help="Limit the number of datasets to process (samples first N from the resolved list)",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help=(
            "Dataset selector list. Supports explicit dataset IDs and lexical indices "
            "within the resolved category list, with comma-separated items and numeric "
            "ranges like 'abc123,0,3-5'."
        ),
    )


def add_rerun_args(parser: argparse.ArgumentParser) -> None:
    """--artifact-dir, --datasets, --force, --result-dir"""
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help=(
            "Base directory containing pre-built artifacts_<dataset_name>/ subdirs. "
            "When --type is set, it is appended to this path. "
            "When set, enables retrieval-only mode (no ingest required)."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Explicit dataset IDs to rerun (default: auto-discover)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun even if context already looks valid",
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        help="Optional extra base directory for rerun summaries; appends --type when set",
    )


def resolve_stages(
    stages: list[str] | tuple[str, ...] | None,
    *,
    env_value: str | None = None,
    no_judge: bool = False,
    artifact_dir: str | None = None,
) -> list[str]:
    resolved = list(stages) if stages else []

    if not resolved and env_value:
        raw_items = env_value.replace(",", " ").split()
        for item in raw_items:
            normalized = "qa_eval" if item == "qa" else item
            if normalized in VALID_STAGES:
                resolved.append(normalized)

    if not resolved:
        resolved = list(RETRIEVAL_ONLY_STAGES if artifact_dir is not None else DEFAULT_STAGES)

    deduped: list[str] = []
    for stage in resolved:
        normalized = "qa_eval" if stage == "qa" else stage
        if normalized in VALID_STAGES and normalized not in deduped:
            deduped.append(normalized)

    if artifact_dir is not None:
        deduped = [stage for stage in deduped if stage != "ingest"]

    if no_judge:
        deduped = [stage for stage in deduped if stage != "judge"]

    return deduped
