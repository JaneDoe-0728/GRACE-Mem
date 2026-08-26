#!/usr/bin/env python3
"""Aggregate LoCoMo judge outputs from ``sample_*`` run folders."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence

MODULE_DIR = Path(__file__).resolve().parent
if __package__ in (None, ""):
    repo_root = MODULE_DIR.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiment.locomo.models import AggregateResult
from experiment.locomo.analysis.summary import compute_summary_from_df, compute_summary_from_rows
from experiment.locomo.utils.log import log_event

if TYPE_CHECKING:
    import pandas as pd

    DataFrame = pd.DataFrame
else:
    DataFrame = Any

def find_sample_dirs(root: Path) -> list[Path]:
    return sorted([path for path in root.iterdir() if path.is_dir() and path.name.startswith("sample_")])


def read_csvs(csv_files: Sequence[Path]) -> tuple[list[DataFrame], list[str]]:
    """Read several CSVs into one frame, skipping any that are missing or empty.

    Tolerant because aggregation runs over whatever samples finished; a sample
    that failed leaves no CSV, and that should shorten the aggregate rather than
    abort it.
    """
    pd = _require_pandas()
    frames: list[DataFrame] = []
    errors: list[str] = []
    for csv_path in csv_files:
        try:
            frames.append(pd.read_csv(csv_path))
        except Exception as exc:
            errors.append(f"{csv_path.name}: {exc}")
    return frames, errors


def _latest_judge_csv(sample_dir: Path) -> Path | None:
    candidates = sorted(sample_dir.glob("*_judge*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _require_pandas():
    """Import pandas on demand, with a clear message if it is absent.

    Deferred so the runner can import this module -- and reach its path helpers
    -- without pandas installed.
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit("pandas is required for experiment/locomo/stats/aggregate.py") from exc
    return pd


def aggregate_judge_csv_files(
    csv_files: Sequence[Path],
    *,
    root: Path,
    exclude_adversarial: bool,
    note: str = "overall stats only include samples with judge CSVs",
    sample_name_fn: Callable[[Path], str] | None = None,
) -> tuple[Dict[str, Any], DataFrame]:
    """Combine per-sample judge CSVs into overall and per-category accuracy.

    Rows with no parseable verdict are excluded from both numerator and
    denominator rather than counted wrong -- an unjudged question is missing
    data, and scoring it as incorrect makes an interrupted judge run look like a
    quality regression.
    """
    pd = _require_pandas()
    merged_frames: list[DataFrame] = []
    per_sample: Dict[str, Dict[str, object]] = {}
    sources: Dict[str, str] = {}
    rows_for_overall: list[dict[str, Any]] = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        sample_name = (
            sample_name_fn(csv_path)
            if sample_name_fn is not None
            else f"sample_{csv_path.stem}"
        )
        df.insert(0, "sample", sample_name)
        merged_frames.append(df)
        rows_for_overall.extend(df.to_dict("records"))

        sample_summary = compute_summary_from_df(df, exclude_adversarial=exclude_adversarial)
        per_sample[sample_name] = {
            "avg_correctness": sample_summary["overall"]["avg_correctness"],
            "avg_correctness_percent": sample_summary["overall"]["avg_correctness_percent"],
            "avg_f1": sample_summary["overall"]["avg_f1"],
            "avg_bleu1": sample_summary["overall"]["avg_bleu1"],
            "by_category": sample_summary["by_category"],
            "source": str(csv_path),
        }
        sources[sample_name] = "judge_csv"

    merged_df = pd.concat(merged_frames, ignore_index=True)
    overall_summary = compute_summary_from_rows(
        rows_for_overall,
        exclude_adversarial=exclude_adversarial,
    )
    payload = {
        "root": str(root),
        "per_sample": per_sample,
        "overall": {
            **overall_summary["overall"],
            "by_category": overall_summary["by_category"],
            "note": note,
        },
        "sources": sources,
    }
    return payload, merged_df


def _summary_entry_from_payload(data: dict[str, Any], source: Path | str) -> Dict[str, object]:
    """Extract one sample's summary row from its stats payload."""
    return {
        "avg_correctness": data.get("avg_correctness"),
        "avg_correctness_percent": data.get("avg_correctness_percent"),
        "avg_f1": data.get("avg_f1"),
        "avg_bleu1": data.get("avg_bleu1"),
        "by_category": data.get("by_category", {}),
        "source": str(source),
    }


def _missing_entry() -> Dict[str, object]:
    """Placeholder summary for a sample that produced no output.

    Emitted rather than omitted so a missing sample is visible in the run
    summary. Silently skipping it would make a run of eight samples and a run of
    ten look alike apart from the averages.
    """
    return {
        "avg_correctness": None,
        "avg_correctness_percent": None,
        "avg_f1": None,
        "avg_bleu1": None,
        "by_category": {},
        "source": "missing",
    }


def _print_skipped(errors: Sequence[str]) -> None:
    """Report samples that produced no output.

    Printed rather than passed over, so a run of eight samples is
    distinguishable from a run of ten with two silent failures -- the averages
    alone would not show it.
    """
    if not errors:
        return
    print(f"[WARN] Skipped {len(errors)} files due to read errors:")
    for err in errors[:10]:
        print(f"  {err}")


def _run_locomo(args: argparse.Namespace) -> None:
    """Aggregate a standard LoCoMo run into its summary JSON and merged CSV."""
    if not args.root:
        raise SystemExit("--root is required when --dataset=locomo")

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    sample_dirs = find_sample_dirs(root)
    if not sample_dirs:
        raise SystemExit(f"No sample_* dirs under: {root}")

    output_json = Path(args.output_json) if args.output_json else root / "_correctness_aggregate.json"
    merged_csv = Path(args.merged_csv) if args.merged_csv else root / "_judge_merged.csv"

    judge_csv_files: list[Path] = []
    sample_names_by_judge_csv: dict[Path, str] = {}
    summary_entries: Dict[str, Dict[str, object]] = {}
    missing_sample_names: list[str] = []

    for sample_dir in sample_dirs:
        sample_name = sample_dir.name
        summary_path = sample_dir / "correctness_summary.json"
        judge_csv = _latest_judge_csv(sample_dir)

        if args.use_summaries and summary_path.exists():
            summary_entries[sample_name] = _summary_entry_from_payload(
                json.loads(summary_path.read_text()),
                summary_path,
            )
            continue

        if judge_csv and judge_csv.exists():
            judge_csv_files.append(judge_csv)
            sample_names_by_judge_csv[judge_csv] = sample_name
            continue

        if summary_path.exists():
            summary_entries[sample_name] = _summary_entry_from_payload(
                json.loads(summary_path.read_text()),
                summary_path,
            )
        else:
            missing_sample_names.append(sample_name)

    if judge_csv_files:
        output, merged_df = aggregate_judge_csv_files(
            judge_csv_files,
            root=root,
            exclude_adversarial=args.exclude_adversarial,
            sample_name_fn=lambda path: sample_names_by_judge_csv[path],
        )
    else:
        output = {
            "root": str(root),
            "per_sample": {},
            "overall": {
                "avg_correctness": None,
                "avg_correctness_percent": None,
                "avg_f1": None,
                "avg_bleu1": None,
                "count": 0,
                "count_f1": 0,
                "count_bleu1": 0,
                "by_category": {},
                "macro_avg_by_category": None,
                "macro_avg_by_category_percent": None,
                "exclude_adversarial": bool(args.exclude_adversarial),
                "note": "overall stats only include samples with judge CSVs",
            },
            "sources": {},
        }
        merged_df = None

    output["per_sample"].update(summary_entries)
    output["per_sample"].update({sample_name: _missing_entry() for sample_name in missing_sample_names})
    output["sources"].update({sample_name: "summary" for sample_name in summary_entries})
    output["sources"].update({sample_name: "missing" for sample_name in missing_sample_names})

    print(json.dumps(output, indent=2, ensure_ascii=True))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, ensure_ascii=True))

    if merged_df is not None:
        merged_csv.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(merged_csv, index=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoCoMo stats CLI")
    parser.add_argument(
        "--dataset",
        choices=["locomo"],
        required=True,
        help="Select stats workflow to run",
    )
    parser.add_argument("--root", help="locomo run root containing sample_* directories")
    parser.add_argument("--use-summaries", action="store_true", help="locomo only: use correctness_summary.json only")
    parser.add_argument("--output-json", help="locomo only: write aggregate json to this path")
    parser.add_argument("--merged-csv", help="locomo only: write merged judge CSV to this path")
    parser.add_argument(
        "--exclude-adversarial",
        action="store_true",
        default=True,
        help="locomo only: exclude Adversarial category when computing overall averages",
    )
    parser.add_argument(
        "--include-adversarial",
        action="store_false",
        dest="exclude_adversarial",
        help="locomo only: include Adversarial category when computing overall averages",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _run_locomo(args)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Pipeline-level aggregate runners (called by pipeline.py / orchestrator)
# ---------------------------------------------------------------------------

_AGGREGATE_SCRIPT = Path(__file__).resolve()


def _aggregate_locomo_run(run_root: Path, *, include_adversarial: bool) -> Optional[AggregateResult]:
    """Aggregate one LoCoMo run directory."""
    output_json = run_root / "_correctness_aggregate.json"
    merged_csv = run_root / "_judge_merged.csv"
    cmd = [
        sys.executable,
        str(_AGGREGATE_SCRIPT),
        "--dataset", "locomo",
        "--root", str(run_root),
        "--output-json", str(output_json),
        "--merged-csv", str(merged_csv),
    ]
    if include_adversarial:
        cmd.append("--include-adversarial")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log_event("AGGREGATE][WARN", "locomo/analysis/aggregate.py exited with non-zero status", exit_code=result.returncode)
        return None
    if not output_json.exists():
        return None
    return AggregateResult(output_json=output_json, merged_csv=merged_csv if merged_csv.exists() else None)


def maybe_aggregate_run(
    *,
    dataset: str,
    run_root: Path,
    no_judge: bool,
    include_adversarial: bool,
) -> Optional[AggregateResult]:
    """Aggregate a finished run, if there is anything to aggregate.

    Called unconditionally at the end of a run, so it has to tolerate a run
    that produced nothing -- an aborted sweep should not fail on its way out.
    """
    if no_judge:
        log_event("AGGREGATE", "Skipped because judge phase was disabled")
        return None

    log_event("AGGREGATE", "Building run-level correctness summary", dataset=dataset, run_root=run_root)
    return _aggregate_locomo_run(run_root, include_adversarial=include_adversarial)
