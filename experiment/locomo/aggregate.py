#!/usr/bin/env python3
"""Unified LoCoMo stats CLI.

- ``--dataset locomo`` aggregates judge outputs from ``sample_*`` run folders.
- ``--dataset locomo-plus`` merges flat eval CSVs from an ``eval/`` directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence

MODULE_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = MODULE_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
for _path in (EXPERIMENT_ROOT, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from locomo.models import AggregateResult
from locomo.utils.log import log_event

if TYPE_CHECKING:
    import pandas as pd

    DataFrame = pd.DataFrame
else:
    DataFrame = Any

def find_sample_dirs(root: Path) -> list[Path]:
    return sorted([path for path in root.iterdir() if path.is_dir() and path.name.startswith("sample_")])


def find_numeric_csvs(dir_path: Path) -> list[Path]:
    return sorted(
        (path for path in dir_path.glob("*.csv") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )


def read_csvs(csv_files: Sequence[Path]) -> tuple[list[DataFrame], list[str]]:
    pd = _require_pandas()
    frames: list[DataFrame] = []
    errors: list[str] = []
    for csv_path in csv_files:
        try:
            frames.append(pd.read_csv(csv_path))
        except Exception as exc:
            errors.append(f"{csv_path.name}: {exc}")
    return frames, errors


def load_dataset_items(dataset_json: Path) -> list[dict[str, Any]]:
    with dataset_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise SystemExit(f"Dataset JSON must be a list of items: {dataset_json}")
    return [item for item in data if isinstance(item, dict)]


def build_sample_lookup(items: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    return {
        index: {"category": str(item.get("category", ""))}
        for index, item in enumerate(items)
    }


def merge_eval_csvs(
    csv_files: Sequence[Path],
    sample_lookup: dict[int, dict[str, str]],
) -> tuple[DataFrame, list[str]]:
    pd = _require_pandas()
    frames: list[DataFrame] = []
    errors: list[str] = []
    for csv_path in csv_files:
        sample_id = int(csv_path.stem)
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            errors.append(f"{csv_path.name}: {exc}")
            continue
        df.insert(0, "sample_id", sample_id)
        meta = sample_lookup.get(sample_id, {})
        df["category"] = meta.get("category", "")
        frames.append(df)

    if not frames:
        raise SystemExit("No readable CSV files were available to merge.")
    return pd.concat(frames, ignore_index=True), errors


def _latest_judge_csv(sample_dir: Path) -> Path | None:
    candidates = sorted(sample_dir.glob("*_judge*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _require_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit("pandas is required for experiment/locomo/stats/aggregate.py") from exc
    return pd


def _compute_from_df(df: DataFrame, *, exclude_adversarial: bool) -> Dict[str, object]:
    pd = _require_pandas()
    if "category_label" in df.columns:
        cat_col = "category_label"
    elif "category" in df.columns:
        cat_col = "category"
    else:
        cat_col = None

    mask = pd.Series(True, index=df.index)
    if exclude_adversarial:
        if "category_label" in df.columns:
            mask &= df["category_label"].astype(str).str.strip().str.lower() != "adversarial"
        elif "category" in df.columns:
            mask &= ~df["category"].astype(str).str.strip().isin(["5", "adversarial", "Adversarial"])

    stats: Dict[str, object] = {
        "avg_correctness": None,
        "avg_correctness_percent": None,
        "count_correctness": 0,
        "sum_correctness": 0.0,
        "avg_f1": None,
        "count_f1": 0,
        "sum_f1": 0.0,
        "avg_bleu1": None,
        "count_bleu1": 0,
        "sum_bleu1": 0.0,
        "by_category": {},
    }

    if "correctness" in df.columns:
        scores = pd.to_numeric(df["correctness"], errors="coerce")
        filtered = scores[mask].dropna()
        if not filtered.empty:
            avg = float(filtered.mean())
            stats["avg_correctness"] = round(avg, 6)
            stats["avg_correctness_percent"] = round(avg * 100.0, 2)
            stats["count_correctness"] = int(filtered.shape[0])
            stats["sum_correctness"] = float(filtered.sum())

    if "f1" in df.columns:
        scores = pd.to_numeric(df["f1"], errors="coerce")
        filtered = scores[mask].dropna()
        if not filtered.empty:
            avg = float(filtered.mean())
            stats["avg_f1"] = round(avg, 6)
            stats["count_f1"] = int(filtered.shape[0])
            stats["sum_f1"] = float(filtered.sum())

    if "bleu1" in df.columns:
        scores = pd.to_numeric(df["bleu1"], errors="coerce")
        filtered = scores[mask].dropna()
        if not filtered.empty:
            avg = float(filtered.mean())
            stats["avg_bleu1"] = round(avg, 6)
            stats["count_bleu1"] = int(filtered.shape[0])
            stats["sum_bleu1"] = float(filtered.sum())

    by_category: Dict[str, Dict[str, float | int]] = {}
    if cat_col:
        cat_df = df[[cat_col]].copy()
        if "correctness" in df.columns:
            cat_df["correctness_num"] = pd.to_numeric(df["correctness"], errors="coerce")
            grouped = cat_df.dropna(subset=["correctness_num"]).groupby(cat_col)["correctness_num"].agg(["mean", "count"])
            for label, row in grouped.iterrows():
                label_str = str(label)
                mean_val = float(row["mean"])
                count_val = int(row["count"])
                by_category.setdefault(label_str, {})
                by_category[label_str].update(
                    {
                        "avg_correctness": round(mean_val, 6),
                        "avg_correctness_percent": round(mean_val * 100.0, 2),
                        "count": count_val,
                    }
                )
        if "f1" in df.columns:
            cat_df["f1_num"] = pd.to_numeric(df["f1"], errors="coerce")
            grouped = cat_df.dropna(subset=["f1_num"]).groupby(cat_col)["f1_num"].agg(["mean", "count"])
            for label, row in grouped.iterrows():
                label_str = str(label)
                by_category.setdefault(label_str, {})
                by_category[label_str]["avg_f1"] = round(float(row["mean"]), 6)
                by_category[label_str]["count_f1"] = int(row["count"])
        if "bleu1" in df.columns:
            cat_df["bleu1_num"] = pd.to_numeric(df["bleu1"], errors="coerce")
            grouped = cat_df.dropna(subset=["bleu1_num"]).groupby(cat_col)["bleu1_num"].agg(["mean", "count"])
            for label, row in grouped.iterrows():
                label_str = str(label)
                by_category.setdefault(label_str, {})
                by_category[label_str]["avg_bleu1"] = round(float(row["mean"]), 6)
                by_category[label_str]["count_bleu1"] = int(row["count"])

    stats["by_category"] = by_category
    return stats


def compute_summary_from_df(df: DataFrame, *, exclude_adversarial: bool) -> Dict[str, Any]:
    stats = _compute_from_df(df, exclude_adversarial=exclude_adversarial)
    by_category = dict(stats["by_category"])
    macro_values = [
        float(category_stats["avg_correctness"])
        for label, category_stats in by_category.items()
        if "avg_correctness" in category_stats
        and not (exclude_adversarial and str(label).strip().lower() == "adversarial")
    ]
    macro_avg = round(sum(macro_values) / len(macro_values), 6) if macro_values else None
    macro_avg_pct = round(macro_avg * 100.0, 2) if macro_avg is not None else None
    return {
        "overall": {
            "avg_correctness": stats["avg_correctness"],
            "avg_correctness_percent": stats["avg_correctness_percent"],
            "avg_f1": stats["avg_f1"],
            "avg_bleu1": stats["avg_bleu1"],
            "count": int(stats["count_correctness"]),
            "count_f1": int(stats["count_f1"]),
            "count_bleu1": int(stats["count_bleu1"]),
            "macro_avg_by_category": macro_avg,
            "macro_avg_by_category_percent": macro_avg_pct,
            "exclude_adversarial": bool(exclude_adversarial),
        },
        "by_category": by_category,
        "raw": stats,
    }


def compute_summary_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    exclude_adversarial: bool,
) -> Dict[str, Any]:
    pd = _require_pandas()
    return compute_summary_from_df(pd.DataFrame(list(rows)), exclude_adversarial=exclude_adversarial)


def aggregate_judge_csv_files(
    csv_files: Sequence[Path],
    *,
    root: Path,
    exclude_adversarial: bool,
    note: str = "overall stats only include samples with judge CSVs",
    sample_name_fn: Callable[[Path], str] | None = None,
) -> tuple[Dict[str, Any], DataFrame]:
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
    return {
        "avg_correctness": data.get("avg_correctness"),
        "avg_correctness_percent": data.get("avg_correctness_percent"),
        "avg_f1": data.get("avg_f1"),
        "avg_bleu1": data.get("avg_bleu1"),
        "by_category": data.get("by_category", {}),
        "source": str(source),
    }


def _missing_entry() -> Dict[str, object]:
    return {
        "avg_correctness": None,
        "avg_correctness_percent": None,
        "avg_f1": None,
        "avg_bleu1": None,
        "by_category": {},
        "source": "missing",
    }


def _print_skipped(errors: Sequence[str]) -> None:
    if not errors:
        return
    print(f"[WARN] Skipped {len(errors)} files due to read errors:")
    for err in errors[:10]:
        print(f"  {err}")


def _run_locomo(args: argparse.Namespace) -> None:
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


def _run_locomo_plus(args: argparse.Namespace) -> None:
    if not args.eval_dir:
        raise SystemExit("--eval-dir is required when --dataset=locomo-plus")
    if not args.dataset_json:
        raise SystemExit("--dataset-json is required when --dataset=locomo-plus")
    if not args.output:
        raise SystemExit("--output is required when --dataset=locomo-plus")

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise SystemExit(f"Eval directory not found: {eval_dir}")

    csv_files = find_numeric_csvs(eval_dir)
    if not csv_files:
        raise SystemExit(f"No numeric CSV files found in {eval_dir}")

    sample_lookup = build_sample_lookup(load_dataset_items(Path(args.dataset_json)))
    merged, errors = merge_eval_csvs(csv_files, sample_lookup)
    _print_skipped(errors)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False, encoding="utf-8")

    print(f"Merged {len(csv_files) - len(errors)} files -> {len(merged)} rows -> {output}")
    print(f"Columns: {list(merged.columns)}")
    if "category" in merged.columns:
        print("\nCategory distribution:")
        print(merged["category"].value_counts().to_string())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoCoMo stats CLI")
    parser.add_argument(
        "--dataset",
        choices=["locomo", "locomo-plus"],
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
    parser.add_argument("--eval-dir", help="locomo-plus only: directory containing numeric eval CSV files")
    parser.add_argument("--dataset-json", help="locomo-plus only: unified input dataset JSON")
    parser.add_argument("--output", help="locomo-plus only: output merged CSV path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dataset == "locomo":
        _run_locomo(args)
        return
    _run_locomo_plus(args)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Pipeline-level aggregate runners (called by pipeline.py / orchestrator)
# ---------------------------------------------------------------------------

_AGGREGATE_SCRIPT = Path(__file__).resolve()


def _aggregate_locomo_run(run_root: Path, *, include_adversarial: bool) -> Optional[AggregateResult]:
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
        log_event("AGGREGATE][WARN", "locomo/aggregate.py exited with non-zero status", exit_code=result.returncode)
        return None
    if not output_json.exists():
        return None
    return AggregateResult(output_json=output_json, merged_csv=merged_csv if merged_csv.exists() else None)


def _aggregate_locomo_plus_run(
    run_root: Path,
    judge_dir: Path,
    *,
    include_adversarial: bool,
) -> Optional[AggregateResult]:
    judge_files = sorted(
        (path for path in judge_dir.glob("*.csv") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if not judge_files:
        log_event("AGGREGATE][WARN", "No judge CSVs found", judge_dir=judge_dir)
        return None

    output, merged_df = aggregate_judge_csv_files(
        judge_files,
        root=run_root,
        exclude_adversarial=not include_adversarial,
    )
    merged_csv = run_root / "_judge_merged.csv"
    merged_df.to_csv(merged_csv, index=False)
    output_json = run_root / "_correctness_aggregate.json"
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return AggregateResult(output_json=output_json, merged_csv=merged_csv)


def maybe_aggregate_run(
    *,
    dataset: str,
    run_root: Path,
    no_judge: bool,
    judge_dir: Optional[Path],
    include_adversarial: bool,
) -> Optional[AggregateResult]:
    if no_judge:
        log_event("AGGREGATE", "Skipped because judge phase was disabled")
        return None

    log_event("AGGREGATE", "Building run-level correctness summary", dataset=dataset, run_root=run_root)
    if dataset == "locomo":
        return _aggregate_locomo_run(run_root, include_adversarial=include_adversarial)
    if judge_dir is None:
        log_event("AGGREGATE][WARN", "Judge directory is missing; cannot aggregate locomo-plus run")
        return None
    return _aggregate_locomo_plus_run(run_root, judge_dir, include_adversarial=include_adversarial)


def maybe_upload_aggregate(
    aggregate_result: Optional[AggregateResult],
    *,
    enabled: bool,
    dataset: str,
    run_root: Path,
    sample_ids: Sequence[int],
    no_judge: bool,
    include_adversarial: bool,
) -> None:
    if not enabled:
        return
    if aggregate_result is None:
        log_event("UPLOAD][WARN", "Upload requested, but no aggregate artifact was produced")
        return
    try:
        from locomo.stages.upload import upload_run_tables

        result = upload_run_tables(
            aggregate_json=aggregate_result.output_json,
            dataset=dataset,
            run_root=run_root,
            sample_ids=sample_ids,
            no_judge=no_judge,
            include_adversarial=include_adversarial,
        )
        for label, table_result in result.items():
            log_event(
                "UPLOAD",
                f"{table_result['status']} {label} table",
                table=table_result["table"],
                table_id=table_result["table_id"],
                rows=table_result["row_count"],
            )
    except Exception as exc:
        log_event("UPLOAD][WARN", "NocoDB upload failed (non-fatal)", error=exc)
