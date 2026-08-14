"""Pure summary calculations used by benchmark aggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Sequence

if TYPE_CHECKING:
    import pandas as pd

    DataFrame = pd.DataFrame
else:
    DataFrame = Any


def _require_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit("pandas is required for LoCoMo summary calculation") from exc
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
    """Summarize a judged dataframe into overall and per-category accuracy."""
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
    """Summarize judged rows without requiring a dataframe.

    The row-based entry point, so the worker can summarize its own results
    without pandas on the critical path.
    """
    pd = _require_pandas()
    return compute_summary_from_df(pd.DataFrame(list(rows)), exclude_adversarial=exclude_adversarial)
