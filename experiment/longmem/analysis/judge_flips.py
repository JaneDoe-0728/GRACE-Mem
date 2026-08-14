"""Collect the questions where the judge flipped and write them out for manual
audit, as CSV and Markdown.

By default it collects questions the single-vote column marked wrong and the
combined column marked right (correctness_4omini=0 -> correctness_final=1) --
that is, the ones the 3-vote rejudge or the strengthened _abs rubric rescued from
a single-vote misfire -- so a human can check whether that was too lenient.
Use --from-val / --to-val to collect the reverse (1 -> 0, say, to see whether the
rejudge cut too deep).

Output is grouped into non-abs and abs (abs = the filename ends in _abs). No LLM
is called.

Usage:
    python -m experiment.longmem.analysis.judge_flips --dirs rerank16-rr2-120b-uasplit
    # The reverse direction (right -> wrong):
    python -m experiment.longmem.analysis.judge_flips --dirs <run> --from-val 1 --to-val 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = _ROOT / "experiment" / "longmem" / "output"

_CATEGORY_SUBDIRS = [
    "single_session_user",
    "single_session_assistant",
    "multi_session",
    "single_session_preference",
    "temporal_reasoning",
    "knowledge_update",
]
_SKIP_FILES = {"progress.csv", "all_answers.csv"}


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower().lstrip("\ufeff"): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _val(x) -> int | None:
    s = str(x).strip()
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def collect_flips(run_dir: str, *, from_col: str, to_col: str,
                  from_val: int, to_val: int) -> list[dict]:
    base = OUTPUT_DIR / run_dir
    rows: list[dict] = []
    for sub in _CATEGORY_SUBDIRS:
        cat_dir = base / sub
        if not cat_dir.exists():
            continue
        for f in sorted(cat_dir.glob("*.csv")):
            if f.name in _SKIP_FILES:
                continue
            df = pd.read_csv(f, encoding="utf-8-sig")
            r = df.iloc[0]
            fc = _find_col(df, [from_col]); tc = _find_col(df, [to_col])
            if not fc or not tc:
                continue
            if _val(r.get(fc)) == from_val and _val(r.get(tc)) == to_val:
                rows.append(dict(
                    type="abs" if f.stem.endswith("_abs") else "non-abs",
                    category=sub, file=f.name,
                    question=str(r.get("question", "")),
                    question_date=str(r.get("question_date", "")),
                    gold=str(r.get(_find_col(df, ["answer", "gold_answer"]) or "answer", "")),
                    generated=str(r.get(_find_col(df, ["Generated_Answer", "generated_answer", "model_answer"]) or "Generated_Answer", "")),
                    **{from_col: from_val, to_col: to_val},
                    human_verdict="", note=""))
    rows.sort(key=lambda x: (x["type"] == "abs", x["category"], x["file"]))
    return rows


def write_outputs(run_dir: str, rows: list[dict], *, from_col: str, to_col: str,
                  from_val: int, to_val: int) -> tuple[Path, Path]:
    base = OUTPUT_DIR / run_dir
    tag = f"{from_col}{from_val}_to_{to_col}{to_val}"
    cols = ["type", "category", "file", "question", "question_date", "gold",
            "generated", from_col, to_col, "human_verdict", "note"]
    csv_path = base / f"flips_{tag}.csv"
    pd.DataFrame(rows, columns=cols).to_csv(csv_path, index=False)

    n_non = sum(1 for r in rows if r["type"] == "non-abs")
    n_abs = sum(1 for r in rows if r["type"] == "abs")
    md = [f"# Flip audit: {from_col}={from_val} -> {to_col}={to_val}", "",
          f"Run: `{run_dir}`  |  total {len(rows)} (non-abs {n_non} / abs {n_abs})", "",
          "> non-abs = 3-vote general rubric; abs = single-vote strengthened abstention rubric.",
          "> Fill in `human_verdict` yourself: OK / too lenient / borderline.", ""]
    for grp, label in [("non-abs", "non-abs (3-vote rejudge)"), ("abs", "abs (single-vote strengthened rubric)")]:
        md.append(f"\n---\n\n## {label}\n")
        idx = 0
        for r in rows:
            if r["type"] != grp:
                continue
            idx += 1
            gen = r["generated"].replace("\n", "\n  ")
            md += [f"### [{grp} #{idx}] {r['category']} / `{r['file']}`", "",
                   f"- **Q**: {r['question']}",
                   f"- **question_date**: {r['question_date']}",
                   f"- **GOLD**: {r['gold']}",
                   "- **GENERATED**:", "", "  ```", "  " + gen, "  ```", "",
                   "- human_verdict: ______   note: ______", ""]
    md_path = base / f"flips_{tag}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="run directory names, under OUTPUT_DIR")
    ap.add_argument("--from-col", default="correctness_4omini", help="the source scoring column")
    ap.add_argument("--to-col", default="correctness_final", help="the target scoring column")
    ap.add_argument("--from-val", type=int, default=0, help="value in the source column (default 0 = wrong)")
    ap.add_argument("--to-val", type=int, default=1, help="value in the target column (default 1 = right)")
    args = ap.parse_args()

    for run_dir in args.dirs:
        if not (OUTPUT_DIR / run_dir).exists():
            print(f"[MISS] {run_dir}: dir not found")
            continue
        rows = collect_flips(run_dir, from_col=args.from_col, to_col=args.to_col,
                             from_val=args.from_val, to_val=args.to_val)
        n_non = sum(1 for r in rows if r["type"] == "non-abs")
        n_abs = sum(1 for r in rows if r["type"] == "abs")
        csv_path, md_path = write_outputs(run_dir, rows, from_col=args.from_col,
                                          to_col=args.to_col, from_val=args.from_val,
                                          to_val=args.to_val)
        print(f"\n=== {run_dir} ===")
        print(f"flips {args.from_col}={args.from_val} -> {args.to_col}={args.to_val}: "
              f"{len(rows)}  (non-abs={n_non}, abs={n_abs})")
        print(f"CSV -> {csv_path}")
        print(f"MD  -> {md_path}")


if __name__ == "__main__":
    main()
