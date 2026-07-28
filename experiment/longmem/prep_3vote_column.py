"""Prep the composite rejudge column before running the codex 3-vote / _abs flow.

實作 docs/JUDGING.md §三「錯題再用 3 票重判」的前置步驟:對一個 run 下的每題 CSV
建一個 composite 欄(預設 `correctness_final`),讓 rejudge_output_dirs.py 只重判
「該判的列」:

  - 非 _abs 題、單票(來源欄,預設 correctness_4omini)判「對」→ carry 1(工具會 skip)
  - 非 _abs 題判「錯」                                    → 留空 → 工具跑 3 票 general rubric
  - 全部 _abs 題(不管舊單票對錯)                        → 留空 → 工具跑單票強化 _abs rubric

carry 舊單票「對」的判決是 JUDGING.md 的省 API 近似(假設判對的 3 票不會翻錯);
_abs 題一律留空重判,因為舊 correctness_4omini 不是用強化版 _abs rubric 產的。

本腳本不呼叫任何 LLM。只新增/覆寫 composite 欄,其餘欄位一字不動。

Usage:
    python experiment/longmem/prep_3vote_column.py --dirs rerank16-rr2-120b-uasplit
    # 之後:
    python experiment/longmem/rejudge_output_dirs.py \
        --dirs rerank16-rr2-120b-uasplit --col correctness_final --votes 3 --workers 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = _ROOT / "experiment" / "longmem" / "output"

# 與 rejudge_output_dirs.py 對齊:只處理這 6 個 category 子目錄。
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
    lower = {c.lower().lstrip("﻿"): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def prep_csv(path: Path, *, out_col: str, src_col_candidates: list[str]) -> tuple[int, int, int]:
    """Returns (carried, left_empty, abs_rows)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    is_abs = path.stem.endswith("_abs")
    src = _find_col(df, src_col_candidates)

    carried = empty = abs_rows = 0
    values: list[object] = []
    for _, row in df.iterrows():
        if is_abs:
            values.append("")          # 全部 _abs 留空 → 工具走單票 _abs rubric
            empty += 1
            abs_rows += 1
            continue
        single = str(row.get(src, "")).strip() if src else ""
        if single == "1":
            values.append(1)           # carry 判對題
            carried += 1
        else:
            values.append("")          # 判錯 / 無單票 → 留空 → 工具跑 3 票
            empty += 1
    df[out_col] = values
    df.to_csv(path, index=False)
    return carried, empty, abs_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="run dir 名(OUTPUT_DIR 下)")
    ap.add_argument("--out-col", default="correctness_final",
                    help="composite 欄名(要跟 rejudge 的 --col 一致)")
    ap.add_argument("--src-col", default="correctness_4omini",
                    help="carry 判對題的來源單票欄(fallback: correctness_new/correctness)")
    args = ap.parse_args()

    src_candidates = [args.src_col, "correctness_4omini", "correctness_new", "correctness"]

    for run_dir in args.dirs:
        base = OUTPUT_DIR / run_dir
        if not base.exists():
            print(f"[MISS] {run_dir}: dir not found")
            continue
        print(f"\n=== {run_dir} (out_col={args.out_col}) ===")
        tot_c = tot_e = tot_a = tot_f = 0
        for sub in _CATEGORY_SUBDIRS:
            cat_dir = base / sub
            if not cat_dir.exists():
                continue
            csvs = [p for p in sorted(cat_dir.glob("*.csv")) if p.name not in _SKIP_FILES]
            c = e = a = 0
            for p in csvs:
                cc, ee, aa = prep_csv(p, out_col=args.out_col, src_col_candidates=src_candidates)
                c += cc; e += ee; a += aa
            tot_c += c; tot_e += e; tot_a += a; tot_f += len(csvs)
            print(f"  [{sub}] files={len(csvs)} carried={c} to_judge={e} (_abs rows={a})")
        print(f"  TOTAL files={tot_f} carried(1)={tot_c} to_judge(empty)={tot_e} _abs={tot_a}")
        print(f"  -> 下一步:python experiment/longmem/rejudge_output_dirs.py "
              f"--dirs {run_dir} --col {args.out_col} --votes 3 --workers 6")


if __name__ == "__main__":
    main()
