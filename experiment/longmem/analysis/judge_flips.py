"""撈出 judge 翻轉題並輸出人工稽核檔(CSV + Markdown)。

預設撈「單票欄判錯 → 合成欄判對」(correctness_4omini=0 → correctness_final=1)的題,
即 3 票重判 / 強化 _abs rubric 把單票誤殺救回來的題,供人工檢查是否過寬。
可用 --from-val / --to-val 反向撈(例如 1→0 看重判是否砍過頭)。

non-abs 與 abs 分組輸出(abs = 檔名以 _abs 結尾)。不呼叫任何 LLM。

Usage:
    python -m experiment.longmem.analysis.judge_flips --dirs rerank16-rr2-120b-uasplit
    # 反向(判對→判錯):
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
    lower = {c.lower().lstrip("﻿"): c for c in df.columns}
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
    md = [f"# 翻轉題稽核:{from_col}={from_val} → {to_col}={to_val}", "",
          f"Run: `{run_dir}`  |  總數 {len(rows)}(non-abs {n_non} / abs {n_abs})", "",
          "> non-abs = 3 票 general rubric;abs = 單票強化棄答 rubric。",
          "> `human_verdict` 自行填:OK / 過寬 / 邊緣。", ""]
    for grp, label in [("non-abs", "非 abs(3 票重判)"), ("abs", "abs(單票強化 rubric)")]:
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
    ap.add_argument("--dirs", nargs="+", required=True, help="run dir 名(OUTPUT_DIR 下)")
    ap.add_argument("--from-col", default="correctness_4omini", help="來源判分欄")
    ap.add_argument("--to-col", default="correctness_final", help="目標判分欄")
    ap.add_argument("--from-val", type=int, default=0, help="來源欄的值(預設 0=錯)")
    ap.add_argument("--to-val", type=int, default=1, help="目標欄的值(預設 1=對)")
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
        print(f"翻轉題 {args.from_col}={args.from_val}→{args.to_col}={args.to_val}: "
              f"{len(rows)}  (non-abs={n_non}, abs={n_abs})")
        print(f"CSV -> {csv_path}")
        print(f"MD  -> {md_path}")


if __name__ == "__main__":
    main()
