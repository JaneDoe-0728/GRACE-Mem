"""Judge raw LoCoMo eval CSVs directly with gpt-4o-mini.

讀 sample_<n>/sample<N>_eval_<tag>.csv,寫 sample<N>_eval_<tag>_judge_4omini.csv。

模式:
  預設(單票)      → 寫 correctness_4omini(temp 0 單票);仍可用 rejudge_3vote_4omini.py 接續。
  --first(合成)   → 對齊 LongMem rejudge_output_dirs 的 --first:每題先單票,判對即
                     carry、判錯才補足 --votes 票多數決,直接寫 correctness_3vote(同時保留
                     correctness_4omini 單票欄)。一個指令搞定,不需再跑 rejudge_3vote。
                     LoCoMo 無 abs 題,不做棄答分流。

Row-level concurrency via --workers(每 thread 自己的 LLMClient);resumable(目標欄已
0/1 的列跳過)。OpenAI 高併發沒問題。

Usage:
    # 合成口徑一鍵(建議):
    python experiment/locomo/judge_eval_4omini.py <tag> --first --votes 3 --workers 48
    # 純單票(舊行為):
    python experiment/locomo/judge_eval_4omini.py <tag> --workers 48
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from KG.llm import LLMClient
from experiment.locomo.rejudge_4omini import NEW_COL, _openai_key, judge_4omini
from experiment.locomo.stages.judge import compute_correctness_stats

OUT = _ROOT / "experiment" / "locomo" / "output" / "standard"
V3_COL = "correctness_3vote"

_tls = threading.local()
_FIRST = False
_VOTES = 3


def _get_llm():
    if getattr(_tls, "llm", None) is None:
        _tls.llm = LLMClient(base_url="https://api.openai.com/v1",
                             model_name="gpt-4o-mini", api_key=_openai_key())
    return _tls.llm


def _judge_row(job):
    """回傳 (i, single, final)。--first 時 final = 合成口徑(對→carry、錯→votes 票);
    否則 final=None(只單票)。"""
    i, q, gold, gen = job
    single = judge_4omini(_get_llm(), question=q, gold=gold, gen=gen)  # votes=1
    if not _FIRST:
        return i, single, None
    final = single if single == 1 else judge_4omini(
        _get_llm(), question=q, gold=gold, gen=gen, votes=_VOTES)
    return i, single, final


def _as01(x):
    try:
        return int(float(str(x).strip()))
    except (ValueError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_tags", nargs="+")
    ap.add_argument("--workers", type=int, default=64,
                    help="Row-level concurrency to OpenAI(Tier5 可放大)")
    ap.add_argument("--first", action="store_true",
                    help="合成口徑:單票→對即 carry、錯才補 --votes 票,直接寫 correctness_3vote")
    ap.add_argument("--votes", type=int, default=3, help="--first 判錯時補足的票數")
    args = ap.parse_args()

    global _FIRST, _VOTES
    _FIRST, _VOTES = args.first, args.votes
    target_col = V3_COL if _FIRST else NEW_COL

    for tag in args.run_tags:
        base = OUT / tag
        frames = []
        for sdir in sorted(base.glob("sample_*")):
            ec = next((f for f in sdir.glob("*_eval_*.csv")
                       if "_judge" not in f.name), None)
            if ec is None:
                continue
            out_csv = ec.with_name(ec.stem + "_judge_4omini.csv")
            src = out_csv if out_csv.exists() else ec
            df = pd.read_csv(src, encoding="utf-8-sig")
            if NEW_COL not in df.columns:
                df[NEW_COL] = ""
            if _FIRST and V3_COL not in df.columns:
                df[V3_COL] = ""

            jobs = []
            for i, row in df.iterrows():
                if _as01(row.get(target_col)) in (0, 1):  # resumable:目標欄已判
                    continue
                q = str(row.get("question", "")).strip()
                gen = str(row.get("model_answer", "")).strip()
                if not q or not gen:
                    continue
                jobs.append((i, q, str(row.get("gold_answer", "")).strip(), gen))

            judged = carried = 0
            if jobs:
                with ThreadPoolExecutor(max_workers=args.workers) as ex_:
                    for i, single, final in ex_.map(_judge_row, jobs):
                        df.at[i, NEW_COL] = single
                        if _FIRST:
                            df.at[i, V3_COL] = final
                            if single == 1:
                                carried += 1
                        judged += 1
                        if judged % 100 == 0:
                            df.to_csv(out_csv, index=False)
            df.to_csv(out_csv, index=False)
            extra = f" carried={carried}" if _FIRST else ""
            print(f"  {tag}/{sdir.name}: judged={judged} n={len(df)}{extra}", flush=True)
            d2 = df.copy()
            d2["correctness"] = pd.to_numeric(d2[target_col], errors="coerce")
            frames.append(d2)

        if frames:
            alldf = pd.concat(frames, ignore_index=True)
            for col in ("f1", "bleu1"):
                if col not in alldf.columns:
                    alldf[col] = pd.NA
            stats = compute_correctness_stats(alldf, exclude_adversarial=True)
            (base / "_correctness_aggregate_4omini.json").write_text(
                json.dumps({"root": str(base), "judge_model": "gpt-4o-mini",
                            "col": target_col, "overall": stats}, indent=2))
            lab = next((c for c in alldf.columns if c.lower() == "category_label"), None)
            print(f"[AGG] {tag} ({target_col}): {stats.get('avg_correctness_percent')}%", flush=True)
            if lab:
                per = collections.defaultdict(lambda: [0, 0])
                for _, r in alldf.iterrows():
                    label = str(r.get(lab, "")).strip()
                    if label.lower() == "adversarial":
                        continue
                    v = _as01(r.get(target_col))
                    if v is None:
                        continue
                    per[label][0] += v
                    per[label][1] += 1
                for label in sorted(per):
                    c, n = per[label]
                    print(f"    {label:<18}{c}/{n}  {100*c/n:.1f}%", flush=True)


if __name__ == "__main__":
    main()
