"""
Gold-session recall + accuracy metrics for a LongMem run.

This run's retrieval context ("Retrieved_Context") is an Entities/Relationships
graph, NOT a summary block: there are no [sid=...] tokens and no
"### Evidence Summary" header. The only signal that links a retrieved entity
back to the source dialogue is the "[mentioned_at:<datetime>]" tag, and in the
source data `dialogue_datetime` is 1:1 with `session_id`. So recall here is
measured at SESSION granularity (the finest we can recover from this format).

For each question (one main output CSV under output/<run>/<category>/<name>.csv):
  retrieved sessions = sessions whose dialogue_datetime appears in any
                       [mentioned_at:...] tag in Retrieved_Context
  gold sessions      = session_id of every has_answer==True turn in
                       script_data/<category>/<name>.csv

  CAVEAT: only entities carry [mentioned_at:...]; relationships and
  untimestamped entities are invisible to this mapping, so retrieved-session
  coverage is a lower bound (recall may read low even when the answer info was
  actually retrieved via an untimestamped relationship).

Metrics:
  整體正確率             = #correct / #questions
  Gold session 返回率    = Σ retrieved-gold-session / Σ gold-session   (micro)
  整題 gold 全中率       = #all-gold-session-hit / #questions-with-gold
  gold 全中的正確率      = #correct among all-gold-hit / #all-gold-hit

Usage:
    python gold_recall_metrics.py --run rerank16-rr2-120b
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(_ROOT))

import pandas as pd

DATA_ROOT = _ROOT / "experiment" / "longmem" / "script_data"
OUTPUT_ROOT = _ROOT / "experiment" / "longmem" / "output"

CATEGORIES = [
    "single_session_user",
    "single_session_assistant",
    "multi_session",
    "single_session_preference",
    "temporal_reasoning",
    "knowledge_update",
]

_MENTIONED_RE = re.compile(r"\[mentioned_at:([^\]]+)\]")



def _is_main_csv(p: Path) -> bool:
    s = p.stem
    bad = ("_abs", "_replay_fact", "_replay_fact_user_only", "_gold_summary")
    return not any(s.endswith(x) for x in bad) and s != "all_answers"



def _is_correct(v) -> bool:
    """correctness is stored as a float (1.0 / 0.0 / nan) in this run."""
    try:
        f = float(v)
        return not math.isnan(f) and f == 1.0
    except (TypeError, ValueError):
        return str(v).strip() in ("1", "1.0")



def _retrieved_sessions(context: str, dt2sess: dict[str, str]) -> set[str]:
    """Sessions whose dialogue_datetime appears in a [mentioned_at:...] tag."""
    if not isinstance(context, str):
        return set()
    out: set[str] = set()
    for m in _MENTIONED_RE.findall(context):
        sess = dt2sess.get(m.strip())
        if sess is not None:
            out.add(sess)
    return out



def _source_maps(source_csv: Path) -> tuple[set[str], dict[str, str]]:
    """Return (gold sessions, dialogue_datetime -> session_id) for a question."""
    df = pd.read_csv(source_csv)
    df.columns = [c.lstrip("﻿").lstrip("�") for c in df.columns]
    if "has_answer" not in df.columns or "session_id" not in df.columns:
        return set(), {}
    dt2sess: dict[str, str] = {}
    if "dialogue_datetime" in df.columns:
        for dt, sess in zip(df["dialogue_datetime"], df["session_id"]):
            dt2sess[str(dt).strip()] = str(sess).strip()
    gold = {str(s).strip() for s in df.loc[df["has_answer"] == True, "session_id"]}  # noqa: E712
    return gold, dt2sess



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run tag, e.g. rerank16-rr2-120b")
    ap.add_argument("--per-category", action="store_true", help="also break down per category")
    args = ap.parse_args()

    run_root = OUTPUT_ROOT / args.run
    if not run_root.exists():
        sys.exit(f"run dir not found: {run_root}")

    n_q = n_correct = 0
    gold_total = gold_hit = 0
    q_with_gold = 0
    allhit = allhit_correct = 0
    missing_src = 0
    per_cat: dict[str, dict] = {}

    for cat in CATEGORIES:
        cdir = run_root / cat
        if not cdir.exists():
            continue
        pc = per_cat.setdefault(cat, dict(n_q=0, n_correct=0, gt=0, gh=0, qg=0, ah=0, ahc=0))
        for p in sorted(cdir.glob("*.csv")):
            if not _is_main_csv(p):
                continue
            df = pd.read_csv(p)
            if "Retrieved_Context" not in df.columns:
                continue
            row = df.iloc[0]  # one question per CSV
            corr = _is_correct(row.get("correctness"))

            n_q += 1
            pc["n_q"] += 1
            if corr:
                n_correct += 1
                pc["n_correct"] += 1

            src = DATA_ROOT / cat / f"{p.stem}.csv"
            if not src.exists():
                missing_src += 1
                continue
            gold, dt2sess = _source_maps(src)
            if not gold:
                continue
            retrieved = _retrieved_sessions(row["Retrieved_Context"], dt2sess)
            hit = gold & retrieved

            gold_total += len(gold)
            gold_hit += len(hit)
            q_with_gold += 1
            pc["gt"] += len(gold)
            pc["gh"] += len(hit)
            pc["qg"] += 1

            if hit == gold:  # all gold sessions retrieved
                allhit += 1
                pc["ah"] += 1
                if corr:
                    allhit_correct += 1
                    pc["ahc"] += 1

    def pct(a, b):
        return f"{a}/{b} = {100*a/b:.1f}%" if b else f"{a}/{b} = n/a"

    print(f"\n=== run: {args.run} ===")
    if missing_src:
        print(f"(note: {missing_src} questions had no source CSV — counted in accuracy only)")
    print(f"整體正確率           {pct(n_correct, n_q)}")
    print(f"Gold session 返回率   {pct(gold_hit, gold_total)}")
    print(f"整題 gold 全中率      {pct(allhit, q_with_gold)}")
    print(f"gold 全中的正確率     {pct(allhit_correct, allhit)}")

    if args.per_category:
        print("\n--- per category ---")
        for cat, d in per_cat.items():
            print(f"\n[{cat}]")
            print(f"  整體正確率        {pct(d['n_correct'], d['n_q'])}")
            print(f"  Gold 返回率       {pct(d['gh'], d['gt'])}")
            print(f"  整題 gold 全中率   {pct(d['ah'], d['qg'])}")
            print(f"  全中的正確率       {pct(d['ahc'], d['ah'])}")



if __name__ == "__main__":
    main()
