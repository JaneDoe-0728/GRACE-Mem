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
  overall accuracy            = #correct / #questions
  gold session recall         = Σ retrieved-gold-session / Σ gold-session (micro)
  all-gold-hit rate           = #all-gold-session-hit / #questions-with-gold
  accuracy when all gold hit  = #correct among all-gold-hit / #all-gold-hit

Usage:
    python -m experiment.longmem.analysis.gold_recall --run rerank16-rr2-120b
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from experiment.common.recall import RecallStats, format_ratio

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
    df.columns = [c.lstrip("\ufeff").lstrip("�") for c in df.columns]
    if "has_answer" not in df.columns or "session_id" not in df.columns:
        return set(), {}
    dt2sess: dict[str, str] = {}
    if "dialogue_datetime" in df.columns:
        for dt, sess in zip(df["dialogue_datetime"], df["session_id"]):
            dt2sess[str(dt).strip()] = str(sess).strip()
    gold = {str(s).strip() for s in df.loc[df["has_answer"] == True, "session_id"]}
    return gold, dt2sess



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run tag, e.g. rerank16-rr2-120b")
    ap.add_argument("--per-category", action="store_true", help="also break down per category")
    args = ap.parse_args()

    run_root = OUTPUT_ROOT / args.run
    if not run_root.exists():
        sys.exit(f"run dir not found: {run_root}")

    total = RecallStats()
    missing_src = 0
    per_cat: dict[str, RecallStats] = {}

    for cat in CATEGORIES:
        cdir = run_root / cat
        if not cdir.exists():
            continue
        pc = per_cat.setdefault(cat, RecallStats())
        for p in sorted(cdir.glob("*.csv")):
            if not _is_main_csv(p):
                continue
            df = pd.read_csv(p)
            if "Retrieved_Context" not in df.columns:
                continue
            row = df.iloc[0]  # one question per CSV
            corr = _is_correct(row.get("correctness"))

            total.add_accuracy(correct=corr)
            pc.add_accuracy(correct=corr)

            src = DATA_ROOT / cat / f"{p.stem}.csv"
            if not src.exists():
                missing_src += 1
                continue
            gold, dt2sess = _source_maps(src)
            if not gold:
                continue
            retrieved = _retrieved_sessions(row["Retrieved_Context"], dt2sess)
            total.add_retrieval(gold=gold, retrieved=retrieved, correct=corr)
            pc.add_retrieval(gold=gold, retrieved=retrieved, correct=corr)

    print(f"\n=== run: {args.run} ===")
    if missing_src:
        print(f"(note: {missing_src} questions had no source CSV — counted in accuracy only)")
    print(f"overall accuracy            {format_ratio(total.correct, total.questions)}")
    print(f"gold session recall         {format_ratio(total.gold_hit, total.gold_total)}")
    print(f"all-gold-hit rate           {format_ratio(total.all_gold_hit, total.questions_with_gold)}")
    print(f"accuracy when all gold hit  {format_ratio(total.all_gold_hit_correct, total.all_gold_hit)}")

    if args.per_category:
        print("\n--- per category ---")
        for cat, d in per_cat.items():
            print(f"\n[{cat}]")
            print(f"  overall accuracy            {format_ratio(d.correct, d.questions)}")
            print(f"  gold recall                 {format_ratio(d.gold_hit, d.gold_total)}")
            print(f"  all-gold-hit rate           {format_ratio(d.all_gold_hit, d.questions_with_gold)}")
            print(f"  accuracy when all gold hit  {format_ratio(d.all_gold_hit_correct, d.all_gold_hit)}")



if __name__ == "__main__":
    main()
