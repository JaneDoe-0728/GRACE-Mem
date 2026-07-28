"""
Gold-summary recall + accuracy metrics for a LongMem run.

For each question (one main output CSV under output/<run>/<category>/<name>.csv):
  retrieved sids = [sid=...] tokens in the "### Evidence Summary" block
  gold sids      = has_answer==True turns in script_data/<category>/<name>.csv,
                   mapped to split-embed sids:
                       user turn      -> {session}:{turn_index+1}:u
                       assistant turn -> {session}:{turn_index}:a

Metrics (micro-averaged over questions with gold):
  整體正確率            = #correct / #questions
  Gold summary 返回率   = Σ retrieved-gold / Σ gold            (recall)
  Gold summary 準確率   = Σ gold-in-retrieved / Σ retrieved    (precision)
  整題 gold 全中率      = #all-gold-hit / #questions-with-gold
  gold 全中的正確率     = #correct among all-gold-hit / #all-gold-hit

Recall and precision numerators differ when a bare pair-level sid (``sess:6``)
covers more than one gold turn (``sess:6:u`` and ``sess:6:a``): recall credits
both gold turns, precision credits the single retrieved sid once.

Usage:
    python experiment/longmem/gold_recall_metrics.py --run split-embed
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
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

_SID_RE = re.compile(r"\[sid=([^\]\s]+)\]")
_EVIDENCE_HEADER = "### Evidence Summary"


def _strip_role(sid: str) -> str:
    """Drop a trailing :u / :a role suffix, leaving ``{session}:{turn}``.

    Split-embed runs emit role-tagged sids (``sess:6:u``); pair-level runs emit
    bare ``sess:6``. Comparing on the stripped form lets one metric cover both.
    """
    parts = sid.split(":")
    if len(parts) >= 3 and parts[-1] in ("u", "a"):
        return ":".join(parts[:-1])
    return sid


def _is_main_csv(p: Path) -> bool:
    # NOTE: _abs (abstention) questions ARE counted — they have their own judged
    # answer, so they belong in overall accuracy. They carry no has_answer gold,
    # so the recall block skips them naturally (empty gold -> `continue`).
    s = p.stem
    bad = ("_replay_fact", "_replay_fact_user_only", "_gold_summary")
    return not any(s.endswith(x) for x in bad) and s != "all_answers"


def _retrieved_sids(context: str) -> set[str]:
    """All [sid=...] tokens in the Evidence Summary block."""
    if not isinstance(context, str):
        return set()
    idx = context.find(_EVIDENCE_HEADER)
    block = context[idx:] if idx != -1 else context
    return {m.strip() for m in _SID_RE.findall(block)}


def _is_correct(row) -> bool:
    """True iff the row is judged correct.

    Prefers the ``correctness`` column, but falls back to any other
    ``correctness*`` column (e.g. ``correctness_4omini``) when the primary one is
    blank/NaN, so runs judged by an alternate judge still count. Values are
    stored as floats (``1.0``/``0.0``); a plain ``== "1"`` compare would miss
    ``"1.0"`` and mark everything wrong.
    """
    cols = ["correctness"] + [
        c for c in row.index if c.startswith("correctness") and c != "correctness"
    ]
    for c in cols:
        v = row.get(c)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            continue
        try:
            return int(float(s)) == 1
        except ValueError:
            return s.lower() in ("true", "correct", "yes")
    return False


def _gold_sids(source_csv: Path) -> set[str]:
    df = pd.read_csv(source_csv)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    if "has_answer" not in df.columns:
        return set()
    out: set[str] = set()
    for _, r in df[df["has_answer"] == True].iterrows():  # noqa: E712
        session = str(r["session_id"]).strip()
        turn = int(r["turn_index"])
        role = str(r["role"]).strip().lower()
        if role == "user":
            out.add(f"{session}:{turn + 1}:u")
        else:
            out.add(f"{session}:{turn}:a")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run tag, e.g. split-embed")
    ap.add_argument("--per-category", action="store_true", help="also break down per category")
    args = ap.parse_args()

    run_root = OUTPUT_ROOT / args.run
    if not run_root.exists():
        sys.exit(f"run dir not found: {run_root}")

    n_q = n_correct = 0
    gold_total = gold_hit = 0
    retr_total = retr_gold = 0
    q_with_gold = 0
    allhit = allhit_correct = 0
    missing_src = 0
    per_cat: dict[str, dict] = {}

    for cat in CATEGORIES:
        cdir = run_root / cat
        if not cdir.exists():
            continue
        pc = per_cat.setdefault(
            cat, dict(n_q=0, n_correct=0, gt=0, gh=0, rt=0, rg=0, qg=0, ah=0, ahc=0)
        )
        for p in sorted(cdir.glob("*.csv")):
            if not _is_main_csv(p):
                continue
            df = pd.read_csv(p)
            if "Retrieved_Context" not in df.columns:
                continue
            row = df.iloc[0]  # one question per CSV
            corr = _is_correct(row)

            n_q += 1
            pc["n_q"] += 1
            if corr:
                n_correct += 1
                pc["n_correct"] += 1

            src = DATA_ROOT / cat / f"{p.stem}.csv"
            if not src.exists():
                missing_src += 1
                continue
            gold = _gold_sids(src)
            if not gold:
                continue
            retrieved = _retrieved_sids(row["Retrieved_Context"])
            # Match role-tagged (split-embed) and bare pair-level sids alike: a
            # gold sid counts as retrieved on an exact hit OR a role-stripped hit.
            retrieved_stripped = {_strip_role(s) for s in retrieved}
            gold_stripped = {_strip_role(g) for g in gold}
            # Recall (gold-centric): each gold turn that was retrieved.
            hit = {
                g for g in gold
                if g in retrieved or _strip_role(g) in retrieved_stripped
            }
            # Precision (retrieved-centric): each retrieved sid that is gold. This
            # differs from |hit| when a pair-level sid covers >1 gold turn.
            retr_hit = {
                r for r in retrieved
                if r in gold or _strip_role(r) in gold_stripped
            }

            gold_total += len(gold)
            gold_hit += len(hit)
            retr_total += len(retrieved)
            retr_gold += len(retr_hit)
            q_with_gold += 1
            pc["gt"] += len(gold)
            pc["gh"] += len(hit)
            pc["rt"] += len(retrieved)
            pc["rg"] += len(retr_hit)
            pc["qg"] += 1

            if hit == gold:  # all gold retrieved
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
    print(f"Gold summary 返回率   {pct(gold_hit, gold_total)}   (recall)")
    print(f"Gold summary 準確率   {pct(retr_gold, retr_total)}   (precision)")
    print(f"整題 gold 全中率      {pct(allhit, q_with_gold)}")
    print(f"gold 全中的正確率     {pct(allhit_correct, allhit)}")

    print("\n--- 各分類正確率 ---")
    for cat, d in per_cat.items():
        if d["n_q"]:
            print(f"  {cat:26s} {pct(d['n_correct'], d['n_q'])}")

    if args.per_category:
        print("\n--- per category ---")
        for cat, d in per_cat.items():
            print(f"\n[{cat}]")
            print(f"  整體正確率        {pct(d['n_correct'], d['n_q'])}")
            print(f"  Gold 返回率       {pct(d['gh'], d['gt'])}   (recall)")
            print(f"  Gold 準確率       {pct(d['rg'], d['rt'])}   (precision)")
            print(f"  整題 gold 全中率   {pct(d['ah'], d['qg'])}")
            print(f"  全中的正確率       {pct(d['ahc'], d['ah'])}")


if __name__ == "__main__":
    main()
