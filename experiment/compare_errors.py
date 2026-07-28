#!/usr/bin/env python3
"""Compare two runs question-by-question: confusion matrix + disagreement dump.

Answers "where does run A get it right and run B get it wrong (and vice versa)".
Auto-detects LongMem (<cat>/<dataset>.csv) vs LoCoMo (sample_N/*_judge_*.csv) layout.

Usage:
    python experiment/compare_errors.py <runA> <runB> [--col COL] [--dump N]
                                        [--only A_right_B_wrong|A_wrong_B_right|both_wrong]
                                        [--category CAT] [--out FILE.csv]

  <runA>/<runB>: full path, or bare run-tag (resolved under experiment/longmem/output/
                 then experiment/locomo/output/standard/).
  --dump N     : print N disagreement questions with both answers + gold.
  --out FILE   : write ALL per-question rows to CSV for offline slicing.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import pandas as pd

LONGMEM_CATS = ["single_session_user", "single_session_assistant", "single_session_preference",
                "multi_session", "knowledge_update", "temporal_reasoning"]
COL_CANDIDATES = ["correctness_4omini", "correctness_20b", "correctness_20b63",
                  "correctness_20b92", "correctness_new", "correctness"]
_SKIP = ("all_answers", "progress")
_ROOTS = ["experiment/longmem/output", "experiment/locomo/output/standard"]
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LoCoMo category id -> name (Adversarial=5 is excluded from scoring)
LOCOMO_CATS = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop", 5: "Adversarial"}


def _to01(v):
    v = str(v).strip()
    if not v or v.lower() in ("nan", "none"):
        return None
    try:
        return 1 if float(v) >= 0.5 else 0
    except ValueError:
        return 1 if v.lower() in ("1", "true", "correct", "yes") else 0


def _resolve(p: str) -> str:
    if os.path.isdir(p):
        return p
    for r in _ROOTS:
        c = os.path.join(_REPO, r, p)
        if os.path.isdir(c):
            return c
    sys.exit(f"[error] run dir not found: {p}")


def _is_longmem(d: str) -> bool:
    return any(os.path.isdir(os.path.join(d, c)) for c in LONGMEM_CATS)


def _load_longmem(d: str) -> pd.DataFrame:
    rows = []
    for cat in LONGMEM_CATS:
        for f in glob.glob(os.path.join(d, cat, "*.csv")):
            if os.path.basename(f).startswith(_SKIP):
                continue
            try:
                r = pd.read_csv(f).iloc[0]
            except Exception:
                continue
            rows.append({"key": f"{cat}/{os.path.basename(f)}", "category": cat,
                         "question": r.get("question"), "gold": r.get("answer"),
                         "answer": r.get("Generated_Answer"),
                         **{c: r[c] for c in r.index if "correct" in c}})
    return pd.DataFrame(rows)


def _locomo_catmap() -> dict:
    """(sample_index, question) -> category name, from the LoCoMo source json."""
    import json
    for p in ("dataset/locomo/locomo10.json", "experiment/locomo/data/locomo10.json",
              "data/locomo10.json", "dataset/locomo10.json"):
        fp = os.path.join(_REPO, p)
        if os.path.exists(fp):
            with open(fp, encoding="utf-8") as fh:
                data = json.load(fh)
            m = {}
            for si, s in enumerate(data):
                for q in s.get("qa", []):
                    m[(si, str(q.get("question", "")).strip())] = LOCOMO_CATS.get(
                        q.get("category"), f"cat{q.get('category')}")
            return m
    return {}


def _load_locomo(d: str) -> pd.DataFrame:
    catmap = _locomo_catmap()
    rows = []
    for sd in sorted(glob.glob(os.path.join(d, "sample_*"))):
        si = int(re.search(r"sample_(\d+)", sd).group(1))
        # prefer the judged CSV; fall back to the raw eval CSV
        files = sorted(glob.glob(os.path.join(sd, "*judge*.csv"))) or \
            sorted(glob.glob(os.path.join(sd, "*_eval_*.csv")))
        if not files:
            continue
        try:
            df = pd.read_csv(files[-1])
        except Exception:
            continue
        for i, r in df.iterrows():
            q = str(r.get("question", "")).strip()
            rows.append({"key": f"s{si}#{i}", "category": catmap.get((si, q), "?"),
                         "sample": si, "question": q,
                         "gold": r.get("gold_answer"), "answer": r.get("model_answer"),
                         **{c: r[c] for c in df.columns if "correct" in c}})
    return pd.DataFrame(rows)


def _load(d: str) -> tuple[pd.DataFrame, str]:
    if _is_longmem(d):
        return _load_longmem(d), "longmem"
    return _load_locomo(d), "locomo"


def _pick_col(a: pd.DataFrame, b: pd.DataFrame, forced):
    avail = [c for c in a.columns if "correct" in c and c in b.columns]
    if forced:
        if forced not in avail:
            sys.exit(f"[error] --col {forced} not in both runs. available: {avail}")
        return forced
    for c in COL_CANDIDATES:
        if c in avail:
            return c
    sys.exit(f"[error] no shared correctness column. available: {avail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runA")
    ap.add_argument("runB")
    ap.add_argument("--col", default=None)
    ap.add_argument("--dump", type=int, default=0, help="print N disagreement questions")
    ap.add_argument("--only", default="A_right_B_wrong",
                    choices=["A_right_B_wrong", "A_wrong_B_right", "both_wrong"])
    ap.add_argument("--category", default=None, help="filter dump/CSV to one category")
    ap.add_argument("--out", default=None, help="write all per-question rows to this CSV")
    args = ap.parse_args()

    dA, dB = _resolve(args.runA), _resolve(args.runB)
    A, bench = _load(dA)
    B, _ = _load(dB)
    if A.empty or B.empty:
        sys.exit(f"[error] empty run: A={len(A)} B={len(B)}")
    col = _pick_col(A, B, args.col)

    A["ok"] = A[col].map(_to01)
    B["ok"] = B[col].map(_to01)
    if bench == "locomo":
        A = A[A["category"] != "Adversarial"]
        B = B[B["category"] != "Adversarial"]

    m = A.merge(B[["key", "ok", "answer"]], on="key", suffixes=("_A", "_B"))
    m = m.dropna(subset=["ok_A", "ok_B"])

    print(f"A = {dA}")
    print(f"B = {dB}")
    print(f"benchmark={bench}  col={col}  matched={len(m)}\n")

    aa = int(((m.ok_A == 1) & (m.ok_B == 1)).sum())
    ab = int(((m.ok_A == 1) & (m.ok_B == 0)).sum())
    ba = int(((m.ok_A == 0) & (m.ok_B == 1)).sum())
    bb = int(((m.ok_A == 0) & (m.ok_B == 0)).sum())
    n = len(m)
    print(f"{'':22s} {'B right':>9s} {'B wrong':>9s}")
    print(f"{'A right':22s} {aa:9d} {ab:9d}")
    print(f"{'A wrong':22s} {ba:9d} {bb:9d}")
    print(f"\nA acc={100*(aa+ab)/n:.2f}%  B acc={100*(aa+ba)/n:.2f}%  "
          f"net B-A={100*(ba-ab)/n:+.2f}pp")
    print(f"flip A→B: gained {ba}, lost {ab}  |  both wrong {bb} ({100*bb/n:.1f}%)")

    print(f"\n{'category':26s} {'n':>5s} {'A✓B✓':>6s} {'A✓B✗':>6s} {'A✗B✓':>6s} "
          f"{'A✗B✗':>6s} {'netΔ':>7s}")
    for c in sorted(m["category"].unique()):
        s = m[m["category"] == c]
        k = len(s)
        c_aa = int(((s.ok_A == 1) & (s.ok_B == 1)).sum())
        c_ab = int(((s.ok_A == 1) & (s.ok_B == 0)).sum())
        c_ba = int(((s.ok_A == 0) & (s.ok_B == 1)).sum())
        c_bb = int(((s.ok_A == 0) & (s.ok_B == 0)).sum())
        print(f"{c:26s} {k:5d} {c_aa:6d} {c_ab:6d} {c_ba:6d} {c_bb:6d} "
              f"{100*(c_ba-c_ab)/k:+6.1f}pp")

    sel = {"A_right_B_wrong": (m.ok_A == 1) & (m.ok_B == 0),
           "A_wrong_B_right": (m.ok_A == 0) & (m.ok_B == 1),
           "both_wrong": (m.ok_A == 0) & (m.ok_B == 0)}[args.only]
    sub = m[sel]
    if args.category:
        sub = sub[sub["category"] == args.category]

    if args.out:
        m.to_csv(args.out, index=False)
        print(f"\n[wrote] {args.out}  ({len(m)} rows)")

    if args.dump:
        print(f"\n=== {args.only}: {len(sub)} 題，列出前 {args.dump} ===")
        for _, r in sub.head(args.dump).iterrows():
            print(f"\n[{r['category']}] {r['key']}")
            print(f"  Q    : {str(r['question'])[:300]}")
            print(f"  GOLD : {str(r['gold'])[:300]}")
            print(f"  A ans: {str(r['answer_A'])[:400]}")
            print(f"  B ans: {str(r['answer_B'])[:400]}")


if __name__ == "__main__":
    main()
