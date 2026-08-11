#!/usr/bin/env python3
"""Score a LongMem output folder: overall + per-category accuracy.

Reads every <category>/<dataset>.csv (one question per file, one row) under the
given run dir. By default it applies the final evaluation protocol:
``correctness_3vote`` for general questions and ``correctness_absrubric`` for
``*_abs.csv`` questions. ``--col`` intentionally scores one custom column.

Usage:
    python experiment/longmem/score_longmem.py <run_dir_or_tag> [--col COL]

  <run_dir_or_tag>: a full path, or a bare run-tag resolved under
                    experiment/longmem/output/<tag>.
  --col: optional custom correctness column for every question.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

CATS = ["single_session_user", "single_session_assistant", "single_session_preference",
        "multi_session", "knowledge_update", "temporal_reasoning"]
GENERAL_COL = "correctness_3vote"
ABSTENTION_COL = "correctness_absrubric"
_SKIP = ("all_answers", "progress")
_OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "longmem", "output")


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
    cand = os.path.join(_OUT_ROOT, p)
    if os.path.isdir(cand):
        return cand
    sys.exit(f"[error] run dir not found: {p}")


def _validate_col(run_dir: str, forced: str | None) -> None:
    cols = set()
    for c in CATS:
        for f in glob.glob(os.path.join(run_dir, c, "*.csv"))[:3]:
            if os.path.basename(f).startswith(_SKIP):
                continue
            try:
                with open(f, encoding="utf-8-sig") as fh:
                    cols |= set(next(csv.reader(fh)))
            except Exception:
                pass
    avail = sorted(x for x in cols if "correct" in x)
    if forced and forced not in cols:
        sys.exit(f"[error] --col {forced} not found. available: {avail}")
    if not forced and not {GENERAL_COL, ABSTENTION_COL} & cols:
        sys.exit(
            f"[error] final protocol columns not found; run experiment/judge.py first. "
            f"available: {avail}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--col", default=None)
    args = ap.parse_args()

    run_dir = _resolve(args.run_dir)
    _validate_col(run_dir, args.col)

    print(f"run   = {run_dir}")
    protocol = args.col or f"{GENERAL_COL} + {ABSTENTION_COL} for *_abs"
    print(f"col   = {protocol}\n")
    print(f"{'category':26s} {'n':>5s} {'correct':>8s} {'accuracy':>9s}")
    tot_c = tot_n = 0
    for c in CATS:
        vals = []
        for f in glob.glob(os.path.join(run_dir, c, "*.csv")):
            if os.path.basename(f).startswith(_SKIP):
                continue
            try:
                with open(f, encoding="utf-8-sig") as fh:
                    row = next(csv.DictReader(fh), None)
            except Exception:
                continue
            col = args.col or (ABSTENTION_COL if os.path.splitext(os.path.basename(f))[0].endswith("_abs") else GENERAL_COL)
            if not row or col not in row:
                continue
            r = _to01(row.get(col))
            if r is not None:
                vals.append(r)
        n = len(vals)
        c_ = sum(vals)
        tot_c += c_
        tot_n += n
        acc = 100 * c_ / n if n else float("nan")
        print(f"{c:26s} {n:5d} {c_:8d} {acc:8.2f}%")
    print("-" * 51)
    acc = 100 * tot_c / tot_n if tot_n else float("nan")
    print(f"{'OVERALL':26s} {tot_n:5d} {tot_c:8d} {acc:8.2f}%")


if __name__ == "__main__":
    main()
