#!/usr/bin/env python3
"""Compare the retrieved top-16 (Evidence Summary sids) between two LongMem output dirs.

For each question present in both runs (matched by category + dataset filename),
extract the ordered sid list from the `Retrieved_Context` Evidence Summary block and
compare: set overlap (Jaccard / common / only-A / only-B), whether the sets are
identical, and whether the ranked order is identical.

Usage:
    python experiment/compare_retrieve_top16.py <dirA> <dirB> [--examples N]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import statistics as st

import pandas as pd

CATS = ["single_session_user", "single_session_assistant", "single_session_preference",
        "multi_session", "knowledge_update", "temporal_reasoning"]
SID_RE = re.compile(r"\[sid=([^\]]+)\]")


def sids_of(ctx: str) -> list[str]:
    i = ctx.find("Evidence Summary")
    block = ctx[i:] if i >= 0 else ctx
    return SID_RE.findall(block)


def load(d: str) -> dict:
    out = {}
    for c in CATS:
        for f in glob.glob(os.path.join(d, c, "*.csv")):
            b = os.path.basename(f)
            if b.startswith(("all_answers", "progress")):
                continue
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if "Retrieved_Context" not in df.columns or not len(df):
                continue
            out[(c, b)] = sids_of(str(df["Retrieved_Context"].iloc[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirA")
    ap.add_argument("dirB")
    ap.add_argument("--examples", type=int, default=0, help="show N most-different questions")
    args = ap.parse_args()

    A, B = load(args.dirA), load(args.dirB)
    common = sorted(set(A) & set(B))
    print(f"A = {args.dirA}")
    print(f"B = {args.dirB}")
    print(f"datasets: A={len(A)} B={len(B)} common={len(common)} "
          f"(A-only={len(set(A)-set(B))} B-only={len(set(B)-set(A))})")

    rows = []  # (cat, |A|, |B|, common, onlyA, onlyB, jaccard, set_same, order_same, key)
    for k in common:
        sa, sb = A[k], B[k]
        ssa, ssb = set(sa), set(sb)
        inter, union = ssa & ssb, ssa | ssb
        jac = len(inter) / len(union) if union else 1.0
        rows.append((k[0], len(sa), len(sb), len(inter), len(ssa - ssb), len(ssb - ssa),
                     jac, ssa == ssb, sa == sb, k))

    hdr = (f"{'category':26s} {'n':>4s} {'|A|':>4s} {'|B|':>4s} {'共同':>5s} {'僅A':>4s} "
           f"{'僅B':>4s} {'Jaccard':>8s} {'集合同%':>7s} {'順序同%':>7s}")

    def agg(rs, name):
        if not rs:
            return
        n = len(rs)
        print(f"{name:26s} {n:4d} {st.mean([r[1] for r in rs]):4.1f} {st.mean([r[2] for r in rs]):4.1f} "
              f"{st.mean([r[3] for r in rs]):5.1f} {st.mean([r[4] for r in rs]):4.1f} "
              f"{st.mean([r[5] for r in rs]):4.1f} {st.mean([r[6] for r in rs]):8.3f} "
              f"{100 * sum(r[7] for r in rs) / n:7.1f} {100 * sum(r[8] for r in rs) / n:7.1f}")

    print("\n" + hdr)
    for c in CATS:
        agg([r for r in rows if r[0] == c], c)
    print("-" * len(hdr))
    agg(rows, "ALL")

    if args.examples:
        print(f"\n=== {args.examples} 個差異最大的題(Jaccard 最低)===")
        for r in sorted(rows, key=lambda x: x[6])[:args.examples]:
            cat, key = r[0], r[9]
            print(f"  [{cat}] {key[1]}  Jaccard={r[6]:.2f} 共同={r[3]} 僅A={r[4]} 僅B={r[5]}")
            print(f"    A: {A[key]}")
            print(f"    B: {B[key]}")


if __name__ == "__main__":
    main()
