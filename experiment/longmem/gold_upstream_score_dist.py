"""Upstream vector-score distribution for ALL gold summaries (split-embed).

Unlike summary_score_dist.py (which only sees gold that survived into the
Evidence Summary block), this recomputes the raw question<->gold cosine for
*every* gold sid directly against the summaries VDB — so we also get the score
of gold that never reached the evidence block.

Score == dot(stored_vec, normalized_query_vec), identical to
SummariesVDB.compare_by_id_raw. Verified to match evidence scores.

Gold sids are bucketed into:
    retrieved  — appeared in the question's Evidence Summary block
    missing    — has a VDB vector but was NOT selected into evidence
    no_vector  — sid not present in the VDB at all

Usage:
    EMBEDDING_DEVICE=cpu python experiment/longmem/gold_upstream_score_dist.py \
        --run split-embed --artifact-run oss-20b-0427
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import chromadb

from embeddings import embedder

DATA_ROOT = _ROOT / "experiment" / "longmem" / "script_data"
OUTPUT_ROOT = _ROOT / "experiment" / "longmem" / "output"
CATEGORIES = [
    "single_session_user", "single_session_assistant", "multi_session",
    "single_session_preference", "temporal_reasoning", "knowledge_update",
]
_ENTRY_RE = re.compile(r"\[sid=([^\]]+)\]\[score=([-0-9.eE]+)\]")
_HEADER = "### Evidence Summary"


def _is_main(p: Path) -> bool:
    s = p.stem
    bad = ("_abs", "_replay_fact", "_replay_fact_user_only", "_gold_summary")
    return not any(s.endswith(x) for x in bad) and s != "all_answers"


def _gold_sids(src: Path) -> set[str]:
    df = pd.read_csv(src)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    if "has_answer" not in df.columns:
        return set()
    out = set()
    for _, r in df[df["has_answer"] == True].iterrows():  # noqa: E712
        t = int(r["turn_index"]); role = str(r["role"]).strip().lower()
        mid = t + 1 if role == "user" else t
        out.add(f"{str(r['session_id']).strip()}:{mid}:{'u' if role == 'user' else 'a'}")
    return out


def _evidence_sids(ctx) -> set[str]:
    if not isinstance(ctx, str):
        return set()
    i = ctx.find(_HEADER)
    block = ctx[i:] if i != -1 else ctx
    return {m[0].strip() for m in _ENTRY_RE.findall(block)}


def _stats(xs):
    if not xs:
        return "n=0"
    xs = sorted(xs); n = len(xs)
    q = lambda p: xs[min(n - 1, int(p * n))]
    return (f"n={n}  mean={sum(xs)/n:.3f}  min={xs[0]:.3f}  p10={q(.1):.3f}  p25={q(.25):.3f}  "
            f"median={q(.5):.3f}  p75={q(.75):.3f}  p90={q(.9):.3f}  max={xs[-1]:.3f}")


def _hist(xs, lo, hi, bins=14, width=40):
    if not xs:
        return "  (empty)"
    step = (hi - lo) / bins
    c = [0] * bins
    for x in xs:
        b = max(0, min(bins - 1, int((x - lo) / step) if step else 0)); c[b] += 1
    mx = max(c) or 1
    return "\n".join(f"  [{lo+b*step:5.2f},{lo+(b+1)*step:5.2f})  {c[b]:5d} | {'█'*round(width*c[b]/mx)}"
                     for b in range(bins))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--artifact-run", default="oss-20b-0427", help="run holding the VDB artifacts")
    ap.add_argument("--exclude-ku", action="store_true")
    args = ap.parse_args()
    run_root = OUTPUT_ROOT / args.run
    art_root = OUTPUT_ROOT / args.artifact_run

    retrieved, missing = [], []
    no_vector = 0
    no_artifacts = 0
    n_gold = 0
    max_diff = 0.0  # validation: recomputed vs evidence score for retrieved gold

    for cat in CATEGORIES:
        if args.exclude_ku and cat == "knowledge_update":
            continue
        cdir = run_root / cat
        if not cdir.exists():
            continue
        for p in sorted(cdir.glob("*.csv")):
            if not _is_main(p):
                continue
            df = pd.read_csv(p)
            if "Retrieved_Context" not in df.columns:
                continue
            row = df.iloc[0]
            src = DATA_ROOT / cat / f"{p.stem}.csv"
            if not src.exists():
                continue
            gold = _gold_sids(src)
            if not gold:
                continue
            vdb_path = art_root / cat / f"artifacts_{p.stem}" / "summaries_chroma"
            if not vdb_path.exists():
                no_artifacts += 1
                continue

            ctx = row["Retrieved_Context"]
            ev = {m[0].strip(): float(m[1]) for m in _ENTRY_RE.findall(
                ctx[ctx.find(_HEADER):] if isinstance(ctx, str) and _HEADER in ctx else "")}
            ev_sids = set(ev)

            try:
                col = chromadb.PersistentClient(path=str(vdb_path)).get_collection("summaries")
            except Exception:
                no_artifacts += 1
                continue
            qv = embedder.embed([str(row["question"])])[0].astype(np.float32)
            qv = qv / (np.linalg.norm(qv) or 1.0)

            got = col.get(ids=sorted(gold), include=["embeddings"])
            vec_by_id = {i: np.asarray(v, dtype=np.float32) for i, v in zip(got["ids"], got["embeddings"])}

            for sid in gold:
                n_gold += 1
                v = vec_by_id.get(sid)
                if v is None:
                    no_vector += 1
                    continue
                sc = float(np.dot(v, qv))
                if sid in ev_sids:
                    retrieved.append(sc)
                    max_diff = max(max_diff, abs(sc - ev[sid]))
                else:
                    missing.append(sc)

    lo, hi = 0.0, 1.0
    tag = " (ex-KU)" if args.exclude_ku else ""
    print(f"\n=== {args.run}{tag} — upstream gold vector-score distribution ===")
    print(f"(artifacts: {args.artifact_run};  validation max |recomputed-evidence| = {max_diff:.4f})")
    print(f"total gold sids={n_gold}  retrieved={len(retrieved)}  missing(has vec)={len(missing)}  "
          f"no_vector={no_vector}  datasets_skipped(no artifacts)={no_artifacts}\n")
    print("RETRIEVED gold :", _stats(retrieved))
    print("MISSING  gold  :", _stats(missing))
    print(f"\n[RETRIEVED gold]\n{_hist(retrieved, lo, hi)}")
    print(f"\n[MISSING gold (had a vector but not selected)]\n{_hist(missing, lo, hi)}")


if __name__ == "__main__":
    main()
