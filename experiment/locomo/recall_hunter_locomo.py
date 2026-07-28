"""LoCoMo Recall Hunter — supplemental gold-recall diagnostic (offline).

The LoCoMo analogue of LongMem-v2's recall_hunter (experiment/longmem/recall_hunter.py).
Same question: beyond the frozen rerank16 top-16 (P0), how much NOVEL gold evidence
can supplemental routes recover? P0 is never displaced; routes only ADD chunks to a
supplemental pool with route attribution. No answering — retrieval diagnosis only.

FULLY OFFLINE — no retriever / GPU / LLM rerun:
  * P0            = the run's `selected_evidence_ids` (rerank16 top-16), already in
                    <run>/_judge_merged.csv (falls back to [sid=..] context markers).
  * gold          = locomo10.json qa[].evidence dia_ids -> chunk sids, via the exact
                    ingest-replay chunk mapping reused from locomo_gold_recall_metrics.
  * chunk universe= derived from the same turn-index + N (every pos//N that exists).

Routes (LoCoMo chunk granularity — differs from LongMem's turn-pair granularity):
  adj     ±1/±2 chunk within the same session of every P0 chunk  (local context)
  fanout  every chunk of the top-3 sessions P0 hit most           (session expansion)
  bm25    in-memory BM25 over RAW turns -> their chunk sids        (lexical route)
Dropped vs LongMem: role (no :u/:a side in LoCoMo), entity/relation/hyde (would need
a per-sample retriever+GPU rerun; LongMem showed structural routes dominate, and
LoCoMo baseline recall is already high — start with the cheap decisive cut).

Usage:
    python experiment/locomo/recall_hunter_locomo.py --run locomo-n8 --chunk-turns 8 \
        [--per-category] [--out experiment/locomo/output/recall_hunter_locomo.jsonl]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import pandas as pd

# reuse the exact gold/sid/chunk machinery that produced the 86% baseline number
from locomo_gold_recall_metrics import (  # noqa: E402
    DATASET_JSON, DEFAULT_RUN_ROOT,
    _build_turn_index, _build_question_evidence, _gold_sids, _retrieved_sids,
    _infer_chunk_turns, _sample_index, _load_judge_df,
)

QUOTA = {"adj": 32, "fanout": 32, "bm25": 16}
FANOUT_SESSIONS = 3
ROUTE_ORDER = ["fanout", "adj", "bm25"]
BUDGETS = [16, 32, 64, 128]


# --------------------------------------------------------------------------- #
# offline assets: chunk universe + raw turns per sample (from locomo10.json)   #
# --------------------------------------------------------------------------- #
def _chunk_universe(turn_index: dict, n: int) -> dict[int, dict[int, set[int]]]:
    """{sample: {session: set(chunk_idx)}} — every chunk that actually exists."""
    uni: dict[int, dict[int, set[int]]] = {}
    for si, sess_map in turn_index.items():
        per = {}
        for sess, kept in sess_map.items():
            per[sess] = {(pos // n if n > 0 else 0) for pos in kept.values()}
        uni[si] = per
    return uni


def _raw_turns(data: list, turn_index: dict, n: int) -> dict[int, list[tuple[str, str]]]:
    """{sample: [(chunk_sid, turn_text), ...]} for the BM25 route."""
    out: dict[int, list[tuple[str, str]]] = {}
    for si, sample in enumerate(data):
        conv = sample.get("conversation", {}) or {}
        rows: list[tuple[str, str]] = []
        for key, turns in conv.items():
            if not key.startswith("session_") or key.endswith("_date_time") or not isinstance(turns, list):
                continue
            sess = int(key.split("_", 1)[1])
            kept = turn_index.get(si, {}).get(sess, {})
            for turn in turns:
                m = re.match(r"D\d+:(\d+)", str(turn.get("dia_id", "")))
                if not m:
                    continue
                pos = kept.get(int(m.group(1)))
                if pos is None:
                    continue
                text = str(turn.get("text", "")).strip()
                cap = str(turn.get("blip_caption", "")).strip()
                blob = (text + " " + cap).strip()
                if not blob:
                    continue
                chunk = pos // n if n > 0 else 0
                rows.append((f"{si}__{sess}:{chunk}" if n > 0 else f"{si}__{sess}", blob))
        out[si] = rows
    return out


def _bm25_chunks(turns: list[tuple[str, str]], query: str, k: int) -> list[str]:
    """Tiny in-memory BM25 over raw turns; return distinct chunk sids (best-first)."""
    tok = lambda s: re.findall(r"[a-z0-9']+", s.lower())
    docs = [tok(t) for _sid, t in turns]
    if not docs:
        return []
    N = len(docs)
    avgdl = sum(len(d) for d in docs) / N
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    q = tok(query)
    scored = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * len(d) / avgdl))
        if s > 0:
            scored.append((s, i))
    scored.sort(reverse=True)
    seen, out = set(), []
    for _s, i in scored:
        sid = turns[i][0]
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
        if len(out) >= k:
            break
    return out


def _parse_chunk_sid(sid: str) -> tuple[int, int, int] | None:
    m = re.match(r"(\d+)__(\d+)(?::(\d+))?", sid)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)) if m else None


def _cap(seq, k):
    return list(dict.fromkeys(seq))[:k]


# --------------------------------------------------------------------------- #
# per-question route building                                                  #
# --------------------------------------------------------------------------- #
def build_routes(p0: set[str], sample_idx: int, question: str, n: int,
                 universe: dict, raw_turns: list) -> dict[str, list[str]]:
    routes: dict[str, list[str]] = {}
    uni = universe.get(sample_idx, {})

    # adj: ±1/±2 chunk within the same session
    adj = []
    for sid in p0:
        p = _parse_chunk_sid(sid)
        if not p:
            continue
        _si, sess, ch = p
        exists = uni.get(sess, set())
        for d in (-2, -1, 1, 2):
            base = f"{sample_idx}__{sess}:{ch + d}"
            if n > 0 and (ch + d) in exists and base not in p0:
                adj.append(base)
    routes["adj"] = _cap(adj, QUOTA["adj"])

    # fanout: all chunks of the top sessions P0 hit most
    sess_hits = Counter()
    for sid in p0:
        p = _parse_chunk_sid(sid)
        if p:
            sess_hits[p[1]] += 1
    fan = []
    for sess, _c in sess_hits.most_common(FANOUT_SESSIONS):
        for ch in sorted(uni.get(sess, set())):
            cand = f"{sample_idx}__{sess}:{ch}" if n > 0 else f"{sample_idx}__{sess}"
            if cand not in p0:
                fan.append(cand)
    routes["fanout"] = _cap(fan, QUOTA["fanout"])

    # bm25: lexical over raw turns -> chunk sids
    bm = [s for s in _bm25_chunks(raw_turns, question, QUOTA["bm25"] * 2) if s not in p0]
    routes["bm25"] = _cap(bm, QUOTA["bm25"])
    return routes


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--chunk-turns", default="auto")
    ap.add_argument("--per-category", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        run_dir = DEFAULT_RUN_ROOT / args.run
    if not run_dir.exists():
        sys.exit(f"run dir not found: {args.run}")

    data = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    turn_index = _build_turn_index(data)
    q_evidence = _build_question_evidence(data)
    df = _load_judge_df(run_dir)

    if str(args.chunk_turns).lower() == "auto":
        n = _infer_chunk_turns(run_dir, turn_index)
        print(f"(chunk-turns auto-detected: N={n} -> {'chunk-level' if n > 0 else 'session-level'})")
    else:
        n = int(args.chunk_turns)

    universe = _chunk_universe(turn_index, n)
    raw_turns = _raw_turns(data, turn_index, n)

    # accumulators
    n_q = gold_total = p0_hit = 0
    route_novel = defaultdict(int)      # novel gold each route finds (shared credit)
    route_uniq = defaultdict(int)       # novel gold found by ONLY this route
    route_pool = defaultdict(int)
    union_novel = 0
    budget_hit = {b: 0 for b in BUDGETS}
    per_cat = defaultdict(lambda: dict(gt=0, p0=0, union=0, nq=0))
    out_fh = open(args.out, "w", encoding="utf-8") if args.out else None

    for _, row in df.iterrows():
        sample_idx = _sample_index(row.get("sample"))
        if sample_idx is None:
            continue
        question = str(row.get("question", "")).strip()
        units = q_evidence.get(sample_idx, {}).get(question)
        if units is None:
            continue
        gold = _gold_sids(units, sample_idx, n, turn_index.get(sample_idx, {}))
        if not gold:
            continue
        p0 = _retrieved_sids(row.get("selected_evidence_ids"), row.get("retrieved_context"),
                             sample_idx, n)
        cat = str(row.get("category_label") or row.get("category") or "?").strip()

        routes = build_routes(p0, sample_idx, question, n, universe,
                              raw_turns.get(sample_idx, []))
        # attribution
        cand_routes: dict[str, list[str]] = defaultdict(list)
        for rname, sids in routes.items():
            for s in sids:
                cand_routes[s].append(rname)
        p1 = set(cand_routes)
        novel = (p1 & gold) - p0            # new gold this pool recovers

        n_q += 1
        gold_total += len(gold)
        g0 = gold & p0
        p0_hit += len(g0)
        union_novel += len(novel)
        for s in cand_routes:
            for rt in cand_routes[s]:
                route_pool[rt] += 1
            if s in novel:
                for rt in cand_routes[s]:
                    route_novel[rt] += 1
                if len(cand_routes[s]) == 1:
                    route_uniq[cand_routes[s][0]] += 1
        # union recall @ budget (priority order fanout>adj>bm25)
        ordered = []
        for rt in ROUTE_ORDER:
            for s in routes.get(rt, []):
                if s not in ordered:
                    ordered.append(s)
        for b in BUDGETS:
            take = set(ordered[:b])
            budget_hit[b] += len(g0 | ((take & gold) - p0))

        pc = per_cat[cat]
        pc["gt"] += len(gold); pc["p0"] += len(g0)
        pc["union"] += len(g0 | novel); pc["nq"] += 1

        if out_fh:
            out_fh.write(json.dumps({
                "sample": sample_idx, "category": cat, "question": question,
                "gold": sorted(gold), "p0": sorted(p0), "p0_gold": sorted(g0),
                "novel_gold": sorted(novel),
                "p1": [{"sid": s, "routes": cand_routes[s]} for s in p1],
                "route_counts": {k: len(v) for k, v in routes.items()},
            }, ensure_ascii=False) + "\n")
    if out_fh:
        out_fh.close()

    def pct(a, b):
        return f"{a}/{b} = {100 * a / b:.1f}%" if b else "n/a"

    print(f"\n=== LoCoMo Recall Hunter: {run_dir.name}  (N={n}, {n_q} questions with gold) ===")
    print(f"P0 baseline gold recall   {pct(p0_hit, gold_total)}")
    print(f"union recall (P0∪P1, uncapped)  {pct(p0_hit + union_novel, gold_total)}  "
          f"(+{100 * union_novel / gold_total:.1f}pp novel = {union_novel} chunks)")
    for b in BUDGETS:
        print(f"  union recall @+{b:<4d}    {pct(budget_hit[b], gold_total)}")

    print(f"\n--- novel gold by route (shared credit) ---")
    print(f"{'route':8s} {'novel':>7s} {'uniquely':>9s} {'pool':>7s} {'gold%':>7s}")
    for rt in ROUTE_ORDER:
        p = route_pool[rt]
        print(f"{rt:8s} {route_novel[rt]:>7d} {route_uniq[rt]:>9d} {p:>7d} "
              f"{(100 * route_novel[rt] / p if p else 0):>6.1f}%")

    if args.per_category:
        print(f"\n--- per category (P0 -> union recall) ---")
        for cat in sorted(per_cat):
            d = per_cat[cat]
            base = 100 * d["p0"] / d["gt"] if d["gt"] else 0
            uni = 100 * d["union"] / d["gt"] if d["gt"] else 0
            print(f"  {cat:16s} P0 {base:5.1f}% -> union {uni:5.1f}%  (+{uni - base:4.1f}pp, "
                  f"{d['nq']}q, {d['gt']} gold)")
    if args.out:
        print(f"\nper-question records -> {args.out}")


if __name__ == "__main__":
    main()
