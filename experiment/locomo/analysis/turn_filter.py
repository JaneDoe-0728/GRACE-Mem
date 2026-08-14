"""Offline analysis of a turn-granularity filter, with no LLM involved.
Chunk-level narrowing ceilings out at a precision of about 0.06 (the share of the
context that is gold turns); this script measures what precision, gold coverage
and context reduction are reachable by expanding chunks into turns and keeping a
lexical top-K.

Per question:
  retrieved chunks (selected_evidence_ids) -> expand into turns (reproducing
  ingest's filtering) -> score by keyword overlap with the question -> keep the
  top-K turns (optionally with +/-1 neighbours) -> against the gold dia_ids,
  compute turn-level coverage (counting reachable gold only, i.e. gold that lies
  inside the retrieved chunks), precision, and the reduction in context characters.

Usage:
    python -m experiment.locomo.analysis.turn_filter --run locomo-n8-full --chunk-turns 8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

DATA_JSON = _ROOT / "experiment" / "locomo" / "data" / "locomo10.json"
OUT_ROOT = _ROOT / "experiment" / "locomo" / "output" / "standard"

_DIA_RE = re.compile(r"D(\d+):(\d+)")
_SID_CTX_RE = re.compile(r"\[sid=(\d+)__(\d+):(\d+)\]")
_WORD_RE = re.compile(r"[a-z0-9']+")

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "is",
    "was", "were", "are", "be", "did", "do", "does", "what", "when", "where",
    "who", "why", "how", "which", "that", "this", "with", "as", "by", "it",
    "she", "he", "they", "her", "his", "their", "you", "we", "about", "from",
    "has", "have", "had", "not", "would", "could", "will", "been", "than",
    "into", "out", "up", "down", "over", "after", "before", "your", "my",
}


def kept_turns_by_session(sample: dict) -> dict[int, list[dict]]:
    """Reproduce ingest's empty-turn filtering and return session -> kept turns.
    Each entry holds pos, text (caption included), and dia_turn (the turn number
    from the original dia_id)."""
    conv = sample.get("conversation", {}) or {}
    out: dict[int, list[dict]] = {}
    for key, sess_turns in conv.items():
        if not key.startswith("session_") or key.endswith("_date_time") or not isinstance(sess_turns, list):
            continue
        sess = int(key.split("_", 1)[1])
        kept = []
        pos = 0
        for t in sess_turns:
            speaker = str(t.get("speaker", "")).strip()
            text = str(t.get("text", "")).strip()
            caption = str(t.get("blip_caption", "")).strip()
            if not speaker and not text and not caption:
                continue
            line = f"{speaker}: {text}"
            if caption:
                line += f" (Image: {caption})"
            m = _DIA_RE.search(str(t.get("dia_id", "")))
            kept.append({"pos": pos, "text": line,
                         "dia_turn": int(m.group(2)) if m else None})
            pos += 1
        out[sess] = kept
    return out


def tokens(s: str) -> set[str]:
    return {w for w in _WORD_RE.findall(s.lower()) if w not in _STOP and len(w) > 1}


def load_selected_pools(run: str, si: int) -> dict[str, list[tuple[int, int]]]:
    """request_id -> the (session, chunk) list selected pre-narrowing, in rank order.
    Source: evidence_split_selected.sample in logs/kg_retrieval_evidence.jsonl --
    narrowing (a WIP, on by default) cuts the context down to roughly 1.6 entries,
    so the eval CSV no longer holds the complete pool."""
    f = OUT_ROOT / run / f"sample_{si}" / "logs" / "kg_retrieval_evidence.jsonl"
    pools: dict[str, list[tuple[int, int]]] = {}
    if not f.exists():
        return pools
    for line in f.open():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("event") != "evidence_split_selected":
            continue
        pairs = []
        for item in d.get("sample") or []:
            m = re.match(rf"{si}__(\d+):(\d+)$", str(item.get("entry_id", "")))
            if m:
                pairs.append((int(m.group(1)), int(m.group(2))))
        if pairs:
            pools[str(d.get("request_id"))] = pairs
    return pools


def retrieved_chunk_sids(row: dict, sample_idx: int,
                         pools: dict[str, list[tuple[int, int]]]) -> list[tuple[int, int]]:
    """The (session, chunk) list: the evidence log's pre-narrowing pool takes
    precedence, falling back to selected_evidence_ids or a context regex."""
    rid = str(row.get("retrieval_request_id") or "")
    if rid in pools:
        return pools[rid]
    raw = str(row.get("selected_evidence_ids") or "")
    pairs = []
    for m in re.finditer(r"(\d+)__(\d+):(\d+)", raw):
        if int(m.group(1)) == sample_idx:
            pairs.append((int(m.group(2)), int(m.group(3))))
    if not pairs:
        for m in _SID_CTX_RE.finditer(str(row.get("retrieved_context") or "")):
            if int(m.group(1)) == sample_idx:
                pairs.append((int(m.group(2)), int(m.group(3))))
    seen = set(); out = []
    for p in pairs:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="locomo-n8-full")
    ap.add_argument("--chunk-turns", type=int, default=8)
    ap.add_argument("--topk", default="8,12,16,24,32")
    ap.add_argument("--neighbor", action="store_true", help="also keep the +/-1 neighbours of a matched turn")
    args = ap.parse_args()
    n = args.chunk_turns
    ks = [int(x) for x in args.topk.split(",")]

    data = json.loads(DATA_JSON.read_text())
    # One accumulator per K, plus a chunk-level baseline accumulator
    acc = {k: {"gold_hit": 0, "gold_tot": 0, "allhit_q": 0, "q": 0,
               "kept_turns": 0, "gold_kept": 0, "chars": 0} for k in ks}
    base = {"chars": 0, "turns": 0, "gold_in": 0, "gold_tot": 0, "q": 0}

    for si in range(10):
        src = OUT_ROOT / args.run / f"sample_{si}" / f"sample{si}_eval_{args.run}.csv"
        if not src.exists():
            continue
        sample = data[si]
        sess_turns = kept_turns_by_session(sample)
        # dia_turn -> kept pos mapping (gold dia_ids use the original turn numbers)
        dia2pos = {s: {t["dia_turn"]: t["pos"] for t in ts if t["dia_turn"] is not None}
                   for s, ts in sess_turns.items()}
        q2gold = {}
        for qa in sample.get("qa", []):
            ev = qa.get("evidence")
            units = []
            text = ";".join(str(e) for e in ev) if isinstance(ev, (list, tuple)) else str(ev or "")
            for m in _DIA_RE.finditer(text):
                units.append((int(m.group(1)), int(m.group(2))))
            q2gold[str(qa.get("question", "")).strip()] = units

        pools = load_selected_pools(args.run, si)
        df = pd.read_csv(src)
        for row in df.to_dict("records"):
            q = str(row.get("question", "")).strip()
            gold = q2gold.get(q, [])
            if not gold:
                continue
            chunks = retrieved_chunk_sids(row, si, pools)
            if not chunks:
                continue
            # Expand into turns
            pool = []  # (sess, pos, text)
            for sess, ci in chunks:
                for t in sess_turns.get(sess, []):
                    if t["pos"] // n == ci:
                        pool.append((sess, t["pos"], t["text"]))
            pool_keys = {(s, p) for s, p, _ in pool}
            # gold -> (sess, kept pos); reachable = lies inside the retrieved chunks
            gold_pos = [(s, dia2pos.get(s, {}).get(t)) for s, t in gold]
            gold_pos = [(s, p) for s, p in gold_pos if p is not None]
            reachable = [(s, p) for s, p in gold_pos if (s, p) in pool_keys]

            base["q"] += 1
            base["chars"] += sum(len(t) for *_, t in pool)
            base["turns"] += len(pool)
            base["gold_in"] += len(reachable)
            base["gold_tot"] += len(gold_pos)

            qtok = tokens(q)
            scored = sorted(pool, key=lambda x: len(qtok & tokens(x[2])), reverse=True)
            for k in ks:
                keep = {(s, p) for s, p, _ in scored[:k]}
                if args.neighbor:
                    for s, p, _ in scored[:k]:
                        keep.add((s, p - 1)); keep.add((s, p + 1))
                    keep &= pool_keys
                kept_txt = [(s, p, t) for s, p, t in pool if (s, p) in keep]
                hit = [g for g in reachable if g in keep]
                a = acc[k]
                a["q"] += 1
                a["gold_hit"] += len(hit)
                a["gold_tot"] += len(reachable)
                a["allhit_q"] += 1 if (reachable and len(hit) == len(reachable)) else 0
                a["kept_turns"] += len(kept_txt)
                a["gold_kept"] += len(hit)
                a["chars"] += sum(len(t) for *_, t in kept_txt)

    print(f"run={args.run} N={n} neighbor={args.neighbor}")
    print(f"baseline (all 16 chunks kept): {base['q']} q, avg {base['turns']/max(base['q'],1):.0f} turns "
          f"/ {base['chars']/max(base['q'],1):.0f} chars, reachable gold = "
          f"{base['gold_in']}/{base['gold_tot']} ({base['gold_in']/max(base['gold_tot'],1)*100:.1f}%), "
          f"turn-level precision = {base['gold_in']/max(base['turns'],1):.4f}")
    print(f"{'K':>4} {'goldcov%':>9} {'allhit%':>8} {'precision':>10} {'avg turns':>10} {'avg chars':>10} {'shrink':>7}")
    for k in ks:
        a = acc[k]
        cov = a["gold_hit"] / max(a["gold_tot"], 1) * 100
        allhit = a["allhit_q"] / max(a["q"], 1) * 100
        prec = a["gold_kept"] / max(a["kept_turns"], 1)
        print(f"{k:>4} {cov:>8.1f}% {allhit:>7.1f}% {prec:>10.4f} "
              f"{a['kept_turns']/max(a['q'],1):>10.1f} {a['chars']/max(a['q'],1):>10.0f} "
              f"{a['chars']/max(base['chars'],1)*100:>6.1f}%")


if __name__ == "__main__":
    main()
