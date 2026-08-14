"""
Gold-evidence recall + accuracy metrics for a LoCoMo run.

Summary granularity (set by the ingest configuration):
  - N == 0 : one summary per whole session   -> sid  {sample}__{session}:0
  - N  > 0 : one summary per N consecutive kept turns
                                             -> sid  {sample}__{session}:{chunk}
             where chunk = (turn's position among the session's kept turns) // N.

Because a chunked run has finer summaries, recall MUST be scored at chunk
granularity: retrieving some other chunk of the right session is NOT a hit.
This script picks the granularity from --chunk-turns:
  - --chunk-turns 0    : session-level (drops the ':chunk', old behaviour)
  - --chunk-turns N>0  : chunk-level, gold turn -> its specific chunk sid
  - --chunk-turns auto : infer N from the run's summaries_meta artifacts
                         (default; falls back to session-level if not chunked)

The chunk index is reconstructed by replaying ingest's dialogue filter
(helpers/dataset.py: turns with no speaker/text/caption are dropped), so a
gold dia_id maps to exactly the chunk it was ingested into.

For each question (one row in ``<run>/_judge_merged.csv``):
  retrieved sids = selected_evidence_ids   (session- or chunk-level per mode)
  gold sids      = qa["evidence"] dia_ids  (from locomo10.json, sample+question;
                   falls back to the CSV gold_evidence_source column, which is
                   session-level only)

Metrics (aligned with the LongMem gold-recall report):
  overall accuracy            = #correct / #questions
  gold summary recall         = Σ retrieved-gold / Σ gold (micro recall)
  all-gold-hit rate           = #all-gold-hit / #questions-with-gold
  accuracy when all gold hit  = #correct among all-gold-hit / #all-gold-hit

Usage:
    python -m experiment.locomo.analysis.gold_recall --run locomo-n8
    python -m experiment.locomo.analysis.gold_recall --run rerank-v1-cache-top16 --chunk-turns 0
    python -m experiment.locomo.analysis.gold_recall --run /abs/path --chunk-turns 8 --per-category
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from experiment.common.recall import RecallStats, format_ratio

DATASET_JSON = _ROOT / "experiment" / "locomo" / "data" / "locomo10.json"
DEFAULT_RUN_ROOT = _ROOT / "experiment" / "locomo" / "output" / "standard"

_DIA_RE = re.compile(r"D(\d+):(\d+)")          # D{session}:{turn}
_SID_RE = re.compile(r"(\d+)__(\d+)(?::(\d+))?")  # {sample}__{session}[:{chunk}]
_CTX_SID_RE = re.compile(r"\[sid=([^\]]+)\]")   # [sid=0__1:0] markers in retrieved_context


def _sample_index(sample_field: object) -> int | None:
    m = re.search(r"(\d+)", str(sample_field))
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Dataset: reconstruct the kept-turn index for every dia_id (mirrors ingest)   #
# --------------------------------------------------------------------------- #
def _build_turn_index(data: list) -> dict[int, dict[int, dict[int, int]]]:
    """{sample: {session: {dia_turn: kept_position}}}.

    Replays helpers/dataset.build_session_records_from_json's filter so the
    kept_position matches ingest's dialogue offset exactly (empty turns dropped).
    """
    out: dict[int, dict[int, dict[int, int]]] = {}
    for si, sample in enumerate(data):
        conv = sample.get("conversation", {}) or {}
        per_sess: dict[int, dict[int, int]] = {}
        for key, turns in conv.items():
            if not key.startswith("session_") or key.endswith("_date_time") or not isinstance(turns, list):
                continue
            session_id = int(key.split("_", 1)[1])
            kept: dict[int, int] = {}
            pos = 0
            for turn in turns:
                speaker = str(turn.get("speaker", "")).strip()
                text = str(turn.get("text", "")).strip()
                caption = str(turn.get("blip_caption", "")).strip()
                if not speaker and not text and not caption:
                    continue  # dropped by ingest -> does not occupy a position
                m = re.match(r"D\d+:(\d+)", str(turn.get("dia_id", "")))
                if m:
                    kept[int(m.group(1))] = pos
                pos += 1
            per_sess[session_id] = kept
        out[si] = per_sess
    return out


def _evidence_units(evidence: object) -> list[tuple[int, int]]:
    """['D16:4', 'D2:1'] -> [(16, 4), (2, 1)]."""
    text = ";".join(str(e) for e in evidence) if isinstance(evidence, (list, tuple)) else str(evidence or "")
    return [(int(s), int(t)) for s, t in _DIA_RE.findall(text)]


def _build_question_evidence(data: list) -> dict[int, dict[str, list[tuple[int, int]]]]:
    """{sample: {question: [(session, turn), ...]}}."""
    out: dict[int, dict[str, list[tuple[int, int]]]] = {}
    for si, conv in enumerate(data):
        per_q: dict[str, list[tuple[int, int]]] = {}
        for q in conv.get("qa", []) or []:
            question = str(q.get("question", "")).strip()
            if question:
                per_q[question] = _evidence_units(q.get("evidence"))
        out[si] = per_q
    return out


# --------------------------------------------------------------------------- #
# sid helpers                                                                  #
# --------------------------------------------------------------------------- #
def _gold_sids(units: list[tuple[int, int]], sample_idx: int, n: int,
              turn_index: dict[int, dict[int, int]]) -> set[str]:
    out: set[str] = set()
    for sess, turn in units:
        if n <= 0:
            out.add(f"{sample_idx}__{sess}")
        else:
            pos = turn_index.get(sess, {}).get(turn)
            if pos is None:
                continue  # gold turn was dropped at ingest (all-empty) — rare
            out.add(f"{sample_idx}__{sess}:{pos // n}")
    return out


def _raw_retrieved_ids(selected: object, context: object) -> list[str]:
    """Prefer the selected_evidence_ids column; if empty/missing (a known
    logging gap in some runs), recover the sids from the [sid=...] markers
    embedded in retrieved_context (one per selected summary)."""
    if isinstance(selected, str) and selected.strip():
        try:
            ids = json.loads(selected)
        except json.JSONDecodeError:
            ids = [f"{a}__{b}" + (f":{c}" if c else "") for a, b, c in _SID_RE.findall(selected)]
        if ids:
            return [str(x) for x in ids]
    if isinstance(context, str) and context.strip():
        return _CTX_SID_RE.findall(context)
    return []


def _retrieved_sids(selected: object, context: object, sample_idx: int, n: int) -> set[str]:
    ids = _raw_retrieved_ids(selected, context)
    out: set[str] = set()
    for sid in ids:
        m = _SID_RE.match(str(sid).strip())
        if not m or int(m.group(1)) != sample_idx:
            continue
        s_i, sess, chunk = m.group(1), m.group(2), m.group(3)
        out.add(f"{s_i}__{sess}" if n <= 0 else f"{s_i}__{sess}:{chunk or 0}")
    return out


def _gold_from_csv_field(field: object, sample_idx: int) -> set[str]:
    """Fallback (session-level only): gold_evidence_source is '5,8,2'."""
    if not isinstance(field, str):
        return set()
    return {f"{sample_idx}__{s.strip()}" for s in field.split(",") if s.strip().isdigit()}


# --------------------------------------------------------------------------- #
# Run loading + N inference                                                    #
# --------------------------------------------------------------------------- #
def _load_judge_df(run_dir: Path) -> pd.DataFrame:
    merged = run_dir / "_judge_merged.csv"
    if merged.exists():
        return pd.read_csv(merged)
    frames = []
    for p in sorted(run_dir.glob("sample_*/*_judge*.csv")):
        df = pd.read_csv(p)
        if "sample" not in df.columns:
            df["sample"] = p.parent.name
        frames.append(df)
    if not frames:
        sys.exit(f"no _judge_merged.csv or sample_*/*_judge*.csv under {run_dir}")
    return pd.concat(frames, ignore_index=True)


def _infer_chunk_turns(run_dir: Path, turn_index: dict[int, dict[int, dict[int, int]]]) -> int:
    """Infer N from summaries_meta artifacts: for each session, chunks C and kept
    turns T constrain N to ceil(T/C) <= N <= floor((T-1)/(C-1)); intersect."""
    lo, hi = 1, 10 ** 9
    constrained = False
    for meta in sorted(run_dir.glob("sample_*/artifacts/summaries_meta.jsonl")):
        chunks: dict[str, set[int]] = {}
        for line in meta.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            sid = str(o.get("session_id", ""))          # '2__16'
            try:
                mid = int(o.get("message_id"))
            except (TypeError, ValueError):
                mid = 0
            chunks.setdefault(sid, set()).add(mid)
        for sid, mids in chunks.items():
            m = re.match(r"(\d+)__(\d+)", sid)
            if not m:
                continue
            si, sess = int(m.group(1)), int(m.group(2))
            t = len(turn_index.get(si, {}).get(sess, {}))
            c = len(mids)
            if t == 0 or c <= 1:
                continue
            constrained = True
            lo = max(lo, math.ceil(t / c))
            hi = min(hi, (t - 1) // (c - 1))
    if not constrained:
        return 0                     # every session is a single summary → session mode
    if lo > hi:
        print(f"⚠️  ambiguous/inconsistent chunk size (lo={lo} hi={hi}); pass --chunk-turns explicitly")
    return lo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="run dir name under experiment/locomo/output/standard, or an absolute path")
    ap.add_argument("--chunk-turns", default="auto",
                    help="0=session-level, N>0=chunk-level, 'auto'=infer from artifacts (default)")
    ap.add_argument("--per-category", action="store_true")
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
        print(f"(chunk-turns auto-detected: N={n}  -> {'chunk-level' if n > 0 else 'session-level'})")
    else:
        n = int(args.chunk_turns)

    total = RecallStats()
    per_cat: dict[str, RecallStats] = {}

    for _, row in df.iterrows():
        sample_idx = _sample_index(row.get("sample"))
        if sample_idx is None:
            continue
        cat = str(row.get("category_label") or row.get("category") or "?").strip()
        pc = per_cat.setdefault(cat, RecallStats())

        try:
            corr = float(row.get("correctness")) == 1.0
        except (TypeError, ValueError):
            corr = False
        total.add_accuracy(correct=corr)
        pc.add_accuracy(correct=corr)

        question = str(row.get("question", "")).strip()
        units = q_evidence.get(sample_idx, {}).get(question)
        if units is not None:
            gold = _gold_sids(units, sample_idx, n, turn_index.get(sample_idx, {}))
        else:  # question not matched in json -> CSV fallback (session-level)
            gold = _gold_from_csv_field(row.get("gold_evidence_source"), sample_idx)
            if n > 0 and gold:
                # can't resolve chunk without the match; degrade this row to session-level
                pass
        if not gold:
            continue

        retrieved = _retrieved_sids(row.get("selected_evidence_ids"),
                                    row.get("retrieved_context"), sample_idx, n)
        total.add_retrieval(gold=gold, retrieved=retrieved, correct=corr)
        pc.add_retrieval(gold=gold, retrieved=retrieved, correct=corr)

    print(f"\n=== run: {run_dir.name}  (granularity: {'chunk N=' + str(n) if n > 0 else 'session'}) ===")
    print(f"overall accuracy            {format_ratio(total.correct, total.questions)}")
    print(f"gold summary recall         {format_ratio(total.gold_hit, total.gold_total)}")
    print(f"all-gold-hit rate           {format_ratio(total.all_gold_hit, total.questions_with_gold)}")
    print(f"accuracy when all gold hit  {format_ratio(total.all_gold_hit_correct, total.all_gold_hit)}")

    if args.per_category:
        print("\n--- per category ---")
        for cat in sorted(per_cat):
            d = per_cat[cat]
            print(f"\n[{cat}]")
            print(f"  overall accuracy            {format_ratio(d.correct, d.questions)}")
            print(f"  gold recall                 {format_ratio(d.gold_hit, d.gold_total)}")
            print(f"  all-gold-hit rate           {format_ratio(d.all_gold_hit, d.questions_with_gold)}")
            print(f"  accuracy when all gold hit  {format_ratio(d.all_gold_hit_correct, d.all_gold_hit)}")


if __name__ == "__main__":
    main()
