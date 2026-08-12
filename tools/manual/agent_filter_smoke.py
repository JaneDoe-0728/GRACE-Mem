"""Standalone smoke test for the grep agent harness.

模擬「檢索結果不完美」的情境:seed 候選故意放 幾個 distractor + 部分 gold,
看 agent 能否 (a) 保住 gold、(b) grep 補回缺的 gold、(c) 丟掉 distractor。

Usage:
    python -m tools.manual.agent_filter_smoke \
        --category single_session_assistant --name <stem> [--drop-gold 1] [--n-distractors 12]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from grace_mem.llm import LLMClient
from experiment.agent_filter.corpus import load_corpus
from experiment.agent_filter.harness import refine_context
from experiment.common.evaluation.oracle import longmem_gold_sids

DATA_ROOT = _ROOT / "experiment" / "longmem" / "script_data"


def fake_retrieved_context(src_csv: Path, *, drop_gold: int, n_distractors: int) -> tuple[str, list[str], list[str]]:
    corpus = load_corpus(src_csv)
    frame = pd.read_csv(src_csv, encoding="utf-8-sig")
    gold = corpus.normalize_sids(longmem_gold_sids(frame))
    kept_gold = gold[:-drop_gold] if drop_gold else gold
    gold_set = set(gold)
    distractors = [t.sid for t in corpus.turns[:: max(1, len(corpus.turns) // (n_distractors * 3))]
                   if t.sid not in gold_set][:n_distractors]
    seed = kept_gold + distractors
    lines = ["=== Entities ===", "- (omitted)", "", "### Evidence Summary"]
    for s in seed:
        t = corpus.resolve(s)[0]
        lines.append(f"  • [{t.date}][sid={s}][score=0.500] {corpus.display_entry(s, max_chars=300)} ")
    return "\n".join(lines), gold, seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="single_session_assistant")
    ap.add_argument("--name", default="")
    ap.add_argument("--drop-gold", type=int, default=1)
    ap.add_argument("--n-distractors", type=int, default=12)
    ap.add_argument("--mode", default="filter_fetch")
    args = ap.parse_args()

    cdir = DATA_ROOT / args.category
    src = (cdir / f"{args.name}.csv") if args.name else sorted(cdir.glob("*.csv"))[0]
    df = pd.read_csv(src)
    question = str(df["question"].dropna().iloc[0])
    qdate = str(df["question_date"].dropna().iloc[0]) if df["question_date"].notna().any() else None
    print(f"Q: {question}\ngold answer: {df['answer'].dropna().iloc[0] if df['answer'].notna().any() else '?'}\n")

    context, gold, seed = fake_retrieved_context(src, drop_gold=args.drop_gold, n_distractors=args.n_distractors)
    print(f"gold sids       : {gold}")
    print(f"seed (candidates): {len(seed)} sids, gold in seed = {sorted(set(seed) & set(gold))}\n")

    from experiment.experiment_config import GREP_AGENT_PARAMS
    params = {**GREP_AGENT_PARAMS, "grep_agent_mode": args.mode}
    refined, trace = refine_context(
        question=question,
        context=context,
        csv_path=src,
        llm=LLMClient(timeout=300.0),
        question_date=qdate,
        category=args.category,
        params=params,
    )

    print("=== TRACE ===")
    print(json.dumps({k: v for k, v in trace.items() if k != "error"}, ensure_ascii=False, indent=2)[:4000])
    if trace.get("error"):
        print("ERROR:", trace["error"])

    final = trace.get("final_sids", [])
    ctx_sids = trace.get("context_sids", final)
    gold_set = set(gold)
    print("\n=== SCORE ===")
    print(f"final sids ({len(final)}): {final}")
    print(f"context sids ({len(ctx_sids)}): {ctx_sids}")
    print(f"gold in context : {len(set(ctx_sids) & gold_set)}/{len(gold)}")
    print(f"non-gold in context: {len([s for s in ctx_sids if s not in gold_set])}")


if __name__ == "__main__":
    main()
