"""The LoCoMo flavour of grep agent replay: reuse an existing run's
retrieved_context, let the agent refine it, then answer again with the *original*
LoCoMo answering prompt, emitting the standard eval CSV for the pipeline judge.

Porting design, kept to the minimum change:
  - corpus unit = chunk (matching the evidence sid space, sid =
    {sample}__{session}:{chunk}); chunk splitting reproduces ingest exactly
    (empty-turn filtering + pos//N, as in locomo_gold_recall_metrics)
  - the agent harness is reused untouched
    (grace_mem/agent_filter/harness.refine_context)
  - the answering prompt reproduces the original in stages/qa_eval.py, including
    the conversation_date note, to stay comparable

Usage:
    LLM_API=http://localhost:1234/v1 MODEL_NAME=openai/gpt-oss-20b \
    python -m experiment.benchmarking.post_retrieval.locomo --source-run locomo-n8-full \
        --run-tag locomo-n8-grep --chunk-turns 8 --workers 2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from experiment.experiment_config import INGEST_PARAMS
from experiment.locomo.helpers.agent_filter_corpus import build_chunk_corpus
from grace_mem.agent_filter.harness import refine_context
from grace_mem.agent_filter.retrieval.corpus import Corpus

if TYPE_CHECKING:
    from grace_mem.services.llm import LLMClient

DATA_JSON = _ROOT / "experiment" / "locomo" / "data" / "locomo10.json"
OUT_ROOT = _ROOT / "experiment" / "locomo" / "output" / "standard"

_T_TAG = re.compile(r"\[t=([^\]]+)\]")
_tls = threading.local()


def _llm() -> LLMClient:
    if getattr(_tls, "llm", None) is None:
        from grace_mem.services.llm import LLMClient

        _tls.llm = LLMClient(timeout=300.0)
    return _tls.llm


def locomo_answer(llm, question: str, kg_context: str) -> str:
    """Reproduce the original answering prompt from stages/qa_eval.py, including the
    conversation date note."""
    tags = _T_TAG.findall(kg_context)
    date_note = (
        f"\nNote: These conversations took place around {tags[-1]}. "
        "For questions about durations or how long ago something happened, "
        "calculate from this date, not from today."
    ) if tags else ""
    messages = [
        {"role": "system", "content": f"---Retrieved Context---\n{kg_context}\n------------------"},
        {"role": "user", "content": (
            "Please answer based on the retrieved knowledge graph context above. "
            f"Be concise and accurate.{date_note}\n\n"
            f"Question: {question}\n\nAnswer:"
        )},
    ]
    resp = llm.chat(messages=messages, temperature=0.0, max_tokens=1024)
    return (resp.choices[0].message.content or "").strip()


def process_row(row: dict, corpus: Corpus, params: dict, trace_fh, lock,
                artifact_dir=None) -> dict:
    q = str(row.get("question", "")).strip()
    ctx = str(row.get("retrieved_context", ""))
    llm = _llm()
    new_ctx, trace = refine_context(
        question=q, context=ctx, csv_path="", llm=llm,
        category=None, params=params, corpus=corpus,
        artifact_dir=artifact_dir,
    )
    ans = locomo_answer(llm, q, new_ctx)
    out = dict(row)
    out["retrieved_context"] = new_ctx
    out["model_answer"] = ans
    with lock:
        # Write the full trace (timing, commands, dropped), matching
        # LongMem's.
        # Keep the LongMem replay's timing and agent fields for experiment/benchmarking/evaluation/score.py.
        trace_fh.write(json.dumps({"question": q[:120], **trace}, ensure_ascii=False) + "\n")
        trace_fh.flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", default="locomo-n8-full")
    ap.add_argument("--run-tag", default="locomo-n8-grep")
    ap.add_argument("--chunk-turns", type=int, default=INGEST_PARAMS["chunk_turns"])
    ap.add_argument("--samples", default="0-9")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="max questions per sample")
    ap.add_argument("--questions-file", default=None,
                    help="CSV of (sample, question): run only the listed questions -- the error set or the retention gate")
    ap.add_argument("--granularity", choices=["chunk", "turn"], default="chunk",
                    help="corpus unit: chunk (8 turns) or turn (a single turn, for finer localization)")
    args = ap.parse_args()

    from experiment.experiment_config import GREP_AGENT_PARAMS
    params = dict(GREP_AGENT_PARAMS)

    data = json.loads(DATA_JSON.read_text())
    ids = []
    for part in args.samples.split(","):
        if "-" in part:
            a, b = part.split("-"); ids += list(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))

    for si in ids:
        src = OUT_ROOT / args.source_run / f"sample_{si}" / f"sample{si}_eval_{args.source_run}.csv"
        if not src.exists():
            print(f"[skip] sample_{si}: no source eval csv"); continue
        out_dir = OUT_ROOT / args.run_tag / f"sample_{si}"
        out_path = out_dir / f"sample{si}_eval_{args.run_tag}.csv"
        if out_path.exists():
            print(f"[skip] sample_{si}: done"); continue
        out_dir.mkdir(parents=True, exist_ok=True)
        corpus = build_chunk_corpus(data[si], si, args.chunk_turns, unit=args.granularity)
        # VECTOR tool: the summary VDB sits in the source folder at
        # sample_<si>/artifacts/summaries_chroma.
        artifact_dir = OUT_ROOT / args.source_run / f"sample_{si}" / "artifacts"
        if not (artifact_dir / "summaries_chroma").exists():
            artifact_dir = None
        print(f"  [sample_{si}] VECTOR {'ON' if artifact_dir else 'OFF (no summaries_chroma)'}",
              flush=True)
        df = pd.read_csv(src)
        rows = df.to_dict("records")
        if args.questions_file:
            only = {(str(r["sample"]), str(r["question"]).strip())
                    for _, r in pd.read_csv(args.questions_file).iterrows()}
            rows = [r for r in rows
                    if (f"sample_{si}", str(r.get("question","")).strip()) in only]
        if args.limit:
            rows = rows[: args.limit]
        print(f"sample_{si}: {len(rows)} questions, corpus={len(corpus.turns)} chunks", flush=True)
        lock = threading.Lock()
        t0 = time.time()
        results = [None] * len(rows)
        with (
            open(out_dir / "_grep_traces.jsonl", "w") as tf,
            ThreadPoolExecutor(max_workers=args.workers) as ex,
        ):
                futs = {ex.submit(process_row, r, corpus, params, tf, lock,
                                  artifact_dir): i
                        for i, r in enumerate(rows)}
                for done, fut in enumerate(as_completed(futs), start=1):
                    i = futs[fut]
                    try:
                        results[i] = fut.result()
                    except Exception as e:
                        r = dict(rows[i]); r["model_answer"] = f"(grep replay error: {e})"
                        results[i] = r
                    if done % 25 == 0 or done == len(rows):
                        rate = done / max(time.time() - t0, 1)
                        print(f"  ({done}/{len(rows)}) {rate*60:.1f}/min", flush=True)
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
