"""LoCoMo 版 grep agent replay:重用既有 run 的 retrieved_context,agent 精煉後
以「原版 LoCoMo 答題 prompt」重答,輸出標準 eval CSV 供 pipeline judge。

移植設計(最小改動):
  - corpus 單位 = chunk(與 evidence sid 空間一致,sid = {sample}__{session}:{chunk});
    chunk 切分完全復刻 ingest(空 turn 過濾 + pos//N,同 locomo_gold_recall_metrics)
  - agent harness 原封復用(experiment/agent_filter/harness.refine_context)
  - 答題 prompt 復刻 stages/qa_eval.py 的原版(含 conversation_date note),保可比性

Usage:
    LLM_API=http://localhost:1234/v1 MODEL_NAME=openai/gpt-oss-20b \
    python experiment/locomo/grep_replay.py --source-run locomo-n8-full \
        --run-tag locomo-n8-grep --chunk-turns 8 --workers 2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from KG.llm import LLMClient
from experiment.agent_filter.corpus import Corpus, Turn
from experiment.agent_filter.harness import refine_context
from experiment.agent_filter.ledger import append_ledger, compile_table
from experiment.agent_filter.skills import SKILLS as _SKILLS
_TEMPORAL_DET = dict((n, d) for n, d, _ in _SKILLS)["temporal-computation"]

DATA_JSON = _ROOT / "experiment" / "locomo" / "data" / "locomo10.json"
OUT_ROOT = _ROOT / "experiment" / "locomo" / "output" / "standard"

_T_TAG = re.compile(r"\[t=([^\]]+)\]")
_tls = threading.local()


def _llm() -> LLMClient:
    if getattr(_tls, "llm", None) is None:
        _tls.llm = LLMClient(timeout=300.0)
    return _tls.llm


def build_chunk_corpus(sample: dict, sample_idx: int, n: int, unit: str = "chunk") -> Corpus:
    """chunk 級 corpus:切分邏輯與 ingest 完全一致(空 turn 過濾 → pos//N)。"""
    conv = sample.get("conversation", {}) or {}
    turns: list[Turn] = []
    for key, sess_turns in conv.items():
        if not key.startswith("session_") or key.endswith("_date_time") or not isinstance(sess_turns, list):
            continue
        sess = int(key.split("_", 1)[1])
        date = str(conv.get(f"session_{sess}_date_time", "") or "")
        kept_lines: list[str] = []
        pos = 0
        chunks: dict[int, list[str]] = {}
        for t in sess_turns:
            speaker = str(t.get("speaker", "")).strip()
            text = str(t.get("text", "")).strip()
            caption = str(t.get("blip_caption", "")).strip()
            if not speaker and not text and not caption:
                continue
            line = f"{speaker}: {text}"
            if caption:
                line += f" (Image: {caption})"
            chunks.setdefault(pos // n, []).append(line)
            pos += 1
        if unit == "turn":
            # turn 粒度:每個 kept turn 獨立單位(記憶:session 級 precision 天花板
            # ~0.06,更細單位才有 localization 空間)。sid 帶 chunk 追溯。
            for ci in sorted(chunks):
                for off, line in enumerate(chunks[ci]):
                    turns.append(Turn(
                        sid=f"{sample_idx}__{sess}:{ci}t{off}",
                        session_id=f"{sample_idx}__{sess}",
                        turn_index=ci * 100 + off,
                        pos=ci * 100 + off,
                        role="dialogue",
                        date=date,
                        text=line,
                    ))
        else:
            for ci in sorted(chunks):
                turns.append(Turn(
                    sid=f"{sample_idx}__{sess}:{ci}",
                    session_id=f"{sample_idx}__{sess}",
                    turn_index=ci,
                    pos=ci,
                    role="dialogue",
                    date=date,
                    text="\n".join(chunks[ci]),
                ))
    return Corpus(turns)


def locomo_answer(llm, question: str, kg_context: str) -> str:
    """復刻 stages/qa_eval.py 的原版答題 prompt(含 conversation date note)。"""
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


_compiler_tls = threading.local()


def _compiler():
    # Ledger fact-table compiler (used by --ledger). Defaults to the main LLM
    # (LLM_API/MODEL_NAME); optionally point it at a stronger endpoint via
    # LEDGER_COMPILER_API / LEDGER_COMPILER_MODEL.
    if getattr(_compiler_tls, "c", None) is None:
        _compiler_tls.c = LLMClient(
            base_url=os.getenv("LEDGER_COMPILER_API") or None,
            model_name=os.getenv("LEDGER_COMPILER_MODEL") or None,
            timeout=300.0,
        )
    return _compiler_tls.c


def process_row(row: dict, corpus: Corpus, params: dict, trace_fh, lock,
                use_ledger: bool = False, artifact_dir=None) -> dict:
    q = str(row.get("question", "")).strip()
    ctx = str(row.get("retrieved_context", ""))
    llm = _llm()
    new_ctx, trace = refine_context(
        question=q, context=ctx, csv_path="", llm=llm,
        category=None, params=params, corpus=corpus,
        artifact_dir=artifact_dir,
    )
    if use_ledger and _TEMPORAL_DET.search(q):
        idx = new_ctx.find("### Evidence Summary")
        table = compile_table(_compiler(), new_ctx[idx:] if idx != -1 else new_ctx)
        new_ctx = append_ledger(new_ctx, table)
    ans = locomo_answer(llm, q, new_ctx)
    out = dict(row)
    out["retrieved_context"] = new_ctx
    out["model_answer"] = ans
    with lock:
        # 寫完整 trace(含 timing/commands/sufficiency/dropped),與 LongMem 的
        # replay_run 對齊,讓 score_grep_x3 能統計時間與 agent 數據。question 截短。
        trace_fh.write(json.dumps({"question": q[:120], **trace}, ensure_ascii=False) + "\n")
        trace_fh.flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", default="locomo-n8-full")
    ap.add_argument("--run-tag", default="locomo-n8-grep")
    ap.add_argument("--chunk-turns", type=int, default=8)
    ap.add_argument("--samples", default="0-9")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="max questions per sample")
    ap.add_argument("--questions-file", default=None,
                    help="CSV(sample,question):只跑清單內的題(錯題集/保持集閘)")
    ap.add_argument("--granularity", choices=["chunk", "turn"], default="chunk",
                    help="corpus 單位:chunk(8-turn) 或 turn(單 turn,更細 localization)")
    ap.add_argument("--ledger", action="store_true",
                    help="temporal-shape 問題:evidence 編譯成 dated fact table 附加(compile=120B@.34)")
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
        # VECTOR 工具:summary VDB 就在 source folder 的 sample_<si>/artifacts/summaries_chroma。
        artifact_dir = OUT_ROOT / args.source_run / f"sample_{si}" / "artifacts"
        if not (artifact_dir / "summaries_chroma").exists():
            artifact_dir = None
        print(f"  [sample_{si}] VECTOR {'ON' if artifact_dir else 'OFF(無 summaries_chroma)'}",
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
        with open(out_dir / "_grep_traces.jsonl", "w") as tf:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(process_row, r, corpus, params, tf, lock, args.ledger,
                                  artifact_dir): i
                        for i, r in enumerate(rows)}
                done = 0
                for fut in as_completed(futs):
                    i = futs[fut]
                    done += 1
                    try:
                        results[i] = fut.result()
                    except Exception as e:  # noqa: BLE001
                        r = dict(rows[i]); r["model_answer"] = f"(grep replay error: {e})"
                        results[i] = r
                    if done % 25 == 0 or done == len(rows):
                        rate = done / max(time.time() - t0, 1)
                        print(f"  ({done}/{len(rows)}) {rate*60:.1f}/min", flush=True)
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
