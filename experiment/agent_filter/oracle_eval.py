"""Raw-turn oracle(grep agent 的 upside 參照)。

Context = gold turns(has_answer=True)的 **raw text**,格式與 grep agent
rebuild 的 Evidence Summary block 完全一致 → 這就是「完美 grep agent」
(檢索/定位全對)能到的正確率天花板。

特性:
  - 餵 raw turn text(非 LLMlingua 壓縮、非 user-side-only)→ 不需要 artifacts
  - question / answer / question_date 直接取自 script_data → 不需要既有 run
  - 輸出標準 run 目錄格式 → 統一 judge 工具可直接使用

Usage:
    python -m experiment.agent_filter.oracle_eval --run-tag oracle-raw
    python -m experiment.agent_filter.oracle_eval --run-tag oracle-raw --category single_session_user --limit 5
    python -m experiment.agent_filter.oracle_eval --run-tag oracle-raw --workers 4
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from KG.llm import LLMClient
from experiment.judge import LONGMEM_CATEGORIES, JudgeEngine
from experiment.agent_filter.corpus import load_corpus
from experiment.longmem.stages.qa_eval import QAEvalStage

DATA_ROOT = _ROOT / "experiment" / "longmem" / "script_data"
OUTPUT_ROOT = _ROOT / "experiment" / "longmem" / "output"

_tls = threading.local()


def _clients() -> tuple[LLMClient, LLMClient]:
    """Per-thread (answer_llm, judge_llm)。judge 比照 rerun_split_experiments 用 JUDGE_*。"""
    if getattr(_tls, "clients", None) is None:
        answer = LLMClient(timeout=300.0)
        judge = LLMClient(
            base_url=os.getenv("JUDGE_LLM_API") or None,
            model_name=os.getenv("JUDGE_MODEL_NAME") or None,
        )
        _tls.clients = (answer, judge)
    return _tls.clients


def gold_sids(src_csv: Path) -> list[str]:
    df = pd.read_csv(src_csv)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    if "has_answer" not in df.columns:
        return []
    out: list[str] = []
    for _, r in df[df["has_answer"] == True].iterrows():  # noqa: E712
        session = str(r["session_id"]).strip()
        turn = int(r["turn_index"])
        role = str(r["role"]).strip().lower()
        out.append(f"{session}:{turn + 1}:u" if role == "user" else f"{session}:{turn}:a")
    return out


def build_oracle_context(src_csv: Path) -> tuple[str, list[str]]:
    corpus = load_corpus(src_csv)
    sids = corpus.normalize_sids(gold_sids(src_csv))
    lines = ["### Evidence Summary"]
    for s in sids:
        t = corpus.resolve(s)[0]
        entry = corpus.display_entry(s)
        dt_str = f"[{t.date}]" if t.date else ""
        lines.append(f"  • {dt_str}[sid={s}][score=--] {entry} ")
    return "\n".join(lines), sids


def process_one(src_csv: Path, out_path: Path, category: str, *, no_judge: bool) -> str:
    df = pd.read_csv(src_csv)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    question = str(df["question"].dropna().iloc[0]).strip()
    gold_answer = str(df["answer"].dropna().iloc[0]).strip() if df["answer"].notna().any() else ""
    question_date = None
    if "question_date" in df.columns and df["question_date"].notna().any():
        question_date = str(df["question_date"].dropna().iloc[0]).strip()

    context, sids = build_oracle_context(src_csv)
    if not sids:
        return f"[SKIP] {src_csv.stem}: no gold turns"

    stage = QAEvalStage()
    answer_llm, judge_llm = _clients()
    rewritten = stage.rewrite_temporal_question(question, query_time=question_date)
    answer = stage.ask_llm(answer_llm, question=rewritten, context=context, question_date=question_date)

    correctness = ""
    if not no_judge and gold_answer:
        correctness = str(JudgeEngine(judge_llm, "longmem").judge(
            question=question,
            gold=gold_answer,
            generated=answer,
            category=category,
            is_abstention=src_csv.stem.endswith("_abs"),
        ))

    stage.single_result_frame(
        question=question,
        question_date=question_date,
        context=context,
        answer=answer,
        gold=gold_answer,
        correctness=correctness,
    ).to_csv(out_path, index=False)
    return f"[OK]   {src_csv.stem}: gold={len(sids)} correct={correctness or '?'} | {answer[:60]!r}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="oracle-raw")
    ap.add_argument("--category", default="", help="only this category dir")
    ap.add_argument("--name", default="", help="only this dataset name")
    ap.add_argument("--limit", type=int, default=0, help="max questions per category (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    jobs: list[tuple[Path, Path, str]] = []
    for cat_sub, category in LONGMEM_CATEGORIES.items():
        if args.category and cat_sub != args.category:
            continue
        cdir = DATA_ROOT / cat_sub
        if not cdir.exists():
            continue
        out_dir = OUTPUT_ROOT / args.run_tag / cat_sub
        out_dir.mkdir(parents=True, exist_ok=True)
        picked = 0
        for p in sorted(cdir.glob("*.csv")):
            if args.name and p.stem != args.name:
                continue
            out_path = out_dir / f"{p.stem}.csv"
            if out_path.exists() and not args.force:
                continue
            jobs.append((p, out_path, category))
            picked += 1
            if args.limit and picked >= args.limit:
                break

    print(f"{len(jobs)} questions to run → output/{args.run_tag}/")
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, src, out, cat, no_judge=args.no_judge): src for src, out, cat in jobs}
        for fut in as_completed(futs):
            done += 1
            try:
                msg = fut.result()
            except Exception as e:  # noqa: BLE001
                msg = f"[ERR]  {futs[fut].stem}: {e}"
            print(f"({done}/{len(jobs)}) {msg}", flush=True)


if __name__ == "__main__":
    main()
