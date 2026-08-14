"""The resampling control arm: answer the source run's questions once more, using
the stored Retrieved_Context completely untouched.

Why: the flip count of any clean answer-only intervention (a strict prompt, table
compilation, a different model) is contaminated by answer-resampling noise -- the
existing measurement shows 14.5% of answers flip even with identical sids. This
script is the no-op control arm: same question, same context, zero changes,
answered again, which measures the pure resampling break/fix rate. An
intervention's real effect is its arm minus this one.

Usage:
    LLM_API=http://localhost:1234/v1 MODEL_NAME=gpt-oss-20b \
    python -m experiment.longmem.analysis.agent_filter_resample \
        --source-run adjudicate-v1 --run-tag evext-resample \
        --names-file /tmp/evext_all229.txt --workers 2
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from grace_mem.llm import LLMClient
from experiment.longmem.stages.qa_eval import QAEvalStage

OUTPUT_ROOT = _ROOT / "experiment" / "longmem" / "output"

CATEGORIES = [
    "single_session_user", "single_session_assistant", "multi_session",
    "single_session_preference", "temporal_reasoning", "knowledge_update",
]

_tls = threading.local()


def _client():
    if getattr(_tls, "c", None) is None:
        _tls.c = LLMClient(timeout=300.0)
    return _tls.c


def process_one(src_csv: Path, out_path: Path) -> str:
    df = pd.read_csv(src_csv)
    row = df.iloc[0]
    question = str(row["question"]).strip()
    question_date = str(row.get("question_date") or "").strip() or None
    context = str(row["Retrieved_Context"])
    gold = str(row.get("answer") or "")

    stage = QAEvalStage()
    rewritten = stage.rewrite_temporal_question(question, query_time=question_date)
    answer = stage.ask_llm(_client(), question=rewritten, context=context,
                           question_date=question_date)
    frame = stage.single_result_frame(
        question=question, question_date=question_date, context=context,
        answer=answer, gold=gold, correctness="",
    )
    frame.to_csv(out_path, index=False)
    return src_csv.stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", default="adjudicate-v1")
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--names-file", required=True, help="list of category,stem pairs")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    only: set[tuple[str, str]] = set()
    for line in Path(args.names_file).read_text().splitlines():
        line = line.strip()
        if line and "," in line:
            c, n = line.split(",", 1)
            only.add((c.strip(), n.strip()))

    src_root = OUTPUT_ROOT / args.source_run
    jobs: list[tuple[Path, Path]] = []
    for cat in CATEGORIES:
        cdir = src_root / cat
        if not cdir.exists():
            continue
        out_dir = OUTPUT_ROOT / args.run_tag / cat
        for p in sorted(cdir.glob("*.csv")):
            if (cat, p.stem) not in only:
                continue
            out_path = out_dir / p.name
            if out_path.exists() and not args.force:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            jobs.append((p, out_path))

    print(f"{len(jobs)} questions (source={args.source_run}) → output/{args.run_tag}/", flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, s, o): s for s, o in jobs}
        for fut in as_completed(futs):
            done += 1
            try:
                msg = fut.result()
            except Exception as e:  # noqa: BLE001
                msg = f"{futs[fut].stem}: ERR {e}"
            if done % 10 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1)
                eta = (len(jobs) - done) / max(rate, 1e-9) / 60
                print(f"({done}/{len(jobs)}) {msg} | {rate*60:.1f}/min ETA {eta:.0f}m", flush=True)


if __name__ == "__main__":
    main()
