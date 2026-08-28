"""Fast agent-layer replay: reuse an existing run's Retrieved_Context and rerun
only what happens after the 16 summaries (grep agent -> answering), skipping
retrieval entirely -- no embedder, no reranker, no falkordb.

Why:
  1. Fast -- only LLM calls remain per question, and LM Studio can be hit from
     several threads (parallel=4)
  2. Clean -- both arms get bit-identical retrieval input, which removes
     retrieval nondeterminism and keeps attribution pure
  3. Iterable -- later agent changes (prompt, loop, parameters) all compare
     through this same path

Usage:
    python -m experiment.agent_filter.replay_run \
        --source-run rr16-base-split --run-tag rr16-grep-v3 --workers 3
    # --limit N / --category X for a small sample
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from experiment.agent_filter.harness import refine_context
from experiment.common.paths import REPO_ROOT
from experiment.longmem.stages.qa_eval import QAEvalStage
from grace_mem.llm import LLMClient

OUTPUT_ROOT = REPO_ROOT / "experiment" / "longmem" / "output"
DATA_ROOT = REPO_ROOT / "experiment" / "longmem" / "script_data"

QUESTION_CATEGORIES = [
    "single_session_user",
    "single_session_assistant",
    "multi_session",
    "single_session_preference",
    "temporal_reasoning",
    "knowledge_update",
]

_tls = threading.local()
_trace_lock = threading.Lock()  # enriched traces got large; serialize concurrent appends so interleaving cannot corrupt a line


def _llm() -> LLMClient:
    if getattr(_tls, "llm", None) is None:
        _tls.llm = LLMClient(timeout=300.0)
    return _tls.llm


def process_one(src_csv: Path, out_path: Path, trace_path: Path, cat: str,
                *, use_agent: bool, params: dict, artifact_root: Path | None = None,
                answer_system: str | None = None,
                strip_graph_context: bool = False) -> str:
    df = pd.read_csv(src_csv)
    row = df.iloc[0]
    question = str(row["question"]).strip()
    question_date = str(row.get("question_date") or "").strip() or None
    context = str(row["Retrieved_Context"])
    if strip_graph_context:
        marker = "### Evidence Summary"
        marker_idx = context.find(marker)
        if marker_idx != -1:
            context = context[marker_idx:]
    gold = str(row.get("answer") or "")

    stage = QAEvalStage()
    if answer_system:
        stage.SYSTEM_PROMPT = answer_system  # the instance attr shadows the class attr
    llm = _llm()
    rewritten = stage.rewrite_temporal_question(question, query_time=question_date)

    trace = {}
    if use_agent:
        artifact_dir = None
        if artifact_root is not None:
            cand = artifact_root / cat / f"artifacts_{src_csv.stem}"
            artifact_dir = cand if cand.exists() else None
        _t_agent = time.time()
        context, trace = refine_context(
            question=rewritten,
            context=context,
            csv_path=DATA_ROOT / cat / f"{src_csv.stem}.csv",
            llm=llm,
            question_date=question_date,
            category=cat,
            params=params,
            artifact_dir=artifact_dir,
        )
        trace["agent_ms"] = round((time.time() - _t_agent) * 1000)

    # Hypothesis recovery, productionized: the agent's self-reported HYPOTHESIS is
    # attached to the answering model as a defensive hint.
    # This replaces hyp-v1's post-hoc 4o-mini extraction -- self-consistent within
    # one model, with no external dependency.
    hyp = trace.get("hypothesis") if trace else None
    if hyp:
        context = context + (
            "\n\nNOTE: A preliminary evidence-search analysis tentatively concluded "
            f"the answer may be: \"{hyp}\". Treat this only as a hint — verify it against "
            "the evidence above; if the evidence contradicts it, trust the evidence."
        )

    answer = stage.ask_llm(llm, question=rewritten, context=context, question_date=question_date)
    stage.single_result_frame(
        question=question, question_date=question_date, context=context,
        answer=answer, gold=gold, correctness="",
    ).to_csv(out_path, index=False)
    if trace:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {"sample": src_csv.stem, "category": cat, "question": question,
             "gold": gold, "answer": answer, **trace},
            ensure_ascii=False) + "\n"
        with _trace_lock, open(trace_path, "a", encoding="utf-8") as f:
            f.write(line)

    fb = trace.get("fallback")
    suff = trace.get("sufficiency", [])
    tag = f"fb={fb}" if fb else (
        f"kept={len(trace.get('kept', []))} added={len(trace.get('added', []))} "
        f"suff={'!' if any(not s.get('sufficient', True) for s in suff) else '-'}"
    ) if trace else "no-agent"
    return f"{src_csv.stem}: {tag}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", default="rr16-base-split",
                    help="the existing run supplying Retrieved_Context (the retrieval input)")
    ap.add_argument("--source-root", default="",
                    help="optional filesystem root containing source-run (for helper runs)")
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--category", default="")
    ap.add_argument("--limit", type=int, default=0, help="max per category (0=all)")
    ap.add_argument("--artifact-root", default=os.getenv("LONGMEM_ARTIFACT_ROOT", ""),
                    help="artifacts root for the summary VDB used by the VECTOR tool. "
                         "Can also be set via the LONGMEM_ARTIFACT_ROOT env var; "
                         "an empty string disables VECTOR.")
    ap.add_argument("--names-file", default=None,
                    help="run only the listed questions, one 'category,stem' per line -- validate a fix on the error set before the full run")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from experiment.experiment_config import GREP_AGENT_PARAMS
    params = dict(GREP_AGENT_PARAMS)
    use_agent = True

    only_names: set[tuple[str, str]] | None = None
    if args.names_file:
        only_names = set()
        for line in Path(args.names_file).read_text().splitlines():
            line = line.strip()
            if line and "," in line:
                c, n = line.split(",", 1)
                only_names.add((c.strip(), n.strip()))

    src_root = (Path(args.source_root).resolve() / args.source_run
                if args.source_root else OUTPUT_ROOT / args.source_run)
    jobs: list[tuple[Path, Path, Path, str]] = []
    for cat in QUESTION_CATEGORIES:
        if args.category and cat != args.category:
            continue
        cdir = src_root / cat
        if not cdir.exists():
            continue
        out_dir = OUTPUT_ROOT / args.run_tag / cat
        out_dir.mkdir(parents=True, exist_ok=True)
        picked = 0
        for p in sorted(cdir.glob("*.csv")):
            if p.stem in ("all_answers", "progress") or p.name.endswith(".lock"):
                continue
            if only_names is not None and (cat, p.stem) not in only_names:
                continue
            out_path = out_dir / p.name
            if out_path.exists() and not args.force:
                continue
            jobs.append((p, out_path, out_dir / "_grep_agent_traces.jsonl", cat))
            picked += 1
            if args.limit and picked >= args.limit:
                break

    print(f"{len(jobs)} questions (source={args.source_run}, agent={'ON' if use_agent else 'OFF'}, "
          f"workers={args.workers}) → output/{args.run_tag}/", flush=True)
    t0 = time.time()
    done = 0
    artifact_root = Path(args.artifact_root).resolve() if args.artifact_root else None
    # Visibility: print the artifact-root and how many summary VDBs were actually
    # found, so a wrong path cannot silently switch VECTOR off
    if artifact_root is not None:
        _n_vdb = (len(list(artifact_root.glob("*/artifacts_*/summaries_chroma")))
                  if artifact_root.exists() else 0)
        print(f"artifact-root = {artifact_root}  "
              f"(summaries_chroma: {_n_vdb} -> VECTOR {'ON' if _n_vdb else 'OFF (no VDB at this path)'})",
              flush=True)
    else:
        print("artifact-root = (none) → VECTOR OFF", flush=True)
    answer_system = None
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, s, o, tp, c, use_agent=use_agent, params=params,
                          artifact_root=artifact_root, answer_system=answer_system,
                          strip_graph_context=False): s
                for s, o, tp, c in jobs}
        for fut in as_completed(futs):
            done += 1
            try:
                msg = fut.result()
            except Exception as e:
                msg = f"{futs[fut].stem}: ERR {e}"
            if done % 10 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1)
                eta = (len(jobs) - done) / max(rate, 1e-9) / 60
                print(f"({done}/{len(jobs)}) {msg} | {rate*60:.1f}/min ETA {eta:.0f}m", flush=True)


if __name__ == "__main__":
    main()
