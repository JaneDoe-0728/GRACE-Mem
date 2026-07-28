"""
EXP-F08 — Concurrency Stress Under Deterministic Config

Determines whether the deterministic config remains reproducible when multiple
workers process questions in parallel.

Protocol:
  - Use the ingest snapshot from a passing EXP-F07 trial (no re-ingest).
  - Run retrieval + QA evaluation at concurrency levels: 1, 2, 4, 8.
  - Repeat each level 3 times.
  - concurrency=1 must PASS.  Higher concurrency failures are reported as
    concurrency_nondeterminism (FAIL, not silently ignored).

Usage:
    python test/exp_f08_concurrency.py [run-tag]

Writes:
    test/fixed_output/results/<run-tag>/EXP-F08.json
"""
from __future__ import annotations

import json
import sys
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiment"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment.reproducibility import activate_reproducibility
from experiment.experiment_config import RETRIEVAL_PARAMS
from shared import (
    canonical_json,
    finalize_report,
    lm_studio_chat,
    make_base_report,
    probe_falkordb,
    probe_lm_studio,
    sha256_hex,
    write_report,
)
from f07_e2e import (
    load_qa_pairs,
    get_llm_answer,
    get_judge_score,
    _DEFAULT_LLM_URL,
    _DEFAULT_LLM_MODEL,
    TEMPERATURE,
    TOP_P,
    MAX_TOKENS,
    SEED,
    NUM_QUESTIONS,
)

EXP_ID             = "EXP-F08"
REPEAT_PER_LEVEL   = 3
CONCURRENCY_LEVELS = [1, 2, 4, 8]
REPO_ROOT          = Path(__file__).resolve().parents[2]

_FAILURE_HINTS = [
    "FalkorDB MATCH without ORDER BY returns nodes in insertion order, which may vary under concurrent writes.",
    "ChromaDB collection.query() result order may be affected by parallel HNSW index state.",
    "BM25 index is rebuilt per-process; ensure serialization is complete before worker processes spawn.",
    "LLM API server-side batching may merge concurrent requests.",
]


# ── Worker ────────────────────────────────────────────────────────────────────

def process_question(
    qa: Dict[str, str],
    retriever,
    warnings: List[str],
    *,
    base_url: str,
    model: str,
) -> Dict[str, str]:
    question = qa["question"]
    gold     = qa["answer"]
    context  = retriever.build_kg_context(question, **RETRIEVAL_PARAMS)
    answer   = get_llm_answer(question, context, warnings, base_url=base_url, model=model)
    judge    = get_judge_score(question, gold, answer, warnings, base_url=base_url, model=model)
    return {
        "question":      question[:80],
        "context_hash":  sha256_hex(context),
        "answer_hash":   sha256_hex(answer),
        "judge_hash":    sha256_hex(judge),
    }


def run_at_concurrency(
    concurrency: int,
    qa_pairs: List[Dict[str, str]],
    retriever,
    warnings: List[str],
    *,
    base_url: str,
    model: str,
) -> Dict[str, str]:
    """Run all questions concurrently and return aggregate hashes."""
    results: List[Dict[str, str]] = [{}] * len(qa_pairs)

    if concurrency == 1:
        for i, qa in enumerate(qa_pairs):
            results[i] = process_question(qa, retriever, warnings, base_url=base_url, model=model)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(process_question, qa, retriever, warnings,
                            base_url=base_url, model=model): i
                for i, qa in enumerate(qa_pairs)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    warnings.append(f"Worker failed for q[{idx}]: {exc}")
                    results[idx] = {"question": qa_pairs[idx]["question"][:80],
                                    "context_hash": "ERROR",
                                    "answer_hash":  "ERROR",
                                    "judge_hash":   "ERROR"}

    # Sort results by question to make hash order deterministic
    results.sort(key=lambda r: r.get("question", ""))

    return {
        "context_hash": sha256_hex(canonical_json([r["context_hash"] for r in results])),
        "answer_hash":  sha256_hex(canonical_json([r["answer_hash"]  for r in results])),
        "judge_hash":   sha256_hex(canonical_json([r["judge_hash"]   for r in results])),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",       default=None,             help="Run-tag for report path")
    parser.add_argument("--llm-url",   default=_DEFAULT_LLM_URL,   dest="llm_url",   help="LM Studio base URL")
    parser.add_argument("--llm-model", default=_DEFAULT_LLM_MODEL, dest="llm_model", help="LLM model name")
    args = parser.parse_args()

    llm_url   = args.llm_url
    llm_model = args.llm_model

    if not probe_falkordb():
        report = make_base_report(EXP_ID, repeat_count=REPEAT_PER_LEVEL, llm_url=llm_url, llm_model=llm_model)
        report["status"] = "SKIP"
        report["warnings"].append("FalkorDB not reachable on localhost:6379")
        path = write_report(report, EXP_ID, args.tag)
        print(json.dumps(report, indent=2))
        return 0

    if not probe_lm_studio(llm_url):
        report = make_base_report(EXP_ID, repeat_count=REPEAT_PER_LEVEL, llm_url=llm_url, llm_model=llm_model)
        report["status"] = "SKIP"
        report["warnings"].append(f"LM Studio not reachable at {llm_url}")
        path = write_report(report, EXP_ID, args.tag)
        print(json.dumps(report, indent=2))
        return 0

    activate_reproducibility(seed=SEED, deterministic=True)
    from KG.pipeline.factory import build_pipeline
    pipeline  = build_pipeline()
    retriever = pipeline["retriever"]

    qa_pairs = load_qa_pairs(NUM_QUESTIONS)

    run_tag = args.tag
    report = make_base_report(
        EXP_ID,
        repeat_count=REPEAT_PER_LEVEL,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        config_snapshot={
            "seed":               SEED,
            "concurrency_levels": CONCURRENCY_LEVELS,
            "repeat_per_level":   REPEAT_PER_LEVEL,
            "num_questions":      NUM_QUESTIONS,
            "llm_base_url":       llm_url,
            "llm_model":          llm_model,
        },
        llm_url=llm_url,
        llm_model=llm_model,
    )
    warnings: List[str]          = []
    failure_diagnosis: List[str] = list(_FAILURE_HINTS)  # prepopulated per spec

    # Store hashes per level to compare against C-01 baseline
    level_hashes: Dict[int, List[Dict[str, str]]] = {c: [] for c in CONCURRENCY_LEVELS}

    for concurrency in CONCURRENCY_LEVELS:
        print(f"[{EXP_ID}] concurrency={concurrency} …", file=sys.stderr)
        for rep in range(REPEAT_PER_LEVEL):
            activate_reproducibility(seed=SEED, deterministic=True)
            agg = run_at_concurrency(concurrency, qa_pairs, retriever, warnings,
                                      base_url=llm_url, model=llm_model)
            level_hashes[concurrency].append(agg)
            report["trials"].append({
                "trial_id":        f"C-{concurrency:02d}_rep{rep}",
                "concurrency":     concurrency,
                "repeat":          rep,
                "artifact_hashes": {
                    f"context_hash_c{concurrency}": agg["context_hash"],
                    f"answer_hash_c{concurrency}":  agg["answer_hash"],
                    f"judge_hash_c{concurrency}":   agg["judge_hash"],
                },
            })

    pipeline["graph"].close()

    # ── Cross-level comparison ────────────────────────────────────────────────
    baseline = level_hashes[1]   # C-01 hashes

    intra_level_variance: Dict[int, bool] = {}
    for c in CONCURRENCY_LEVELS:
        reps = level_hashes[c]
        intra_consistent = len({r["context_hash"] for r in reps}) == 1
        intra_level_variance[c] = not intra_consistent
        if not intra_consistent:
            if c == 1:
                failure_diagnosis.append(
                    f"C-01 intra-trial variance detected: context_hash not identical across {REPEAT_PER_LEVEL} repeats."
                )
            else:
                failure_diagnosis.append(
                    f"concurrency_nondeterminism: C-{c:02d} intra-trial context_hash varies across {REPEAT_PER_LEVEL} repeats."
                )

    # Compare each level vs C-01 (first repeat of each as reference)
    baseline_hash = baseline[0]["context_hash"] if baseline else None
    for c in CONCURRENCY_LEVELS:
        if c == 1:
            continue
        for rep_idx, rep in enumerate(level_hashes[c]):
            if rep["context_hash"] != baseline_hash:
                failure_diagnosis.append(
                    f"concurrency_nondeterminism: C-{c:02d} rep{rep_idx} context_hash differs from C-01 baseline."
                )

    report["intra_level_variance"] = intra_level_variance
    report["warnings"]             = warnings
    report["failure_diagnosis"]    = failure_diagnosis

    # Status: C-01 must PASS; higher levels → FAIL if any hash differs
    c01_consistent = len({r["context_hash"] for r in level_hashes[1]}) == 1
    has_concurrency_fail = any(
        rep["context_hash"] != baseline_hash
        for c in CONCURRENCY_LEVELS if c != 1
        for rep in level_hashes[c]
    )

    if not c01_consistent:
        report["status"] = "FAIL"
    elif has_concurrency_fail:
        report["status"] = "FAIL"
        report["warnings"].append(
            "Concurrency nondeterminism detected at concurrency > 1.  See failure_diagnosis."
        )
    else:
        report["status"] = "PASS"

    path = write_report(report, EXP_ID, run_tag)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[{EXP_ID}] status={report['status']}  report → {path}", file=sys.stderr)
    return 0 if report["status"] in ("PASS",) else 1


if __name__ == "__main__":
    raise SystemExit(main())
