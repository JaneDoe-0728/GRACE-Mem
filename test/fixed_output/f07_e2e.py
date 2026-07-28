"""
EXP-F07 — End-to-End Deterministic Validation

Validates that the full pipeline — fresh ingest → retrieval → LLM answer →
judge score — produces identical artifacts across independent trials.

Config:
    seed=42, deterministic=True, temperature=0.0, concurrency=1

Protocol per trial:
  1. Clear all state (refresh_system.py)
  2. Ingest first 3 sessions of locomo10.json
  3. Run QA evaluation over corresponding questions
  4. Export canonical artifacts
  Repeat ≥ 3 times.

Blocking dependencies: EXP-F02-a PASS, EXP-F05 PASS, EXP-F06 PASS.

Usage:
    python test/exp_f07_e2e.py [run-tag]

Writes:
    test/fixed_output/results/<run-tag>/EXP-F07.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiment"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment.reproducibility import activate_reproducibility
from experiment.experiment_config import (
    INGEST_PARAMS, RETRIEVAL_PARAMS, RERANKER_PARAMS, REPRODUCIBILITY_PARAMS,
)
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
from f05_ingest import (
    load_first_n_sessions,
    build_ingest_df,
    clear_state,
    export_canonical_artifacts,
    hash_artifacts,
)

EXP_ID        = "EXP-F07"
SEED          = 42
REPEAT        = 3
NUM_SESSIONS  = 3
NUM_QUESTIONS = 10
TEMPERATURE   = 0.0
TOP_P         = 1.0
MAX_TOKENS    = 512
REPO_ROOT     = Path(__file__).resolve().parents[2]
LOCOMO_JSON   = REPO_ROOT / "experiment" / "locomo" / "data" / "locomo10.json"

_DEFAULT_LLM_URL   = "http://localhost:1234/v1"
_DEFAULT_LLM_MODEL = "gpt-oss-20b"


# ── QA helpers ────────────────────────────────────────────────────────────────

def _normalize_qa_text(value: Any) -> str:
    """Normalize QA fields from locomo JSON to a trimmed string."""
    if value is None:
        return ""
    return str(value).strip()


def load_qa_pairs(n: int) -> List[Dict[str, str]]:
    data = json.loads(LOCOMO_JSON.read_text(encoding="utf-8"))
    pairs: List[Dict[str, str]] = []
    seen: set[str] = set()
    for sample in data:
        for qa in sample.get("qa", []):
            q = _normalize_qa_text(qa.get("question", ""))
            a = _normalize_qa_text(qa.get("answer", ""))
            if q and q not in seen:
                seen.add(q)
                pairs.append({"question": q, "answer": a})
            if len(pairs) >= n:
                return pairs
    return pairs


def get_llm_answer(
    question: str, context: str, warnings: List[str],
    *, base_url: str, model: str,
) -> str:
    try:
        messages = [
            {"role": "system", "content": "Answer the question based only on the provided context. Be concise."},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        return lm_studio_chat(
            messages,
            base_url=base_url,
            model=model,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            seed=SEED,
        )
    except Exception as exc:
        warnings.append(f"LLM answer failed for '{question[:40]}': {exc}")
        return "ERROR"


def get_judge_score(
    question: str, gold: str, pred: str, warnings: List[str],
    *, base_url: str, model: str,
) -> str:
    try:
        messages = [
            {"role": "system", "content":
                "You are a strict evaluation judge. Given a question, a gold answer, and a predicted answer, "
                "output ONLY a JSON object: {\"score\": 0 or 1, \"rationale\": \"<one sentence>\"}. "
                "Score 1 if the prediction is factually correct and complete, else 0."},
            {"role": "user", "content":
                f"Question: {question}\nGold: {gold}\nPrediction: {pred}"},
        ]
        return lm_studio_chat(
            messages,
            base_url=base_url,
            model=model,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            seed=SEED,
        )
    except Exception as exc:
        warnings.append(f"Judge failed for '{question[:40]}': {exc}")
        return "ERROR"


# ── Trial runner ──────────────────────────────────────────────────────────────

def run_trial(
    trial_id: int,
    session_records: List[Dict],
    qa_pairs: List[Dict[str, str]],
    warnings: List[str],
    failure_diagnosis: List[str],
    *,
    base_url: str,
    model: str,
) -> Dict[str, Any]:
    activate_reproducibility(seed=SEED, deterministic=True)

    from KG.pipeline.factory import build_pipeline
    from experiment.locomo.stages.ingest import ingest_by_session_one_turn

    pipeline  = build_pipeline()
    ingestor  = pipeline["ingestor"]
    retriever = pipeline["retriever"]

    # Ingest
    df = build_ingest_df(session_records)
    ingest_by_session_one_turn(
        ingestor, df,
        prev_k=INGEST_PARAMS.get("prev_k", 2),
        entity_sim_topk=INGEST_PARAMS.get("entity_sim_topk", 3),
        entity_sim_threshold=INGEST_PARAMS.get("entity_sim_threshold", 0.6),
    )

    # Graph hash (reuse F05 logic)
    artifacts     = export_canonical_artifacts(pipeline)
    graph_hash    = sha256_hex(canonical_json(artifacts["graph_export"]))

    # Retrieve + answer + judge per question
    per_q_results: List[Dict] = []
    score_sum = 0
    for qa in qa_pairs:
        question = qa["question"]
        gold     = qa["answer"]

        context        = retriever.build_kg_context(question, **RETRIEVAL_PARAMS)
        answer         = get_llm_answer(question, context, warnings, base_url=base_url, model=model)
        judge_output   = get_judge_score(question, gold, answer, warnings, base_url=base_url, model=model)

        # Try to parse judge JSON for aggregate metrics
        try:
            jdata = json.loads(judge_output)
            score_sum += int(jdata.get("score", 0))
        except Exception:
            pass

        per_q_results.append({
            "question":          question[:80],
            "retrieval_hash":    sha256_hex(context),
            "answer_hash":       sha256_hex(answer),
            "judge_hash":        sha256_hex(judge_output),
        })

    aggregate_metrics = {
        "correct": score_sum,
        "total":   len(qa_pairs),
        "accuracy": round(score_sum / max(len(qa_pairs), 1), 4),
    }

    # Aggregate hashes across questions
    agg_retrieval = sha256_hex(canonical_json([r["retrieval_hash"] for r in per_q_results]))
    agg_answer    = sha256_hex(canonical_json([r["answer_hash"]    for r in per_q_results]))
    agg_judge     = sha256_hex(canonical_json([r["judge_hash"]     for r in per_q_results]))
    agg_metrics   = sha256_hex(canonical_json(aggregate_metrics))

    pipeline["graph"].close()

    return {
        "trial_id":        trial_id,
        "per_question":    per_q_results,
        "aggregate_metrics": aggregate_metrics,
        "artifact_hashes": {
            "graph_hash":             graph_hash,       # primary
            "retrieval_context_hash": agg_retrieval,   # primary
            "final_answer_hash":      agg_answer,      # primary
            "judge_output_hash":      agg_judge,       # primary
            "aggregate_metric_hash":  agg_metrics,     # primary
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",       default=None,                      help="Run-tag for report path")
    parser.add_argument("--llm-url",   default=_DEFAULT_LLM_URL,           dest="llm_url",   help="LM Studio base URL")
    parser.add_argument("--llm-model", default=_DEFAULT_LLM_MODEL,         dest="llm_model", help="LLM model name")
    args = parser.parse_args()

    llm_url   = args.llm_url
    llm_model = args.llm_model

    # Ingest pipeline uses llm_post() which reads env vars
    import os
    os.environ["LLM_API"]    = llm_url
    os.environ["MODEL_NAME"] = llm_model

    if not probe_falkordb():
        report = make_base_report(EXP_ID, repeat_count=REPEAT, llm_url=llm_url, llm_model=llm_model)
        report["status"] = "SKIP"
        report["warnings"].append("FalkorDB not reachable on localhost:6379")
        path = write_report(report, EXP_ID, args.tag)
        print(json.dumps(report, indent=2))
        return 0

    if not probe_lm_studio(llm_url):
        report = make_base_report(EXP_ID, repeat_count=REPEAT, llm_url=llm_url, llm_model=llm_model)
        report["status"] = "SKIP"
        report["warnings"].append(f"LM Studio not reachable at {llm_url}")
        path = write_report(report, EXP_ID, args.tag)
        print(json.dumps(report, indent=2))
        return 0

    session_records = load_first_n_sessions(NUM_SESSIONS)
    qa_pairs        = load_qa_pairs(NUM_QUESTIONS)

    report = make_base_report(
        EXP_ID,
        repeat_count=REPEAT,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        config_snapshot={
            "seed":           SEED,
            "num_sessions":   NUM_SESSIONS,
            "num_questions":  NUM_QUESTIONS,
            "llm_base_url":   llm_url,
            "llm_model":      llm_model,
            **INGEST_PARAMS,
            **RETRIEVAL_PARAMS,
            **REPRODUCIBILITY_PARAMS,
        },
        llm_url=llm_url,
        llm_model=llm_model,
    )
    warnings: List[str]          = []
    failure_diagnosis: List[str] = []

    for trial_id in range(REPEAT):
        print(f"[{EXP_ID}] Trial {trial_id + 1}/{REPEAT}: clearing state…", file=sys.stderr)
        if not clear_state(warnings):
            failure_diagnosis.append(f"Trial {trial_id}: refresh_system.py failed")
            continue

        print(f"[{EXP_ID}] Trial {trial_id + 1}/{REPEAT}: running pipeline…", file=sys.stderr)
        try:
            trial = run_trial(
                trial_id, session_records, qa_pairs, warnings, failure_diagnosis,
                base_url=llm_url, model=llm_model,
            )
            report["trials"].append(trial)
        except Exception as exc:
            failure_diagnosis.append(f"Trial {trial_id} pipeline error: {exc}")
            import traceback; traceback.print_exc()

    report["warnings"]          = warnings
    report["failure_diagnosis"] = failure_diagnosis

    finalize_report(
        report,
        primary_keys=[
            "graph_hash", "retrieval_context_hash",
            "final_answer_hash", "judge_output_hash", "aggregate_metric_hash",
        ],
    )

    path = write_report(report, EXP_ID, args.tag)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[{EXP_ID}] status={report['status']}  report → {path}", file=sys.stderr)
    return 0 if report["status"] in ("PASS", "WARN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
