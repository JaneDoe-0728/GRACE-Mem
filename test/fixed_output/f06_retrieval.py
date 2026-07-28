"""
EXP-F06 — Retrieval Determinism via LoCoMo QA Entry

Verifies that retrieval on a fixed, pre-built graph remains deterministic when
driven from the real LoCoMo QA-stage question loader instead of calling the
retriever directly with ad hoc strings.

Protocol:
  1. Load one LoCoMo sample and its QA items via the LoCoMo QA helpers.
  2. Run retrieval-only for the first N questions on the live graph.
  3. Record per-question retrieval traces:
     - low/high keywords
     - reranker scores
     - selected evidence
  4. Hash those artifacts across repeated trials.

Notes:
  - This audit matches the current LoCoMo retrieval path on question loading.
  - The current LoCoMo QA stage does not rewrite the question text before
    retrieval, so `query_time` is recorded for debugging only and is not passed
    into `build_kg_context()`.

Writes:
    test/fixed_output/results/<run-tag>/EXP-F06.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiment"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment.experiment_config import RETRIEVAL_PARAMS, RERANKER_PARAMS, REPRODUCIBILITY_PARAMS
from experiment.reproducibility import activate_reproducibility
from shared import (
    canonical_json,
    finalize_report,
    make_base_report,
    probe_falkordb,
    run_output_dir,
    sha256_hex,
    write_report,
)

EXP_ID = "EXP-F06"
SEED = 42
REPEAT = 10
NUM_QUESTIONS = 10
SAMPLE_INDEX = 0
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCOMO_JSON = REPO_ROOT / "experiment" / "locomo" / "data" / "locomo10.json"


# ── Tie-break audit ───────────────────────────────────────────────────────────

_AUDIT_PATTERNS = [
    ("KG/graph/falkordb.py", r"MATCH", "ORDER BY"),
    ("KG/storage/chroma_vdb.py", r"\.query\(", "sort"),
    ("KG/pipeline/retrieval_steps/", r"sorted\(", "key="),
]


def run_tiebreak_audit(warnings: List[str]) -> None:
    """Grep for MATCH/query calls without ORDER BY / explicit sort; log findings."""
    import subprocess

    for rel_path, pattern, required_token in _AUDIT_PATTERNS:
        target = REPO_ROOT / rel_path
        if not target.exists():
            continue
        result = subprocess.run(
            ["grep", "-rn", pattern, str(target)],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if required_token.lower() not in line.lower():
                warnings.append(f"[tie-break audit] possible missing deterministic sort: {line.strip()}")


# ── LoCoMo QA loading ─────────────────────────────────────────────────────────

def _load_locomo_qa_items(sample_index: int, limit: int) -> List[Dict[str, Any]]:
    from experiment.locomo.helpers.dataset import get_sample_conversation, load_qa_items, load_raw_samples

    samples = load_raw_samples(LOCOMO_JSON)
    sample = samples[sample_index]
    conv = get_sample_conversation(sample)
    query_time = _resolve_locomo_query_time(conv)

    qa_items = load_qa_items(
        LOCOMO_JSON,
        sample_index=sample_index,
        include_adversarial=False,
    )

    rows: List[Dict[str, Any]] = []
    for item in qa_items:
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        rows.append(
            {
                "question": question,
                "query_time": query_time,
                "category": str(item.get("category_label") or item.get("category") or ""),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _resolve_locomo_query_time(conversation: Dict[str, Any]) -> str | None:
    from KG.utils.query_time_parser import parse_query_time

    best_text: str | None = None
    best_dt = None
    for key, value in conversation.items():
        if not key.startswith("session_") or not key.endswith("_date_time"):
            continue
        text = str(value or "").strip()
        if not text:
            continue
        dt = parse_query_time(text)
        if dt is None:
            continue
        if best_dt is None or dt > best_dt:
            best_dt = dt
            best_text = text
    return best_text


# ── Trace normalization ───────────────────────────────────────────────────────

def _canonical_ids(trace: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        "entity_ids": [str(x) for x in trace.get("final_entity_ids", [])],
        "relationship_ids": [str(x) for x in trace.get("final_relationship_ids", [])],
    }


def _canonical_keywords(trace: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        "low_level_keywords": [str(x) for x in trace.get("low_level_keywords", [])],
        "high_level_keywords": [str(x) for x in trace.get("high_level_keywords", [])],
    }


def _canonical_selected_evidence(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence = trace.get("selected_evidence") or []
    rows: List[Dict[str, Any]] = []
    for item in evidence:
        rows.append(
            {
                "rank": int(item.get("rank", 0)),
                "score": float(item.get("score", 0.0)),
                "summary_id": str(item.get("summary_id") or ""),
                "dialogue_datetime": str(item.get("dialogue_datetime") or ""),
                "preview": str(item.get("preview") or ""),
            }
        )
    return rows


def _canonical_reranker_rows(log_dir: Path, request_id: str) -> List[Dict[str, Any]]:
    from KG.utils.error_analysis import read_reranker_rows

    rows = read_reranker_rows(log_dir, request_id=request_id)
    out: List[Dict[str, Any]] = []
    for row in rows:
        try:
            rank = int(row.get("rank", 0))
        except (TypeError, ValueError):
            rank = 0
        try:
            score = float(row.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        out.append(
            {
                "item_type": str(row.get("item_type") or ""),
                "item_id": str(row.get("item_id") or ""),
                "name": str(row.get("name") or ""),
                "score": score,
                "rank": rank,
                "above_threshold": str(row.get("above_threshold") or ""),
                "selected": str(row.get("selected") or ""),
                "threshold": str(row.get("threshold") or ""),
                "top_k": str(row.get("top_k") or ""),
                "drop_reason": str(row.get("drop_reason") or ""),
            }
        )
    return sorted(out, key=lambda r: (r["item_type"], r["rank"], r["item_id"], r["name"]))


def hash_retrieval_result(
    context_text: str,
    trace: Dict[str, Any],
    reranker_rows: List[Dict[str, Any]],
) -> Dict[str, str]:
    context_text_hash = sha256_hex(context_text)
    context_ids_hash = sha256_hex(canonical_json(_canonical_ids(trace)))
    keyword_trace_hash = sha256_hex(canonical_json(_canonical_keywords(trace)))
    selected_evidence_hash = sha256_hex(canonical_json(_canonical_selected_evidence(trace)))
    reranker_scores_hash = sha256_hex(canonical_json(reranker_rows))
    return {
        "context_text_hash": context_text_hash,
        "context_ids_hash": context_ids_hash,
        "keyword_trace_hash": keyword_trace_hash,
        "selected_evidence_hash": selected_evidence_hash,
        "reranker_scores_hash": reranker_scores_hash,
    }


# ── Trial runner ──────────────────────────────────────────────────────────────

def run_trial(
    trial_id: int,
    retriever,
    qa_items: List[Dict[str, Any]],
    warnings: List[str],
    trace_dir: Path,
) -> Dict[str, Any]:
    activate_reproducibility(seed=SEED, deterministic=True)

    import experiment.locomo.stages.qa_eval as locomo_qa_eval
    import KG.pipeline.retrieval_steps.filtering as filtering_module

    locomo_qa_eval.retriever = retriever

    trace_dir.mkdir(parents=True, exist_ok=True)
    reranker_csv = trace_dir / "reranker_scores.csv"
    if reranker_csv.exists():
        reranker_csv.unlink()

    prev_reranker_csv = filtering_module._RERANKER_SCORE_CSV
    filtering_module._RERANKER_SCORE_CSV = str(reranker_csv)

    per_query: List[Dict[str, Any]] = []
    try:
        for item in qa_items:
            question = item["question"]
            query_time = item.get("query_time")

            context_text = retriever.build_kg_context(question=question, **RETRIEVAL_PARAMS)
            trace = getattr(retriever, "last_retrieval_trace", None) or {}
            request_id = str(trace.get("request_id") or "")
            reranker_rows = _canonical_reranker_rows(trace_dir, request_id) if request_id else []

            if request_id and not reranker_rows:
                warnings.append(f"trial={trial_id} question='{question[:40]}' produced no reranker rows")

            hashes = hash_retrieval_result(context_text, trace, reranker_rows)
            per_query.append(
                {
                    "question": question[:80],
                    "query_time": query_time,
                    "category": item.get("category", ""),
                    "request_id": request_id,
                    "context_length": len(context_text),
                    "low_level_keywords": _canonical_keywords(trace)["low_level_keywords"],
                    "high_level_keywords": _canonical_keywords(trace)["high_level_keywords"],
                    "selected_evidence": _canonical_selected_evidence(trace),
                    "reranker_scores": reranker_rows,
                    **hashes,
                }
            )
    finally:
        filtering_module._RERANKER_SCORE_CSV = prev_reranker_csv

    return {
        "trial_id": trial_id,
        "per_query": per_query,
        "artifact_hashes": {
            "context_text_hash": sha256_hex(canonical_json([row["context_text_hash"] for row in per_query])),
            "context_ids_hash": sha256_hex(canonical_json([row["context_ids_hash"] for row in per_query])),
            "keyword_trace_hash": sha256_hex(canonical_json([row["keyword_trace_hash"] for row in per_query])),
            "selected_evidence_hash": sha256_hex(canonical_json([row["selected_evidence_hash"] for row in per_query])),
            "reranker_scores_hash": sha256_hex(canonical_json([row["reranker_scores_hash"] for row in per_query])),
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None, help="Run-tag for report path")
    parser.add_argument("--sample-index", type=int, default=SAMPLE_INDEX, help="LoCoMo sample index")
    args = parser.parse_args()

    if not probe_falkordb():
        report = make_base_report(EXP_ID, repeat_count=REPEAT)
        report["status"] = "SKIP"
        report["warnings"].append("FalkorDB not reachable on localhost:6379")
        path = write_report(report, EXP_ID, args.tag)
        print(json.dumps(report, indent=2))
        print(f"\n[{EXP_ID}] SKIP  report → {path}", file=sys.stderr)
        return 0

    qa_items = _load_locomo_qa_items(args.sample_index, NUM_QUESTIONS)
    if not qa_items:
        print(f"[{EXP_ID}] ERROR: could not load locomo QA items from {LOCOMO_JSON}", file=sys.stderr)
        return 1

    activate_reproducibility(seed=SEED, deterministic=True)

    from KG.pipeline.factory import build_pipeline

    pipeline = build_pipeline()
    retriever = pipeline["retriever"]

    run_tag = args.tag
    trace_root = run_output_dir(run_tag) / f"{EXP_ID}-traces"
    report = make_base_report(
        EXP_ID,
        repeat_count=REPEAT,
        config_snapshot={
            "seed": SEED,
            "num_questions": len(qa_items),
            "sample_index": args.sample_index,
            "dataset_json": str(LOCOMO_JSON),
            "qa_entry": "experiment.locomo.stages.qa_eval",
            "locomo_query_rewrite": False,
            **RETRIEVAL_PARAMS,
            **RERANKER_PARAMS,
            **REPRODUCIBILITY_PARAMS,
        },
    )
    warnings: List[str] = [
        "LoCoMo QA stage currently sends the original question into retrieval; query_time is recorded for debugging only.",
    ]
    failure_diagnosis: List[str] = []

    print(f"[{EXP_ID}] Running tie-break audit…", file=sys.stderr)
    run_tiebreak_audit(warnings)

    for trial_id in range(REPEAT):
        print(f"[{EXP_ID}] Trial {trial_id + 1}/{REPEAT}…", file=sys.stderr)
        try:
            trial_trace_dir = trace_root / f"trial_{trial_id}"
            trial = run_trial(trial_id, retriever, qa_items, warnings, trial_trace_dir)
            report["trials"].append(trial)
        except Exception as exc:
            failure_diagnosis.append(f"Trial {trial_id} failed: {exc}")
            import traceback

            traceback.print_exc()

    pipeline["graph"].close()

    report["warnings"] = warnings
    report["failure_diagnosis"] = failure_diagnosis

    finalize_report(
        report,
        primary_keys=[
            "context_text_hash",
            "context_ids_hash",
            "keyword_trace_hash",
            "selected_evidence_hash",
            "reranker_scores_hash",
        ],
    )

    path = write_report(report, EXP_ID, run_tag)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[{EXP_ID}] status={report['status']}  report → {path}", file=sys.stderr)
    return 0 if report["status"] in ("PASS", "WARN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
