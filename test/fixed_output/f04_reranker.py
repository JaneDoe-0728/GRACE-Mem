"""
EXP-F04 — Reranker Determinism

Verifies that Qwen3-Reranker-0.6B produces identical scores and rankings for
the same (query, candidate) pairs across repeated calls.

Runs under torch.use_deterministic_algorithms(True) and float32 precision.
If the model uses float16 internally, a float32-cast sub-trial is included
to isolate precision as the variance source.

Usage:
    python test/exp_f04_reranker.py [run-tag]

Writes:
    test/fixed_output/results/<run-tag>/EXP-F04.json
"""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiment"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment.common.reproducibility import activate_reproducibility
from shared import (
    canonical_json,
    finalize_report,
    make_base_report,
    sha256_hex,
    write_report,
)

EXP_ID = "EXP-F04"
SEED   = 42
REPEAT = 10
TOP_K  = 3

# 5 fixed queries × 10 fixed candidate texts = 50 pairs
QUERIES: List[str] = [
    "Where does Alice work?",
    "What did Marie Curie discover?",
    "Who is Bob's closest colleague?",
    "When was the Eiffel Tower built?",
    "What is the capital of Japan?",
]

CANDIDATES: List[str] = [
    "Alice is a senior software engineer at Anthropic in San Francisco.",
    "Alice moved to New York last year to pursue a career in finance.",
    "Marie Curie discovered polonium and radium in the late 1890s.",
    "The Nobel Prize was awarded to Marie Curie twice.",
    "Bob collaborates closely with Dr. Sara Chen from MIT on NLP research.",
    "Bob prefers to work alone and rarely attends team meetings.",
    "The Eiffel Tower was constructed in 1887–1889 as the entrance arch for the 1889 World's Fair.",
    "The Eiffel Tower is painted every seven years using 60 tonnes of paint.",
    "Tokyo is the capital and most populous city of Japan.",
    "Japan is an island country in East Asia known for its technology and culture.",
]


def scores_to_ranking(scores: List[float]) -> List[str]:
    """Sort candidate indices by descending score, stable tie-break by index."""
    indexed = sorted(enumerate(scores), key=lambda x: (-x[1], x[0]))
    return [str(idx) for idx, _ in indexed]


def scores_to_topk(scores: List[float], k: int) -> List[str]:
    ranking = scores_to_ranking(scores)
    return ranking[:k]


def run_trial(
    trial_id: int,
    reranker,
    warnings: List[str],
    force_float32: bool = False,
) -> Dict[str, Any]:
    activate_reproducibility(seed=SEED, deterministic=True)

    per_query_scores: List[List[float]] = []
    for query in QUERIES:
        # rank_pairs returns List[Tuple[int, float]] sorted by score desc
        ranked = reranker.rank_pairs(query, CANDIDATES)
        # Re-index to original candidate order
        score_map = dict(ranked)
        scores = [float(score_map.get(i, -999.0)) for i in range(len(CANDIDATES))]
        if force_float32:
            scores = [float(np.float32(s)) for s in scores]
        per_query_scores.append(scores)

    # Hash 1: raw score vector (float32 bytes)
    score_arr = np.array(per_query_scores, dtype=np.float32)
    raw_score_hash = sha256_hex(score_arr.tobytes().hex())

    # Hash 2: ranking per query (candidate IDs sorted by descending score)
    rankings = [scores_to_ranking(sq) for sq in per_query_scores]
    ranking_hash = sha256_hex(canonical_json(rankings))

    # Hash 3: top-k per query
    topk = [scores_to_topk(sq, TOP_K) for sq in per_query_scores]
    topk_hash = sha256_hex(canonical_json(topk))

    return {
        "trial_id":          trial_id,
        "force_float32":     force_float32,
        "artifact_hashes": {
            "raw_score_hash": raw_score_hash,  # secondary (allclose)
            "ranking_hash":   ranking_hash,    # primary
            "topk_hash":      topk_hash,       # primary
        },
    }


def run_allclose_check(
    trials: List[Dict],
    reranker,
    warnings: List[str],
) -> None:
    """
    If raw score hashes differ, re-run with forced float32 cast to isolate
    precision as the variance source, and append a sub-trial.
    """
    score_hashes = {t["artifact_hashes"]["raw_score_hash"] for t in trials}
    if len(score_hashes) <= 1:
        return
    warnings.append(
        "raw_score_hash varies across trials.  "
        "Running float32-cast sub-trial to isolate precision variance."
    )
    trial = run_trial(len(trials), reranker, warnings, force_float32=True)
    trial["trial_id"] = f"float32_cast_check_{trial['trial_id']}"
    trials.append(trial)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None, help="Run-tag for report path")
    args = parser.parse_args()

    activate_reproducibility(seed=SEED, deterministic=True)

    from KG.utils.reranker import get_reranker
    reranker = get_reranker()

    report = make_base_report(
        EXP_ID,
        repeat_count=REPEAT,
        config_snapshot={
            "seed":          SEED,
            "query_count":   len(QUERIES),
            "candidate_count": len(CANDIDATES),
            "topk":          TOP_K,
            "deterministic": True,
        },
    )
    warnings: List[str] = []

    for i in range(REPEAT):
        trial = run_trial(i, reranker, warnings)
        report["trials"].append(trial)

    run_allclose_check(report["trials"], reranker, warnings)
    report["warnings"] = warnings

    finalize_report(
        report,
        primary_keys=["ranking_hash", "topk_hash"],
        warn_keys=["raw_score_hash"],
    )

    path = write_report(report, EXP_ID, args.tag)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[{EXP_ID}] status={report['status']}  report → {path}", file=sys.stderr)
    return 0 if report["status"] in ("PASS", "WARN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
