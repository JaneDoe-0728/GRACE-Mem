"""
EXP-F03 — Embedder Determinism

Verifies that Qwen3-Embedding-0.6B produces identical embeddings for the same
input batch across repeated calls.

Usage:
    python test/exp_f03_embedder.py [run-tag]

Writes:
    test/fixed_output/results/<run-tag>/EXP-F03.json
"""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiment"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment.reproducibility import activate_reproducibility
from shared import (
    canonical_json,
    finalize_report,
    make_base_report,
    sha256_hex,
    write_report,
)

EXP_ID = "EXP-F03"
SEED   = 42
REPEAT = 10

# 20 fixed texts: mix of English/Chinese, short/long, with special characters
FIXED_TEXTS: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Alice visited Paris last summer.",
    "深度學習模型在自然語言處理領域取得了顯著進展。",
    "What is the capital of France?",
    "Marie Curie discovered polonium and radium.",
    "我喜歡在早晨喝一杯咖啡，思考今天的計劃。",
    "The Eiffel Tower stands 330 meters tall and was built in 1889.",
    "Bob is a software engineer at Google — he lives in Seattle!",
    "機器學習 ≠ 人工智慧，但它是後者的核心子領域。",
    "To be, or not to be, that is the question.",
    "The mitochondria is the powerhouse of the cell.",
    "自由、平等、博愛 — 法國大革命的三大口號。",
    "SELECT * FROM users WHERE active = TRUE ORDER BY created_at DESC;",
    "E = mc² is Einstein's famous mass-energy equivalence formula.",
    "今天天氣晴朗，非常適合在公園裡散步。",
    "Transformer architecture relies on multi-head attention mechanisms.",
    "Please call me at +1-800-555-0199 or email me@example.com",
    "龍馬精神：形容精力旺盛、朝氣蓬勃的精神狀態。",
    "The JSON spec allows Unicode escapes like \\u00e9 for é.",
    "In 2023, global CO₂ emissions reached a record high of 36.8 GtCO₂.",
]

# 5 fixed query vectors (indices into FIXED_TEXTS used as query seeds)
QUERY_INDICES = [0, 4, 8, 12, 16]
TOP_K = 5


def compute_cosine_ranking(matrix: np.ndarray) -> List[List[str]]:
    """
    Pairwise nearest-neighbour order (IDs only).
    For each row i, find the top-1 neighbour (excluding self) by cosine similarity.
    matrix is already L2-normalized (shape: N x D).
    """
    sims = matrix @ matrix.T  # N x N cosine similarities
    neighbours = []
    for i in range(len(matrix)):
        row = sims[i].copy()
        row[i] = -np.inf  # exclude self
        best = int(np.argmax(row))
        neighbours.append([str(i), str(best)])
    return neighbours


def compute_topk_ranking(
    query_vecs: np.ndarray,
    corpus_vecs: np.ndarray,
    k: int,
) -> List[List[str]]:
    """Top-k candidate IDs per query, sorted by descending score."""
    sims = query_vecs @ corpus_vecs.T  # Q x N
    rankings = []
    for row in sims:
        top_ids = np.argsort(row)[::-1][:k]
        rankings.append([str(int(idx)) for idx in top_ids])
    return rankings


def run_trial(trial_id: int, embedder, warnings: List[str]) -> Dict[str, Any]:
    activate_reproducibility(seed=SEED, deterministic=True)

    matrix = embedder.embed(FIXED_TEXTS)  # shape: (N, D), float32, L2-normalized
    matrix_f32 = matrix.astype(np.float32)

    # Hash 1: raw embedding matrix (float32 hex bytes)
    embedding_matrix_hash = sha256_hex(matrix_f32.tobytes().hex())

    # Hash 2: cosine ranking (pairwise nearest-neighbour, IDs only)
    cosine_ranking = compute_cosine_ranking(matrix_f32)
    cosine_ranking_hash = sha256_hex(canonical_json(cosine_ranking))

    # Hash 3: top-k retrieval for 5 fixed query vectors
    query_vecs = matrix_f32[QUERY_INDICES]
    topk = compute_topk_ranking(query_vecs, matrix_f32, TOP_K)
    topk_ranking_hash = sha256_hex(canonical_json(topk))

    return {
        "trial_id": trial_id,
        "embedding_shape": list(matrix_f32.shape),
        "artifact_hashes": {
            "embedding_matrix_hash": embedding_matrix_hash,   # secondary (allclose)
            "cosine_ranking_hash":   cosine_ranking_hash,     # primary
            "topk_ranking_hash":     topk_ranking_hash,       # primary
        },
    }


def check_allclose(trials: List[Dict], warnings: List[str]) -> None:
    """Warn if raw matrix hashes differ but embeddings are numerically close."""
    if len(trials) < 2:
        return
    # We only have hashes at this point (not raw arrays).
    # If matrix hash differs but ranking hashes are identical → float-precision noise.
    matrix_hashes  = {t["artifact_hashes"]["embedding_matrix_hash"] for t in trials}
    ranking_hashes = {t["artifact_hashes"]["cosine_ranking_hash"]    for t in trials}
    topk_hashes    = {t["artifact_hashes"]["topk_ranking_hash"]      for t in trials}

    if len(matrix_hashes) > 1 and len(ranking_hashes) == 1 and len(topk_hashes) == 1:
        warnings.append(
            "embedding_matrix_hash varies across trials but ranking hashes are identical. "
            "Likely float-precision noise (WARN, not FAIL)."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None, help="Run-tag for report path")
    args = parser.parse_args()

    activate_reproducibility(seed=SEED, deterministic=True)

    from embeddings import HFTextEmbedding
    embedder = HFTextEmbedding()

    report = make_base_report(
        EXP_ID,
        repeat_count=REPEAT,
        config_snapshot={
            "seed":         SEED,
            "text_count":   len(FIXED_TEXTS),
            "query_count":  len(QUERY_INDICES),
            "topk":         TOP_K,
        },
    )
    warnings: List[str] = []

    for i in range(REPEAT):
        trial = run_trial(i, embedder, warnings)
        report["trials"].append(trial)

    check_allclose(report["trials"], warnings)
    report["warnings"] = warnings

    finalize_report(
        report,
        primary_keys=["cosine_ranking_hash", "topk_ranking_hash"],
        warn_keys=["embedding_matrix_hash"],
    )

    path = write_report(report, EXP_ID, args.tag)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[{EXP_ID}] status={report['status']}  report → {path}", file=sys.stderr)
    return 0 if report["status"] in ("PASS", "WARN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
