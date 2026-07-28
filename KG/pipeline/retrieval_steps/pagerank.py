# pipeline/retrieval_steps/pagerank.py
"""
Personalized PageRank re-ranking on the induced subgraph of RRF candidates.
"""
from typing import Dict, List, Tuple

import numpy as np


class SubgraphPageRank:
    """
    Runs Personalized PageRank on the induced subgraph of a candidate entity set.

    The graph is built from the edges already assembled in steps 2.2/2.4 — no new
    graph query is issued. PPR is used purely as a structural re-ranking step after
    an upstream fusion stage (e.g. RRF) has narrowed the candidate set.
    """

    def run_ppr(
        self,
        entity_ids: List[str],
        rrf_scores: Dict[str, float],
        subgraph_edges: List[Tuple[str, str, str]],
        alpha: float,
        top_k: int,
        inverse_degree_weight: bool,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Run Personalized PageRank and return ranked entities.

        Args:
            entity_ids: Candidate entity IDs (ordered by upstream score).
            rrf_scores: entity_id → unnormalized upstream score used as personalization prior.
            subgraph_edges: (src_id, rel_id, tgt_id) triples; rel_id is ignored here.
            alpha: Restart probability (higher = stronger seed bias).
            top_k: Number of entities to return.
            inverse_degree_weight: If True apply symmetric degree normalization for hub suppression.

        Returns:
            (ranked_entity_ids, ppr_score_dict)
        """
        n = len(entity_ids)
        if n == 0:
            return [], {}

        idx = {eid: i for i, eid in enumerate(entity_ids)}
        ent_set = set(entity_ids)

        # Build adjacency matrix (undirected)
        A = np.zeros((n, n), dtype=np.float32)
        for src_id, _rel_id, tgt_id in subgraph_edges:
            i = idx.get(src_id)
            j = idx.get(tgt_id)
            if i is None or j is None or src_id not in ent_set or tgt_id not in ent_set:
                continue
            A[i, j] += 1.0
            A[j, i] += 1.0

        # Normalize adjacency
        if inverse_degree_weight:
            # Symmetric normalization: D^{-1/2} A D^{-1/2}
            deg = A.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_sqrt_deg = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
            A_norm = (A * inv_sqrt_deg[:, None]) * inv_sqrt_deg[None, :]
        else:
            # Row-stochastic: each row sums to 1 (or 0 for isolated nodes)
            row_sums = A.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            A_norm = A / row_sums

        # Personalization vector from upstream scores
        raw = np.array(
            [rrf_scores.get(eid, 0.0) for eid in entity_ids], dtype=np.float32
        )
        total = raw.sum()
        p = raw / total if total > 0 else np.ones(n, dtype=np.float32) / n

        # Iterative PPR: pr = alpha * A_norm^T @ pr + (1 - alpha) * p
        pr = p.copy()
        for _ in range(100):
            pr_new = alpha * A_norm.T @ pr + (1.0 - alpha) * p
            if np.linalg.norm(pr_new - pr, ord=1) < 1e-6:
                pr = pr_new
                break
            pr = pr_new

        # Rank entities by PPR score
        order = np.argsort(pr)[::-1]
        ranked = [entity_ids[i] for i in order[:top_k]]
        ppr_dict = {entity_ids[i]: float(pr[i]) for i in range(n)}

        return ranked, ppr_dict
