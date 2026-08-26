"""
Graph-linked summary scoring for ablation-friendly summary selection.

Scoring formula:
  score(summary) =
      relation_weight  * matched_surviving_relation_count
    + entity_weight    * matched_surviving_entity_count
    + pair_bonus_weight * entity_relation_pair_coverage_bonus   (if enable_pair_bonus)
    + semantic_weight  * summary_query_similarity
    - popularity_penalty_weight * popularity_penalty            (if enable_popularity_penalty)
    - redundancy_penalty_weight * redundancy_penalty            (if enable_redundancy_penalty)

"Surviving" means: entities/relations that passed the upstream retrieval and
filtering/reranking stages.  The graph index is built exclusively from those
items, so KG-link counts are always query-conditioned.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from grace_mem.services import Provenance
from grace_mem.storage import build_id_to_meta_maps

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScoringWeights:
    """
    All knobs for the graph-linked summary scorer (weighted-sum and RRF modes).
    Safe defaults reproduce the documented recommended weights while leaving
    popularity and redundancy penalties off (must be explicitly enabled).
    """
    relation_weight: float = 2.0
    entity_weight: float = 1.0
    pair_bonus_weight: float = 1.5
    semantic_weight: float = 0.5
    popularity_penalty_weight: float = 1.0
    redundancy_penalty_weight: float = 1.0
    enable_pair_bonus: bool = True
    enable_popularity_penalty: bool = False
    enable_redundancy_penalty: bool = False
    # RRF-specific
    rrf_k: float = 60.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScoringWeights":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# Per-summary scoring result (used for logging and ranking)
# ---------------------------------------------------------------------------

@dataclass
class SummaryScore:
    """Scoring breakdown for one summary candidate."""
    summary_id: str
    matched_entity_count: int
    matched_relation_count: int
    pair_bonus: float
    semantic_score: float
    popularity_penalty: float
    redundancy_penalty: float
    base_score: float    # score before redundancy adjustment
    final_score: float   # base_score - redundancy_penalty_weight * redundancy_penalty
    matched_entity_ids: List[str] = field(default_factory=list)
    matched_entity_names: List[str] = field(default_factory=list)
    matched_relation_ids: List[str] = field(default_factory=list)
    matched_relation_names: List[str] = field(default_factory=list)

    def to_log_dict(self, weights: ScoringWeights, mode: str) -> Dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "final_score": round(self.final_score, 6),
            "base_score": round(self.base_score, 6),
            "matched_entity_count": self.matched_entity_count,
            "matched_relation_count": self.matched_relation_count,
            "pair_bonus": round(self.pair_bonus, 4),
            "semantic_similarity": round(self.semantic_score, 6),
            "popularity_penalty": round(self.popularity_penalty, 6),
            "redundancy_penalty": round(self.redundancy_penalty, 6),
            "matched_entity_ids": self.matched_entity_ids,
            "matched_entity_names": self.matched_entity_names,
            "matched_relation_ids": self.matched_relation_ids,
            "matched_relation_names": self.matched_relation_names,
            "summary_filter_mode": mode,
            "weights": {
                "relation_weight": weights.relation_weight,
                "entity_weight": weights.entity_weight,
                "pair_bonus_weight": weights.pair_bonus_weight,
                "semantic_weight": weights.semantic_weight,
                "popularity_penalty_weight": weights.popularity_penalty_weight,
                "redundancy_penalty_weight": weights.redundancy_penalty_weight,
                "enable_pair_bonus": weights.enable_pair_bonus,
                "enable_popularity_penalty": weights.enable_popularity_penalty,
                "enable_redundancy_penalty": weights.enable_redundancy_penalty,
            },
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_summary_graph_index(
    context_entities: List[Dict],
    context_relationships: List[Dict],
    cache: Dict,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Walk provenance of *surviving* entities/relations to build a reverse index:
        summary_id -> set of entity_ids that reference it
        summary_id -> set of relation_ids that reference it

    Only items in context_entities / context_relationships are considered.
    This ensures KG-link counts are conditioned on the retrieval output.
    """
    entity_id2meta, rel_id2meta = build_id_to_meta_maps(cache)
    summary_to_entities: Dict[str, Set[str]] = {}
    summary_to_rels: Dict[str, Set[str]] = {}

    for ent in (context_entities or []):
        eid = ent.get("id")
        if not eid:
            continue
        meta = entity_id2meta.get(eid, {}) or {}
        for ev in Provenance.prov_to_events(meta.get("prov") or {}):
            sid = ev.get("summary_id")
            if sid:
                summary_to_entities.setdefault(sid, set()).add(eid)

    for rel in (context_relationships or []):
        rid = rel.get("rel_id")
        if not rid:
            continue
        meta = rel_id2meta.get(rid, {}) or {}
        for ev in Provenance.prov_to_events(meta.get("prov") or {}):
            sid = ev.get("summary_id")
            if sid:
                summary_to_rels.setdefault(sid, set()).add(rid)

    return summary_to_entities, summary_to_rels


def _compute_pair_bonus(
    entity_ids: Set[str],
    rel_ids: Set[str],
    rel_id2meta: Dict,
) -> float:
    """
    Count (entity, relation) pairs where the entity is an endpoint of the relation
    and both are in the surviving set that references this summary.
    Each qualifying (entity, endpoint-role) pair adds 1.0 to the bonus.
    """
    if not entity_ids or not rel_ids:
        return 0.0
    bonus = 0.0
    for rid in rel_ids:
        meta = rel_id2meta.get(rid, {}) or {}
        src_id = meta.get("source_id")
        tgt_id = meta.get("target_id")
        if src_id and src_id in entity_ids:
            bonus += 1.0
        if tgt_id and tgt_id in entity_ids:
            bonus += 1.0
    return bonus


def _popularity_penalty(ref_count: int, pool_size: int) -> float:
    """
    Fractional penalty: proportion of the surviving pool that references this summary.
    A summary referenced by all surviving items gets penalty 1.0.
    A summary referenced by exactly 1 item gets penalty 1/pool_size.
    Returns 0 when pool is empty.
    """
    if pool_size <= 0:
        return 0.0
    return min(1.0, ref_count / pool_size)


def _rel_display_label(rid: str, entity_id2meta: Dict, rel_id2meta: Dict) -> str:
    """Render a relationship as "source -> target" for logs and evidence text.

    Stable formatting matters here: the label is parsed back out when replaying
    a prior run's relationships, so a change to it breaks replay against
    existing artifacts.
    """
    meta = rel_id2meta.get(rid, {}) or {}
    if not meta:
        return rid
    src = (
        meta.get("source_entity")
        or (entity_id2meta.get(meta.get("source_id"), {}) or {}).get("name")
        or meta.get("source_id")
        or "?"
    )
    tgt = (
        meta.get("target_entity")
        or (entity_id2meta.get(meta.get("target_id"), {}) or {}).get("name")
        or meta.get("target_id")
        or "?"
    )
    desc = (meta.get("description") or "").strip()
    return f"{src} -> {tgt}" if not desc else f"{src} -> {tgt} | {desc}"


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def score_summary_graph_linked(
    summary_id: str,
    semantic_score: float,
    summary_to_entities: Dict[str, Set[str]],
    summary_to_rels: Dict[str, Set[str]],
    entity_id2meta: Dict,
    rel_id2meta: Dict,
    pool_size: int,
    weights: ScoringWeights,
    max_entity_count: int = 0,
    max_relation_count: int = 0,
    max_pair_bonus: float = 0.0,
) -> SummaryScore:
    """
    Compute graph-linked score for one summary (redundancy penalty = 0 here;
    it is applied later during the greedy ranking pass).

    Args:
        summary_id:          ChromaDB document ID for the summary.
        semantic_score:      Pre-computed cosine similarity to the query vector.
        summary_to_entities: Reverse index: summary_id -> set of entity_ids.
        summary_to_rels:     Reverse index: summary_id -> set of rel_ids.
        entity_id2meta:      entity_id -> metadata dict.
        rel_id2meta:         rel_id -> metadata dict.
        pool_size:           Total surviving entity + relation count (for popularity norm).
        weights:             ScoringWeights instance.
        max_entity_count:    Batch max entity count for [0,1] normalization (0 = disabled).
        max_relation_count:  Batch max relation count for [0,1] normalization (0 = disabled).
        max_pair_bonus:      Batch max pair bonus for [0,1] normalization (0.0 = disabled).

    Returns:
        SummaryScore with final_score = base_score (redundancy not yet applied).
    """
    ent_ids = summary_to_entities.get(summary_id, set())
    rel_ids = summary_to_rels.get(summary_id, set())

    matched_entity_count = len(ent_ids)
    matched_relation_count = len(rel_ids)

    pair_bonus = 0.0
    if weights.enable_pair_bonus:
        pair_bonus = _compute_pair_bonus(ent_ids, rel_ids, rel_id2meta)

    pop_penalty = 0.0
    if weights.enable_popularity_penalty:
        ref_count = matched_entity_count + matched_relation_count
        pop_penalty = _popularity_penalty(ref_count, pool_size)

    # Normalize graph counts to [0, 1] so they share a scale with semantic_score.
    # When max values are 0 (not provided), fall back to raw integer counts.
    ent_score = matched_entity_count / max_entity_count if max_entity_count > 0 else float(matched_entity_count)
    rel_score = matched_relation_count / max_relation_count if max_relation_count > 0 else float(matched_relation_count)
    pair_score = pair_bonus / max_pair_bonus if max_pair_bonus > 0.0 else pair_bonus

    base = (
        weights.relation_weight * rel_score
        + weights.entity_weight * ent_score
        + weights.pair_bonus_weight * pair_score
        + weights.semantic_weight * semantic_score
        - weights.popularity_penalty_weight * pop_penalty
    )

    matched_entity_names = [
        (entity_id2meta.get(eid, {}) or {}).get("name") or eid
        for eid in sorted(ent_ids)
    ]
    matched_relation_names = [
        _rel_display_label(rid, entity_id2meta, rel_id2meta)
        for rid in sorted(rel_ids)
    ]

    return SummaryScore(
        summary_id=summary_id,
        matched_entity_count=matched_entity_count,
        matched_relation_count=matched_relation_count,
        pair_bonus=pair_bonus,
        semantic_score=semantic_score,
        popularity_penalty=pop_penalty,
        redundancy_penalty=0.0,
        base_score=base,
        final_score=base,
        matched_entity_ids=sorted(ent_ids),
        matched_entity_names=matched_entity_names,
        matched_relation_ids=sorted(rel_ids),
        matched_relation_names=matched_relation_names,
    )


# ---------------------------------------------------------------------------
# Greedy redundancy penalty (MMR-style)
# ---------------------------------------------------------------------------

def _apply_redundancy_penalty_greedy(
    ranked: List[SummaryScore],
    summaries_vdb: Any,
    weights: ScoringWeights,
) -> List[SummaryScore]:
    """
    Greedy MMR-style redundancy re-ranking.

    On each round, recomputes every remaining candidate's redundancy penalty as
    the max cosine similarity to *already selected* summaries, then picks the
    highest scoring remaining candidate.  Complexity: O(k * n) VDB lookups where
    k is topk and n is candidate count.

    This function does NOT call .get() on the whole collection; it fetches one
    embedding at a time via summaries_vdb._collection.get() which is safe for
    the ChromaDB backend used by this project.
    """
    selected_vecs: List[np.ndarray] = []
    result: List[SummaryScore] = []
    candidates = [copy.copy(sc) for sc in ranked]

    while candidates:
        # Recompute final_score for all remaining candidates against already-selected set
        for sc in candidates:
            if not selected_vecs:
                sc.redundancy_penalty = 0.0
            else:
                max_sim = 0.0
                for sel_vec in selected_vecs:
                    sim = summaries_vdb.compare_by_id_raw(sc.summary_id, sel_vec)
                    if sim is not None:
                        max_sim = max(max_sim, float(sim))
                sc.redundancy_penalty = max_sim
            # Always derive final_score from base_score so we don't accumulate errors
            sc.final_score = (
                sc.base_score
                - weights.redundancy_penalty_weight * sc.redundancy_penalty
            )

        candidates.sort(key=lambda sc: sc.final_score, reverse=True)
        best = candidates.pop(0)
        result.append(best)

        # Fetch and cache the selected summary's embedding for future rounds
        try:
            raw = summaries_vdb._collection.get(
                ids=[best.summary_id], include=["embeddings"]
            )
            if raw and raw["ids"]:
                selected_vecs.append(np.array(raw["embeddings"][0], dtype=np.float32))
        except Exception as exc:
            logger.debug("Could not fetch embedding for redundancy: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Public ranking function
# ---------------------------------------------------------------------------

def rank_summaries_by_graph_linked_score(
    scored_events: List[Tuple[float, Dict]],
    context_entities: List[Dict],
    context_relationships: List[Dict],
    cache: Dict,
    weights: ScoringWeights,
    summaries_vdb: Any,
    topk: Optional[int] = None,
) -> List[Tuple[SummaryScore, Dict]]:
    """
    Re-rank a list of ``(semantic_score, provenance_event)`` pairs using the
    graph-linked scoring formula.

    Steps:
    1. Build reverse index: summary_id -> {surviving entity_ids / rel_ids}.
    2. Score each unique summary_id (graph counts + semantic + penalties).
    3. Optionally apply greedy MMR redundancy penalty.
    4. Truncate to topk.

    Args:
        scored_events:         List of (semantic_score, ev) from the semantic scoring pass.
                               Each ev must have a ``summary_id`` field.
        context_entities:      Surviving entities from the retrieval stage.
        context_relationships: Surviving relationships from the retrieval stage.
        cache:                 Global GLOBAL_CACHE dict (for id2meta lookups).
        weights:               ScoringWeights controlling formula terms.
        summaries_vdb:         SummariesVDB instance (needed for redundancy penalty).
        topk:                  Optional hard truncation after ranking.

    Returns:
        List of (SummaryScore, ev) sorted by final_score descending.
    """
    entity_id2meta, rel_id2meta = build_id_to_meta_maps(cache)
    summary_to_entities, summary_to_rels = _build_summary_graph_index(
        context_entities, context_relationships, cache
    )
    pool_size = len(context_entities or []) + len(context_relationships or [])

    # Deduplicate by summary_id, keeping the highest semantic_score per summary
    best_semantic: Dict[str, float] = {}
    event_map: Dict[str, Dict] = {}
    for semantic_score, ev in scored_events:
        sid = ev.get("summary_id")
        if not sid:
            continue
        if sid not in best_semantic or semantic_score > best_semantic[sid]:
            best_semantic[sid] = semantic_score
            event_map[sid] = ev

    # Pre-pass: compute per-candidate counts to find batch maxima for normalization.
    # This puts entity/relation counts and semantic_score on the same [0, 1] scale.
    max_entity_count = max(
        (len(summary_to_entities.get(sid, set())) for sid in best_semantic), default=0
    )
    max_relation_count = max(
        (len(summary_to_rels.get(sid, set())) for sid in best_semantic), default=0
    )
    max_pair_bonus = 0.0
    if weights.enable_pair_bonus:
        for sid in best_semantic:
            pb = _compute_pair_bonus(
                summary_to_entities.get(sid, set()),
                summary_to_rels.get(sid, set()),
                rel_id2meta,
            )
            if pb > max_pair_bonus:
                max_pair_bonus = pb

    # Score each unique summary
    scored: List[SummaryScore] = []
    for sid, semantic_score in best_semantic.items():
        sc = score_summary_graph_linked(
            summary_id=sid,
            semantic_score=semantic_score,
            summary_to_entities=summary_to_entities,
            summary_to_rels=summary_to_rels,
            entity_id2meta=entity_id2meta,
            rel_id2meta=rel_id2meta,
            pool_size=pool_size,
            weights=weights,
            max_entity_count=max_entity_count,
            max_relation_count=max_relation_count,
            max_pair_bonus=max_pair_bonus,
        )
        scored.append(sc)

    # Sort by base score (before redundancy)
    scored.sort(key=lambda sc: sc.base_score, reverse=True)

    # Optional greedy redundancy penalty (modifies final_score in place)
    if weights.enable_redundancy_penalty and summaries_vdb is not None:
        scored = _apply_redundancy_penalty_greedy(scored, summaries_vdb, weights)
    else:
        # final_score already equals base_score; nothing to do
        pass

    # Truncate
    if topk is not None:
        scored = scored[:topk]

    return [(sc, event_map[sc.summary_id]) for sc in scored]


# ===========================================================================
# RRF-based summary scoring
# ===========================================================================

# ---------------------------------------------------------------------------
# RRF data structures
# ---------------------------------------------------------------------------

@dataclass
class SummaryRRFFeatures:
    """Raw graph/semantic features for one candidate, before rank assignment."""
    summary_id: str
    semantic_similarity: float
    matched_entity_count: int
    matched_relation_count: int
    pair_coverage_score: float
    popularity_badness: float
    matched_entity_ids: List[str] = field(default_factory=list)
    matched_entity_names: List[str] = field(default_factory=list)
    matched_relation_ids: List[str] = field(default_factory=list)
    matched_relation_names: List[str] = field(default_factory=list)


@dataclass
class SummaryRRFScore:
    """Per-summary RRF scoring breakdown, with optional MMR redundancy field."""
    summary_id: str
    matched_entity_count: int
    matched_relation_count: int
    pair_coverage_score: float
    semantic_similarity: float
    popularity_badness: float
    relation_rank: int
    entity_rank: int
    pair_coverage_rank: int
    semantic_rank: int
    popularity_badness_rank: int
    positive_score: float
    penalty_score: float
    base_score: float    # positive_score - penalty_score, before MMR
    final_score: float   # base_score - redundancy_penalty_weight * redundancy_penalty
    redundancy_penalty: float = 0.0
    matched_entity_ids: List[str] = field(default_factory=list)
    matched_entity_names: List[str] = field(default_factory=list)
    matched_relation_ids: List[str] = field(default_factory=list)
    matched_relation_names: List[str] = field(default_factory=list)

    def to_log_dict(self, weights: ScoringWeights, mode: str) -> Dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "summary_filter_mode": mode,
            "final_score": round(self.final_score, 6),
            "positive_score": round(self.positive_score, 6),
            "penalty_score": round(self.penalty_score, 6),
            "base_score": round(self.base_score, 6),
            "relation_rank": self.relation_rank,
            "entity_rank": self.entity_rank,
            "pair_coverage_rank": self.pair_coverage_rank,
            "semantic_rank": self.semantic_rank,
            "popularity_badness_rank": self.popularity_badness_rank,
            "matched_entity_count": self.matched_entity_count,
            "matched_relation_count": self.matched_relation_count,
            "pair_coverage_score": round(self.pair_coverage_score, 4),
            "semantic_similarity": round(self.semantic_similarity, 6),
            "popularity_badness": round(self.popularity_badness, 6),
            "redundancy_penalty": round(self.redundancy_penalty, 6),
            "matched_entity_ids": self.matched_entity_ids,
            "matched_entity_names": self.matched_entity_names,
            "matched_relation_ids": self.matched_relation_ids,
            "matched_relation_names": self.matched_relation_names,
            "weights": {
                "relation_weight": weights.relation_weight,
                "entity_weight": weights.entity_weight,
                "pair_bonus_weight": weights.pair_bonus_weight,
                "semantic_weight": weights.semantic_weight,
                "popularity_penalty_weight": weights.popularity_penalty_weight,
                "redundancy_penalty_weight": weights.redundancy_penalty_weight,
                "enable_pair_bonus": weights.enable_pair_bonus,
                "enable_popularity_penalty": weights.enable_popularity_penalty,
                "rrf_k": weights.rrf_k,
            },
        }


# ---------------------------------------------------------------------------
# RRF helpers
# ---------------------------------------------------------------------------

def _assign_ranks_desc(values: List[float]) -> List[int]:
    """
    Dense rank (1 = highest value). Tied items share the minimum rank in their group.
    Example: [3.0, 3.0, 1.0] → [1, 1, 2]
    """
    if not values:
        return []
    sorted_unique = sorted(set(values), reverse=True)
    rank_map = {v: i + 1 for i, v in enumerate(sorted_unique)}
    return [rank_map[v] for v in values]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_summary_graph_features(
    summary_id: str,
    semantic_score: float,
    summary_to_entities: Dict[str, Set[str]],
    summary_to_rels: Dict[str, Set[str]],
    entity_id2meta: Dict,
    rel_id2meta: Dict,
    pool_size: int,
    weights: ScoringWeights,
) -> SummaryRRFFeatures:
    """
    Extract raw graph/semantic features for one summary candidate.
    No ranking is performed here; call rank_summary_features() on the full list.
    """
    ent_ids = summary_to_entities.get(summary_id, set())
    rel_ids = summary_to_rels.get(summary_id, set())

    pair_cov = 0.0
    if weights.enable_pair_bonus:
        pair_cov = _compute_pair_bonus(ent_ids, rel_ids, rel_id2meta)

    pop_bad = 0.0
    if weights.enable_popularity_penalty:
        ref_count = len(ent_ids) + len(rel_ids)
        pop_bad = _popularity_penalty(ref_count, pool_size)

    matched_entity_names = [
        (entity_id2meta.get(eid, {}) or {}).get("name") or eid
        for eid in sorted(ent_ids)
    ]
    matched_relation_names = [
        _rel_display_label(rid, entity_id2meta, rel_id2meta)
        for rid in sorted(rel_ids)
    ]

    return SummaryRRFFeatures(
        summary_id=summary_id,
        semantic_similarity=semantic_score,
        matched_entity_count=len(ent_ids),
        matched_relation_count=len(rel_ids),
        pair_coverage_score=pair_cov,
        popularity_badness=pop_bad,
        matched_entity_ids=sorted(ent_ids),
        matched_entity_names=matched_entity_names,
        matched_relation_ids=sorted(rel_ids),
        matched_relation_names=matched_relation_names,
    )


# ---------------------------------------------------------------------------
# Rank fusion
# ---------------------------------------------------------------------------

def rank_summary_features(
    features_list: List[SummaryRRFFeatures],
    weights: ScoringWeights,
) -> List[SummaryRRFScore]:
    """
    Assign per-dimension dense ranks to all candidates, then compute each
    candidate's RRF score:

        positive_score = relation_weight  * RRF(relation_rank)
                       + entity_weight    * RRF(entity_rank)
                       + pair_bonus_weight * RRF(pair_rank)    [if enable_pair_bonus]
                       + semantic_weight  * RRF(semantic_rank)

        penalty_score  = popularity_penalty_weight * RRF(popularity_rank)
                         [if enable_popularity_penalty, else 0]

        base_score = positive_score - penalty_score
    """
    if not features_list:
        return []

    k = weights.rrf_k

    rel_ranks = _assign_ranks_desc([f.matched_relation_count for f in features_list])
    ent_ranks = _assign_ranks_desc([f.matched_entity_count for f in features_list])
    pair_ranks = _assign_ranks_desc([f.pair_coverage_score for f in features_list])
    sem_ranks = _assign_ranks_desc([f.semantic_similarity for f in features_list])
    pop_ranks = _assign_ranks_desc([f.popularity_badness for f in features_list])

    result: List[SummaryRRFScore] = []
    for i, feat in enumerate(features_list):
        rr, er, pr, sr, br = rel_ranks[i], ent_ranks[i], pair_ranks[i], sem_ranks[i], pop_ranks[i]

        pos = (
            weights.relation_weight * (1.0 / (k + rr))
            + weights.entity_weight * (1.0 / (k + er))
            + (weights.pair_bonus_weight * (1.0 / (k + pr)) if weights.enable_pair_bonus else 0.0)
            + weights.semantic_weight * (1.0 / (k + sr))
        )
        pen = (
            weights.popularity_penalty_weight * (1.0 / (k + br))
            if weights.enable_popularity_penalty else 0.0
        )
        base = pos - pen

        result.append(SummaryRRFScore(
            summary_id=feat.summary_id,
            matched_entity_count=feat.matched_entity_count,
            matched_relation_count=feat.matched_relation_count,
            pair_coverage_score=feat.pair_coverage_score,
            semantic_similarity=feat.semantic_similarity,
            popularity_badness=feat.popularity_badness,
            relation_rank=rr,
            entity_rank=er,
            pair_coverage_rank=pr,
            semantic_rank=sr,
            popularity_badness_rank=br,
            positive_score=pos,
            penalty_score=pen,
            base_score=base,
            final_score=base,
            redundancy_penalty=0.0,
            matched_entity_ids=feat.matched_entity_ids,
            matched_entity_names=feat.matched_entity_names,
            matched_relation_ids=feat.matched_relation_ids,
            matched_relation_names=feat.matched_relation_names,
        ))

    return result


# ---------------------------------------------------------------------------
# Greedy MMR for RRF mode
# ---------------------------------------------------------------------------

def _apply_rrf_mmr_greedy(
    ranked: List[SummaryRRFScore],
    summaries_vdb: Any,
    weights: ScoringWeights,
) -> List[SummaryRRFScore]:
    """
    Greedy MMR loop on top of RRF scores.
    final_score = base_score - redundancy_penalty_weight * max_cosine_to_selected
    """
    selected_vecs: List[np.ndarray] = []
    result: List[SummaryRRFScore] = []
    candidates = [copy.copy(sc) for sc in ranked]

    while candidates:
        for sc in candidates:
            if not selected_vecs:
                sc.redundancy_penalty = 0.0
            else:
                max_sim = 0.0
                for sel_vec in selected_vecs:
                    sim = summaries_vdb.compare_by_id_raw(sc.summary_id, sel_vec)
                    if sim is not None:
                        max_sim = max(max_sim, float(sim))
                sc.redundancy_penalty = max_sim
            sc.final_score = sc.base_score - weights.redundancy_penalty_weight * sc.redundancy_penalty

        candidates.sort(key=lambda sc: sc.final_score, reverse=True)
        best = candidates.pop(0)
        result.append(best)

        try:
            raw = summaries_vdb._collection.get(ids=[best.summary_id], include=["embeddings"])
            if raw and raw["ids"]:
                selected_vecs.append(np.array(raw["embeddings"][0], dtype=np.float32))
        except Exception as exc:
            logger.debug("Could not fetch embedding for RRF MMR: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Public RRF selection functions
# ---------------------------------------------------------------------------

def _build_rrf_scored_list(
    scored_events: List[Tuple[float, Dict]],
    context_entities: List[Dict],
    context_relationships: List[Dict],
    cache: Dict,
    weights: ScoringWeights,
) -> Tuple[List[SummaryRRFScore], Dict[str, Dict]]:
    """Shared setup: dedup → feature extraction → rank fusion → base sort."""
    entity_id2meta, rel_id2meta = build_id_to_meta_maps(cache)
    summary_to_entities, summary_to_rels = _build_summary_graph_index(
        context_entities, context_relationships, cache
    )
    pool_size = len(context_entities or []) + len(context_relationships or [])

    best_semantic: Dict[str, float] = {}
    event_map: Dict[str, Dict] = {}
    for sem_score, ev in scored_events:
        sid = ev.get("summary_id")
        if not sid:
            continue
        if sid not in best_semantic or sem_score > best_semantic[sid]:
            best_semantic[sid] = sem_score
            event_map[sid] = ev

    features_list = [
        compute_summary_graph_features(
            summary_id=sid,
            semantic_score=sem_score,
            summary_to_entities=summary_to_entities,
            summary_to_rels=summary_to_rels,
            entity_id2meta=entity_id2meta,
            rel_id2meta=rel_id2meta,
            pool_size=pool_size,
            weights=weights,
        )
        for sid, sem_score in best_semantic.items()
    ]

    scored = rank_summary_features(features_list, weights)
    scored.sort(key=lambda sc: sc.base_score, reverse=True)
    return scored, event_map


def select_summaries_rrf(
    scored_events: List[Tuple[float, Dict]],
    context_entities: List[Dict],
    context_relationships: List[Dict],
    cache: Dict,
    weights: ScoringWeights,
    summaries_vdb: Any,
    topk: Optional[int] = None,
) -> List[Tuple[SummaryRRFScore, Dict]]:
    """
    Rank candidates by RRF score (no redundancy control).
    Use for summary_filter_mode="graph_rrf".
    """
    scored, event_map = _build_rrf_scored_list(
        scored_events, context_entities, context_relationships, cache, weights
    )
    if topk is not None:
        scored = scored[:topk]
    return [(sc, event_map[sc.summary_id]) for sc in scored]


def select_summaries_rrf_mmr(
    scored_events: List[Tuple[float, Dict]],
    context_entities: List[Dict],
    context_relationships: List[Dict],
    cache: Dict,
    weights: ScoringWeights,
    summaries_vdb: Any,
    topk: Optional[int] = None,
) -> List[Tuple[SummaryRRFScore, Dict]]:
    """
    Rank candidates by RRF score then apply greedy MMR redundancy control.
    Use for summary_filter_mode="graph_rrf_mmr".
    """
    scored, event_map = _build_rrf_scored_list(
        scored_events, context_entities, context_relationships, cache, weights
    )
    if summaries_vdb is not None:
        scored = _apply_rrf_mmr_greedy(scored, summaries_vdb, weights)
    if topk is not None:
        scored = scored[:topk]
    return [(sc, event_map[sc.summary_id]) for sc in scored]
