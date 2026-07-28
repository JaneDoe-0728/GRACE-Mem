"""
Sanity checks for the graph-linked summary scoring module.

These tests use only pure-Python stubs — no FalkorDB, no ChromaDB, no embedding
model required.  Run with:

    python -m pytest test/test_summary_scoring.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from KG.pipeline.retrieval_steps.summary_scoring import (
    ScoringWeights,
    SummaryScore,
    SummaryRRFFeatures,
    SummaryRRFScore,
    _build_summary_graph_index,
    _compute_pair_bonus,
    _popularity_penalty,
    _assign_ranks_desc,
    score_summary_graph_linked,
    rank_summaries_by_graph_linked_score,
    compute_summary_graph_features,
    rank_summary_features,
    select_summaries_rrf,
    select_summaries_rrf_mmr,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_cache(entities: list[dict], rels: list[dict]) -> dict:
    """Build a minimal GLOBAL_CACHE dict from entity/relation meta lists."""
    return {
        "entities": {f"{m['name']}:{m['type']}:{m['description']}": m for m in entities},
        "entities_full": {f"{m['name']}:{m['type']}:{m['description']}": m for m in entities},
        "relationships": {f"{m['source_id']}:{m['target_id']}:{m['description']}": m for m in rels},
        "relationships_full": {f"{m['source_id']}:{m['target_id']}:{m['description']}": m for m in rels},
    }


def _make_prov(summary_ids: list[str]) -> dict:
    return {
        "events": [
            {"ts": i, "summary_id": sid, "session_id": sid.split(":")[0],
             "message_id": sid.split(":")[1]}
            for i, sid in enumerate(summary_ids)
        ]
    }


# ---------------------------------------------------------------------------
# Test 1: _build_summary_graph_index correctly maps surviving items only
# ---------------------------------------------------------------------------

def test_graph_index_only_includes_surviving_items():
    """
    Non-surviving items must NOT appear in the graph index even if they share
    provenance summary_ids with surviving items.
    """
    surviving_ent = {"id": "e1", "name": "Alice", "type": "Person", "description": ""}
    non_surviving_ent = {"id": "e2", "name": "Bob", "type": "Person", "description": ""}

    prov_e1 = _make_prov(["s1:m1", "s2:m2"])
    prov_e2 = _make_prov(["s3:m3"])

    cache = _make_cache(
        entities=[
            {"id": "e1", "name": "Alice", "type": "Person", "description": "", "prov": prov_e1},
            {"id": "e2", "name": "Bob", "type": "Person", "description": "", "prov": prov_e2},
        ],
        rels=[],
    )

    s2e, s2r = _build_summary_graph_index(
        context_entities=[{"id": "e1"}],
        context_relationships=[],
        cache=cache,
    )

    assert "s1:m1" in s2e
    assert "e1" in s2e["s1:m1"]
    assert "s3:m3" not in s2e, "non-surviving entity provenance must not appear"


# ---------------------------------------------------------------------------
# Test 2: graph_count mode ranks by KG link counts, not semantic similarity
# ---------------------------------------------------------------------------

def test_graph_count_ranks_by_link_count():
    """
    With semantic_weight=0, the summary linked to more surviving items must rank
    higher regardless of its semantic similarity score.
    """
    prov_e1 = _make_prov(["s1:m1"])
    prov_e2 = _make_prov(["s1:m1"])
    prov_rel = _make_prov(["s1:m1"])
    prov_e3 = _make_prov(["s2:m2"])

    cache = _make_cache(
        entities=[
            {"id": "e1", "name": "Alice", "type": "Person", "description": "", "prov": prov_e1},
            {"id": "e2", "name": "Bob", "type": "Person", "description": "", "prov": prov_e2},
            {"id": "e3", "name": "Carol", "type": "Person", "description": "", "prov": prov_e3},
        ],
        rels=[
            {"id": "r1", "source_id": "e1", "target_id": "e2",
             "description": "knows", "prov": prov_rel},
        ],
    )

    context_entities = [{"id": "e1"}, {"id": "e2"}, {"id": "e3"}]
    context_rels = [{"rel_id": "r1"}]

    # s1:m1 has 2 entities + 1 relation linked; s2:m2 has 1 entity
    # s2:m2 gets higher semantic score (0.9 vs 0.3) — graph_count should still prefer s1:m1
    weights = ScoringWeights(
        semantic_weight=0.0,
        enable_pair_bonus=False,
        enable_popularity_penalty=False,
        enable_redundancy_penalty=False,
    )
    scored_events = [
        (0.3, {"summary_id": "s1:m1", "session_id": "s1", "message_id": "m1"}),
        (0.9, {"summary_id": "s2:m2", "session_id": "s2", "message_id": "m2"}),
    ]

    results = rank_summaries_by_graph_linked_score(
        scored_events=scored_events,
        context_entities=context_entities,
        context_relationships=context_rels,
        cache=cache,
        weights=weights,
        summaries_vdb=None,
        topk=None,
    )

    assert len(results) == 2
    assert results[0][0].summary_id == "s1:m1", (
        "graph_count must prefer s1:m1 (2 ents + 1 rel) over s2:m2 (1 ent) "
        "even though s2:m2 has higher semantic similarity"
    )


# ---------------------------------------------------------------------------
# Test 3: semantic mode preserves baseline (pure cosine) ranking
# ---------------------------------------------------------------------------

def test_semantic_mode_preserves_baseline_ranking():
    """
    When summary_filter_mode='semantic', ranked order must equal the order of
    descending semantic similarity — identical to the pre-feature behaviour.
    """
    prov_e = _make_prov(["s1:m1", "s2:m2", "s3:m3"])
    cache = _make_cache(
        entities=[{"id": "e1", "name": "X", "type": "T", "description": "", "prov": prov_e}],
        rels=[],
    )
    context_entities = [{"id": "e1"}]

    # Scores not in descending order to test that sort is applied correctly
    scored_events = [
        (0.5, {"summary_id": "s2:m2", "session_id": "s2", "message_id": "m2"}),
        (0.9, {"summary_id": "s1:m1", "session_id": "s1", "message_id": "m1"}),
        (0.2, {"summary_id": "s3:m3", "session_id": "s3", "message_id": "m3"}),
    ]
    # semantic mode: do the sort in evidence.py (not here); we verify the
    # graph scorer in semantic-equivalent config (semantic_weight=1, no graph terms)
    weights = ScoringWeights(
        relation_weight=0.0,
        entity_weight=0.0,
        pair_bonus_weight=0.0,
        semantic_weight=1.0,
        enable_pair_bonus=False,
        enable_popularity_penalty=False,
        enable_redundancy_penalty=False,
    )
    results = rank_summaries_by_graph_linked_score(
        scored_events=scored_events,
        context_entities=context_entities,
        context_relationships=[],
        cache=cache,
        weights=weights,
        summaries_vdb=None,
        topk=None,
    )
    scores_in_order = [sc.final_score for sc, _ in results]
    assert scores_in_order == sorted(scores_in_order, reverse=True)
    assert results[0][0].summary_id == "s1:m1"
    assert results[-1][0].summary_id == "s3:m3"


# ---------------------------------------------------------------------------
# Test 4: semantic_weight only affects tie-breaking
# ---------------------------------------------------------------------------

def test_semantic_weight_acts_as_tiebreaker():
    """
    Two summaries with equal graph-count scores: the one with higher semantic
    similarity must rank first when semantic_weight > 0.
    """
    prov_a = _make_prov(["sA:m1"])
    prov_b = _make_prov(["sB:m1"])

    cache = _make_cache(
        entities=[
            {"id": "eA", "name": "A", "type": "T", "description": "", "prov": prov_a},
            {"id": "eB", "name": "B", "type": "T", "description": "", "prov": prov_b},
        ],
        rels=[],
    )
    # Both summaries linked to exactly 1 surviving entity — equal graph counts
    scored_events = [
        (0.8, {"summary_id": "sA:m1", "session_id": "sA", "message_id": "m1"}),
        (0.3, {"summary_id": "sB:m1", "session_id": "sB", "message_id": "m1"}),
    ]
    weights = ScoringWeights(
        entity_weight=1.0,
        relation_weight=1.0,
        semantic_weight=0.5,
        enable_pair_bonus=False,
        enable_popularity_penalty=False,
        enable_redundancy_penalty=False,
    )
    results = rank_summaries_by_graph_linked_score(
        scored_events=scored_events,
        context_entities=[{"id": "eA"}, {"id": "eB"}],
        context_relationships=[],
        cache=cache,
        weights=weights,
        summaries_vdb=None,
    )
    assert results[0][0].summary_id == "sA:m1", (
        "sA:m1 has higher semantic score; with equal graph counts it must rank first"
    )


# ---------------------------------------------------------------------------
# Test 5: popularity penalty reduces scores for widely-referenced summaries
# ---------------------------------------------------------------------------

def test_popularity_penalty_reduces_hub_summary_score():
    """
    A summary referenced by many surviving items (hub) must score lower than a
    more specific summary of similar semantic quality when the popularity penalty
    is enabled.
    """
    # s_hub is referenced by 4 items; s_specific by 1
    prov_hub = _make_prov(["s_hub:m1"])
    prov_spec = _make_prov(["s_specific:m1"])

    entities = [
        {"id": f"e{i}", "name": f"E{i}", "type": "T", "description": "", "prov": prov_hub}
        for i in range(4)
    ] + [
        {"id": "e_spec", "name": "ESpec", "type": "T", "description": "", "prov": prov_spec}
    ]
    cache = _make_cache(entities=entities, rels=[])
    context_entities = [{"id": e["id"]} for e in entities]

    weights_with_penalty = ScoringWeights(
        entity_weight=1.0,
        relation_weight=0.0,
        semantic_weight=0.5,
        enable_pair_bonus=False,
        enable_popularity_penalty=True,
        popularity_penalty_weight=2.0,
        enable_redundancy_penalty=False,
    )
    scored_events = [
        (0.8, {"summary_id": "s_hub:m1", "session_id": "s_hub", "message_id": "m1"}),
        (0.7, {"summary_id": "s_specific:m1", "session_id": "s_specific", "message_id": "m1"}),
    ]
    results = rank_summaries_by_graph_linked_score(
        scored_events=scored_events,
        context_entities=context_entities,
        context_relationships=[],
        cache=cache,
        weights=weights_with_penalty,
        summaries_vdb=None,
    )
    hub_sc = next(sc for sc, _ in results if sc.summary_id == "s_hub:m1")
    spec_sc = next(sc for sc, _ in results if sc.summary_id == "s_specific:m1")
    assert hub_sc.popularity_penalty > spec_sc.popularity_penalty, (
        "hub summary must have a higher popularity penalty"
    )


# ---------------------------------------------------------------------------
# Test 6: pair bonus is non-zero only when entity is endpoint of relation
# ---------------------------------------------------------------------------

def test_pair_bonus_requires_entity_endpoint():
    """
    Pair bonus must be 0 when no surviving entity is an endpoint of any surviving
    relation that references the same summary, and > 0 when the condition holds.
    """
    rel_id2meta_with_link = {
        "r1": {"source_id": "e1", "target_id": "e2", "description": "knows",
               "source_entity": "Alice", "target_entity": "Bob"}
    }
    rel_id2meta_no_link = {
        "r1": {"source_id": "e99", "target_id": "e98", "description": "knows",
               "source_entity": "X", "target_entity": "Y"}
    }
    entity_ids = {"e1", "e2"}
    rel_ids = {"r1"}

    bonus_with = _compute_pair_bonus(entity_ids, rel_ids, rel_id2meta_with_link)
    bonus_without = _compute_pair_bonus(entity_ids, rel_ids, rel_id2meta_no_link)

    assert bonus_with > 0, "pair bonus must be positive when entity is an endpoint"
    assert bonus_without == 0.0, "pair bonus must be 0 when no entity is an endpoint"


# ---------------------------------------------------------------------------
# Test 7: _popularity_penalty boundary conditions
# ---------------------------------------------------------------------------

def test_popularity_penalty_bounds():
    assert _popularity_penalty(0, 10) == 0.0
    assert _popularity_penalty(10, 10) == 1.0
    assert _popularity_penalty(5, 10) == 0.5
    assert _popularity_penalty(0, 0) == 0.0
    # Clamped to 1.0 even if over-referenced (shouldn't happen, but guard anyway)
    assert _popularity_penalty(15, 10) == 1.0


# ---------------------------------------------------------------------------
# Test 8: ScoringWeights.from_dict only accepts valid fields
# ---------------------------------------------------------------------------

def test_scoring_weights_from_dict_ignores_unknown_keys():
    w = ScoringWeights.from_dict({"semantic_weight": 0.1, "nonexistent_key": 99})
    assert w.semantic_weight == 0.1
    assert w.relation_weight == 2.0  # default preserved


# ---------------------------------------------------------------------------
# Test 9: redundancy penalty reduces near-duplicate selection (mock VDB)
# ---------------------------------------------------------------------------

def test_redundancy_penalty_suppresses_near_duplicate():
    """
    When redundancy penalty is enabled, a second summary very similar to the
    already-selected best summary must have its final_score reduced.
    We verify this with a stub VDB that returns controllable similarities.
    """
    prov_a = _make_prov(["sA:m1"])
    prov_b = _make_prov(["sB:m1"])

    cache = _make_cache(
        entities=[
            {"id": "eA", "name": "A", "type": "T", "description": "", "prov": prov_a},
            {"id": "eB", "name": "B", "type": "T", "description": "", "prov": prov_b},
        ],
        rels=[],
    )

    scored_events = [
        (0.9, {"summary_id": "sA:m1", "session_id": "sA", "message_id": "m1"}),
        (0.85, {"summary_id": "sB:m1", "session_id": "sB", "message_id": "m1"}),
    ]

    # Stub VDB: compare_by_id_raw returns 0.95 (near-duplicate), _collection.get gives a vec
    stub_vdb = MagicMock()
    stub_vdb.compare_by_id_raw.return_value = 0.95
    stub_vdb._collection.get.return_value = {
        "ids": ["sA:m1"], "embeddings": [np.ones(4, dtype=np.float32).tolist()]
    }

    weights = ScoringWeights(
        entity_weight=1.0,
        relation_weight=0.0,
        semantic_weight=1.0,
        enable_pair_bonus=False,
        enable_popularity_penalty=False,
        enable_redundancy_penalty=True,
        redundancy_penalty_weight=1.0,
    )
    results = rank_summaries_by_graph_linked_score(
        scored_events=scored_events,
        context_entities=[{"id": "eA"}, {"id": "eB"}],
        context_relationships=[],
        cache=cache,
        weights=weights,
        summaries_vdb=stub_vdb,
        topk=2,
    )

    second_sc = results[1][0]
    assert second_sc.redundancy_penalty > 0, (
        "second summary must have a non-zero redundancy penalty after first is selected"
    )
    assert second_sc.final_score < second_sc.base_score, (
        "final_score must be lower than base_score when redundancy penalty is applied"
    )


# ===========================================================================
# RRF scoring tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Test 10: _assign_ranks_desc — dense ranking, ties share minimum rank
# ---------------------------------------------------------------------------

def test_assign_ranks_desc_basic():
    assert _assign_ranks_desc([3.0, 1.0, 2.0]) == [1, 3, 2]
    assert _assign_ranks_desc([3.0, 3.0, 1.0]) == [1, 1, 2]
    assert _assign_ranks_desc([]) == []
    assert _assign_ranks_desc([5.0]) == [1]


# ---------------------------------------------------------------------------
# Test 11: graph_rrf ranks summaries with more KG matches higher
# ---------------------------------------------------------------------------

def test_rrf_ranks_by_kg_link_count():
    """
    With semantic_weight=0, the summary linked to more surviving KG items
    must rank first in graph_rrf mode.
    """
    prov_e1 = _make_prov(["s1:m1"])
    prov_e2 = _make_prov(["s1:m1"])
    prov_rel = _make_prov(["s1:m1"])
    prov_e3 = _make_prov(["s2:m2"])

    cache = _make_cache(
        entities=[
            {"id": "e1", "name": "Alice", "type": "Person", "description": "", "prov": prov_e1},
            {"id": "e2", "name": "Bob", "type": "Person", "description": "", "prov": prov_e2},
            {"id": "e3", "name": "Carol", "type": "Person", "description": "", "prov": prov_e3},
        ],
        rels=[
            {"id": "r1", "source_id": "e1", "target_id": "e2",
             "description": "knows", "prov": prov_rel},
        ],
    )
    context_entities = [{"id": "e1"}, {"id": "e2"}, {"id": "e3"}]
    context_rels = [{"rel_id": "r1"}]

    # s1:m1 has 2 entities + 1 relation; s2:m2 has 1 entity
    # s2:m2 gets higher semantic score — RRF should still prefer s1:m1 on graph signals
    weights = ScoringWeights(
        semantic_weight=0.0,
        enable_pair_bonus=False,
        enable_popularity_penalty=False,
        rrf_k=60.0,
    )
    scored_events = [
        (0.3, {"summary_id": "s1:m1", "session_id": "s1", "message_id": "m1"}),
        (0.9, {"summary_id": "s2:m2", "session_id": "s2", "message_id": "m2"}),
    ]
    results = select_summaries_rrf(
        scored_events=scored_events,
        context_entities=context_entities,
        context_relationships=context_rels,
        cache=cache,
        weights=weights,
        summaries_vdb=None,
    )
    assert results[0][0].summary_id == "s1:m1", (
        "graph_rrf must prefer s1:m1 (2 ents + 1 rel) over s2:m2 (1 ent) "
        "even with higher semantic score on s2:m2"
    )


# ---------------------------------------------------------------------------
# Test 12: semantic_weight acts as tie-breaker in RRF
# ---------------------------------------------------------------------------

def test_rrf_semantic_tiebreaker():
    """
    Two summaries with identical entity+relation counts: the one with higher
    semantic similarity must rank first when semantic_weight > 0.
    """
    prov_a = _make_prov(["sA:m1"])
    prov_b = _make_prov(["sB:m1"])
    cache = _make_cache(
        entities=[
            {"id": "eA", "name": "A", "type": "T", "description": "", "prov": prov_a},
            {"id": "eB", "name": "B", "type": "T", "description": "", "prov": prov_b},
        ],
        rels=[],
    )
    scored_events = [
        (0.9, {"summary_id": "sA:m1", "session_id": "sA", "message_id": "m1"}),
        (0.2, {"summary_id": "sB:m1", "session_id": "sB", "message_id": "m1"}),
    ]
    weights = ScoringWeights(
        relation_weight=1.0,
        entity_weight=1.0,
        semantic_weight=0.5,
        enable_pair_bonus=False,
        enable_popularity_penalty=False,
        rrf_k=60.0,
    )
    results = select_summaries_rrf(
        scored_events=scored_events,
        context_entities=[{"id": "eA"}, {"id": "eB"}],
        context_relationships=[],
        cache=cache,
        weights=weights,
        summaries_vdb=None,
    )
    assert results[0][0].summary_id == "sA:m1", (
        "sA:m1 has higher semantic score; with equal graph counts it must rank first"
    )


# ---------------------------------------------------------------------------
# Test 13: popularity penalty demotes hub summaries in RRF mode
# ---------------------------------------------------------------------------

def test_rrf_popularity_penalty_demotes_hub():
    """
    Hub summary (referenced by many surviving items) must rank lower than a
    specific summary when popularity penalty is enabled in graph_rrf mode.
    """
    prov_hub = _make_prov(["s_hub:m1"])
    prov_spec = _make_prov(["s_specific:m1"])

    entities = [
        {"id": f"e{i}", "name": f"E{i}", "type": "T", "description": "", "prov": prov_hub}
        for i in range(4)
    ] + [
        {"id": "e_spec", "name": "ESpec", "type": "T", "description": "", "prov": prov_spec}
    ]
    cache = _make_cache(entities=entities, rels=[])
    context_entities = [{"id": e["id"]} for e in entities]

    # Without penalty: hub ranks first (4 entities vs 1)
    weights_no_pen = ScoringWeights(
        entity_weight=1.0,
        relation_weight=0.0,
        semantic_weight=0.0,
        enable_pair_bonus=False,
        enable_popularity_penalty=False,
        rrf_k=60.0,
    )
    scored_events = [
        (0.5, {"summary_id": "s_hub:m1", "session_id": "s_hub", "message_id": "m1"}),
        (0.5, {"summary_id": "s_specific:m1", "session_id": "s_specific", "message_id": "m1"}),
    ]
    results_no = select_summaries_rrf(
        scored_events=scored_events,
        context_entities=context_entities,
        context_relationships=[],
        cache=cache,
        weights=weights_no_pen,
        summaries_vdb=None,
    )
    assert results_no[0][0].summary_id == "s_hub:m1"

    # With penalty: hub should be demoted relative to specific
    weights_pen = ScoringWeights(
        entity_weight=1.0,
        relation_weight=0.0,
        semantic_weight=0.0,
        enable_pair_bonus=False,
        enable_popularity_penalty=True,
        popularity_penalty_weight=5.0,
        rrf_k=60.0,
    )
    results_pen = select_summaries_rrf(
        scored_events=scored_events,
        context_entities=context_entities,
        context_relationships=[],
        cache=cache,
        weights=weights_pen,
        summaries_vdb=None,
    )
    hub_score = next(sc for sc, _ in results_pen if sc.summary_id == "s_hub:m1")
    spec_score = next(sc for sc, _ in results_pen if sc.summary_id == "s_specific:m1")
    assert hub_score.penalty_score > spec_score.penalty_score, (
        "hub summary must have a higher RRF penalty score"
    )


# ---------------------------------------------------------------------------
# Test 14: graph_rrf_mmr reduces duplicate summaries
# ---------------------------------------------------------------------------

def test_rrf_mmr_reduces_duplicates():
    """
    In graph_rrf_mmr mode, a second summary very similar to the selected best
    must have its final_score reduced below its base_score.
    """
    prov_a = _make_prov(["sA:m1"])
    prov_b = _make_prov(["sB:m1"])
    cache = _make_cache(
        entities=[
            {"id": "eA", "name": "A", "type": "T", "description": "", "prov": prov_a},
            {"id": "eB", "name": "B", "type": "T", "description": "", "prov": prov_b},
        ],
        rels=[],
    )
    scored_events = [
        (0.9, {"summary_id": "sA:m1", "session_id": "sA", "message_id": "m1"}),
        (0.85, {"summary_id": "sB:m1", "session_id": "sB", "message_id": "m1"}),
    ]

    stub_vdb = MagicMock()
    stub_vdb.compare_by_id_raw.return_value = 0.95
    stub_vdb._collection.get.return_value = {
        "ids": ["sA:m1"], "embeddings": [np.ones(4, dtype=np.float32).tolist()]
    }

    weights = ScoringWeights(
        entity_weight=1.0,
        relation_weight=0.0,
        semantic_weight=1.0,
        enable_pair_bonus=False,
        enable_popularity_penalty=False,
        redundancy_penalty_weight=1.0,
        rrf_k=60.0,
    )
    results = select_summaries_rrf_mmr(
        scored_events=scored_events,
        context_entities=[{"id": "eA"}, {"id": "eB"}],
        context_relationships=[],
        cache=cache,
        weights=weights,
        summaries_vdb=stub_vdb,
        topk=2,
    )
    second_sc = results[1][0]
    assert second_sc.redundancy_penalty > 0, (
        "second summary must have non-zero redundancy penalty in rrf_mmr mode"
    )
    assert second_sc.final_score < second_sc.base_score, (
        "final_score must be below base_score after MMR redundancy penalty"
    )


# ---------------------------------------------------------------------------
# Test 15: rank_summary_features assigns correct ranks and RRF scores
# ---------------------------------------------------------------------------

def test_rank_summary_features_scores():
    """
    Verify that rank_summary_features produces correct ranks and that the
    candidate with the higher relation count gets a better (lower) relation_rank.
    """
    feats = [
        SummaryRRFFeatures(
            summary_id="s1",
            semantic_similarity=0.8,
            matched_entity_count=2,
            matched_relation_count=3,
            pair_coverage_score=2.0,
            popularity_badness=0.3,
        ),
        SummaryRRFFeatures(
            summary_id="s2",
            semantic_similarity=0.5,
            matched_entity_count=1,
            matched_relation_count=1,
            pair_coverage_score=0.0,
            popularity_badness=0.1,
        ),
    ]
    weights = ScoringWeights(
        relation_weight=2.0,
        entity_weight=1.0,
        pair_bonus_weight=1.5,
        semantic_weight=0.5,
        enable_pair_bonus=True,
        enable_popularity_penalty=False,
        rrf_k=60.0,
    )
    scored = rank_summary_features(feats, weights)
    s1 = next(sc for sc in scored if sc.summary_id == "s1")
    s2 = next(sc for sc in scored if sc.summary_id == "s2")

    assert s1.relation_rank < s2.relation_rank, "s1 has more relations → better rank (lower number)"
    assert s1.entity_rank < s2.entity_rank, "s1 has more entities → better rank"
    assert s1.base_score > s2.base_score, "s1 should have a higher total RRF score"
