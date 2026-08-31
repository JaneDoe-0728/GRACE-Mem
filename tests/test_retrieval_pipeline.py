"""Characterization tests for the retrieval pipeline.

`assemble_context_from_query` is 780 lines at 0% coverage, and it is about to
be taken apart (docs/retriever-refactor.md). These tests exist to make that
safe: they drive the real method through `retrieval_fakes`, snapshot what it
produces, and fail if a refactor changes it.

They assert almost nothing about what the values *should* be. That is the
point -- a characterization test records what the code does today so a
behaviour-preserving change can be shown to preserve it. Judgements about
whether today's behaviour is correct belong in other tests.

Every value of `filter_method` gets its own snapshot, because the five-way
dispatch is exactly the part a reviewer cannot check by reading a large diff.

Regenerate after an intentional behaviour change, never to make a red test
green:

    KG_UPDATE_RETRIEVAL_SNAPSHOTS=1 uv run pytest tests/test_retrieval_pipeline.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from grace_mem.retrieval.pipeline import Retriever
from grace_mem.retrieval.config import RetrieverConfig
from tests.retrieval_fakes import (
    CallLog,
    FakeEvidenceFilter,
    FakeGraph,
    FakePageRank,
    FakeSearcher,
    FakeSpreadingActivation,
    cache,
)

SNAPSHOT_DIR = Path(__file__).parent / "fixtures"
FILTER_METHODS = ["similarity", "rrf", "ppr", "rrf+ppr", "reranker_only"]

QUESTION = "What did the marathon runner say about new shoes?"
LOW_LEVEL = ["marathon", "shoes"]
HIGH_LEVEL = ["running", "purchase"]


def _retriever(filter_method: str, **overrides) -> Retriever:
    """A Retriever wired to doubles, bypassing an __init__ that wants real services.

    The same pattern as tests/test_adaptive_trace.py. Only the components
    assemble_context_from_query actually reaches are populated; anything else
    it touched would raise, which is a useful failure rather than a silent one.
    """
    r = object.__new__(Retriever)
    r.cfg = RetrieverConfig(filter_method=filter_method, use_spreading_activation=True, **overrides)
    r.log = CallLog()
    r.searcher = FakeSearcher(r.log)
    r.graph = FakeGraph(r.log)
    r.evidence_filter = FakeEvidenceFilter(r.log)
    r.ppr_engine = FakePageRank(r.log)
    r.sa_engine = FakeSpreadingActivation(r.log)
    r.cache = cache()
    r.llm = None
    r.embed = None
    r._last_stage_trace = None
    return r


def _capture(filter_method: str) -> dict:
    r = _retriever(filter_method)
    entities, relationships, context_text, query_vec = r.assemble_context_from_query(
        question=QUESTION,
        low_level_keywords=LOW_LEVEL,
        high_level_keywords=HIGH_LEVEL,
        request_id="snapshot",
    )
    return {
        "entity_ids": [e.get("id") for e in entities],
        "relationship_ids": [x.get("rel_id") or x.get("id") for x in relationships],
        "entity_count": len(entities),
        "relationship_count": len(relationships),
        "context_text": context_text,
        "query_vec_shape": list(query_vec.shape) if query_vec is not None else None,
        "calls": r.log.entries,
    }


@pytest.mark.parametrize("filter_method", FILTER_METHODS)
def test_assemble_context_matches_snapshot(filter_method: str) -> None:
    """The candidate sets and rendered context are unchanged for this dispatch path."""
    path = SNAPSHOT_DIR / f"retrieval_{filter_method.replace('+', '_')}.json"
    actual = _capture(filter_method)

    if os.getenv("KG_UPDATE_RETRIEVAL_SNAPSHOTS") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"snapshot rewritten: {path.name}")

    assert path.exists(), (
        f"no snapshot for filter_method={filter_method}. "
        "Generate with KG_UPDATE_RETRIEVAL_SNAPSHOTS=1."
    )
    assert actual == json.loads(path.read_text(encoding="utf-8"))


def test_every_dispatch_path_reaches_the_search_stage() -> None:
    """A guard on the guard: a snapshot of an empty run would pass and prove nothing."""
    for method in FILTER_METHODS:
        captured = _capture(method)
        names = [c["call"] for c in captured["calls"]]
        for required in ("searcher.embed_query", "searcher.search_entities_hybrid",
                         "searcher.search_relationships_by_vec",
                         "filter.compute_subgraph_intersection"):
            assert required in names, f"{method} never called {required}: {names}"
        assert names[0] == "searcher.embed_query", f"{method} did not embed first: {names[0]}"


def test_the_dispatch_paths_do_not_all_agree() -> None:
    """If every filter_method produced the same result, the snapshots would not
    distinguish a broken dispatch from a working one."""
    results = {m: tuple(_capture(m)["entity_ids"]) for m in FILTER_METHODS}
    assert len(set(results.values())) > 1, f"all five paths returned the same entities: {results}"


def test_an_empty_candidate_pool_short_circuits_the_whole_query() -> None:
    """When nothing is reachable, retrieval returns empty rather than continuing.

    This path is not in the snapshots -- the fixture always finds something --
    and it was nearly lost when stage 1 became its own method: the bare `return`
    that used to end the whole query would have ended only the stage. mypy
    caught it because the return types disagreed. This test catches it whether
    or not the types happen to.
    """
    r = _retriever("similarity")

    class _EmptyGraph:
        def get_node_subgraph(self, entity_ids):
            return {}

        def get_edge_subgraph(self, rel_ids):
            return []

    r.graph = _EmptyGraph()
    r.cache = {"entities": {}, "relationships": {}}

    entities, relationships, context_text, query_vec = r.assemble_context_from_query(
        question=QUESTION,
        low_level_keywords=LOW_LEVEL,
        high_level_keywords=HIGH_LEVEL,
        request_id="empty",
    )

    assert entities == []
    assert relationships == []
    assert context_text == ""
    assert query_vec is not None, "the query vector survives the short circuit"

    # The empty result alone does not prove the short circuit fired: the later
    # stages would also produce nothing from an empty pool. What distinguishes
    # them is that they never ran.
    ran = [c["call"] for c in r.log.entries]
    assert not [c for c in ran if c.startswith("filter.")], (
        f"filtering ran after an empty pool: {ran}"
    )
