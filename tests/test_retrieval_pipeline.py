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

from grace_mem.retrieval.config import RetrieverConfig
from grace_mem.retrieval.pipeline import Retriever
from tests.retrieval_fakes import (
    CallLog,
    FakeEvidenceFilter,
    FakeGraph,
    FakeSearcher,
    FakeSpreadingActivation,
    cache,
)

SNAPSHOT_DIR = Path(__file__).parent / "fixtures"
# Only one filter strategy remains: the reranker is the filter. The other four
# were deleted with summary_scoring -- the paper never used them, and neither
# did experiment_config.
FILTER_METHODS = ["reranker_only"]

QUESTION = "What did the marathon runner say about new shoes?"
LOW_LEVEL = ["marathon", "shoes"]
HIGH_LEVEL = ["running", "purchase"]


def _retriever(**overrides) -> Retriever:
    """A Retriever wired to doubles, bypassing an __init__ that wants real services.

    The same pattern as tests/test_adaptive_trace.py. Only the components
    assemble_context_from_query actually reaches are populated; anything else
    it touched would raise, which is a useful failure rather than a silent one.
    """
    r = object.__new__(Retriever)
    r.cfg = RetrieverConfig(use_spreading_activation=True, **overrides)
    r.log = CallLog()
    r.searcher = FakeSearcher(r.log)
    r.graph = FakeGraph(r.log)
    r.evidence_filter = FakeEvidenceFilter(r.log)
    r.sa_engine = FakeSpreadingActivation(r.log)
    r.cache = cache()
    r.llm = None
    r.embed = None
    r._last_stage_trace = None
    return r


def _capture() -> dict:
    r = _retriever()
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


def test_assemble_context_matches_snapshot() -> None:
    """The candidate set and rendered context are unchanged."""
    path = SNAPSHOT_DIR / "retrieval_reranker_only.json"
    actual = _capture()

    if os.getenv("KG_UPDATE_RETRIEVAL_SNAPSHOTS") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"snapshot rewritten: {path.name}")

    assert path.exists(), "no snapshot. Generate with KG_UPDATE_RETRIEVAL_SNAPSHOTS=1."
    assert actual == json.loads(path.read_text(encoding="utf-8"))


def test_the_search_stage_actually_runs() -> None:
    """A guard on the guard: a snapshot of an empty run would pass and prove nothing."""
    for method in FILTER_METHODS:
        captured = _capture()
        names = [c["call"] for c in captured["calls"]]
        for required in ("searcher.embed_query", "searcher.search_entities_hybrid",
                         "searcher.search_relationships_by_vec",
                         "filter.compute_subgraph_intersection"):
            assert required in names, f"{method} never called {required}: {names}"
        assert names[0] == "searcher.embed_query", f"{method} did not embed first: {names[0]}"


def test_the_reranker_does_the_filtering() -> None:
    """Stage 3 passes everything through; the reranker is what cuts.

    Replaces a test that compared the five dispatch paths against each other.
    With one path left, what is worth pinning is that stage 3 does no cutting
    of its own -- if it started to, the reranker would be scoring a
    pre-narrowed pool without anyone noticing."""
    calls = [c for c in _capture()["calls"] if c["call"].startswith("filter.")]
    assert [c["call"] for c in calls] == [
        "filter.compute_subgraph_intersection",
        "filter.rerank_filter",
    ], f"stage 3 filtered before the reranker: {[c[chr(39)+chr(39)] for c in calls]}"


def test_an_empty_candidate_pool_short_circuits_the_whole_query() -> None:
    """When nothing is reachable, retrieval returns empty rather than continuing.

    This path is not in the snapshots -- the fixture always finds something --
    and it was nearly lost when stage 1 became its own method: the bare `return`
    that used to end the whole query would have ended only the stage. mypy
    caught it because the return types disagreed. This test catches it whether
    or not the types happen to.
    """
    r = _retriever()

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


def test_agent_filter_closes_vdb_when_switching_question_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LongMem replay keeps at most one summaries client per worker thread."""
    from experiment.agent_filter import vector_search
    from grace_mem.adapters.vector_store import chroma_vdb

    clients = []

    class FakeSummariesVDB:
        def __init__(self, dim: int, path: str, collection_name: str) -> None:
            self.dim = dim
            self.path = path
            self.collection_name = collection_name
            self.closed = False
            clients.append(self)

        def close(self) -> None:
            self.closed = True

    vector_search.close_vector_search_vdb()
    monkeypatch.setattr(chroma_vdb, "SummariesVDB", FakeSummariesVDB)

    first = vector_search._get_vdb(tmp_path / "question-1")
    assert vector_search._get_vdb(tmp_path / "question-1") is first
    second = vector_search._get_vdb(tmp_path / "question-2")

    assert first.closed is True
    assert second.closed is False
    assert len(clients) == 2
    vector_search.close_vector_search_vdb()
    assert second.closed is True
