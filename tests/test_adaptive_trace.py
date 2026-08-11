from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from KG.pipeline.retriever import Retriever


def _retriever_without_init() -> Retriever:
    return object.__new__(Retriever)


def test_overlap_metrics_use_jaccard_on_unique_ids():
    overlap_count, overlap_pct = Retriever._compute_overlap_metrics(
        ["e1", "e1", "e2"],
        ["e2", "e3", "e3"],
    )

    assert overlap_count == 1
    assert overlap_pct == 1 / 3


def test_overlap_metrics_handle_empty_set_edge_cases():
    both_empty_count, both_empty_pct = Retriever._compute_overlap_metrics([], [])
    pass1_only_count, pass1_only_pct = Retriever._compute_overlap_metrics(["e1"], [])
    pass2_only_count, pass2_only_pct = Retriever._compute_overlap_metrics([], ["e2"])

    assert both_empty_count == 0
    assert both_empty_pct is None
    assert pass1_only_count == 0
    assert pass1_only_pct == 0.0
    assert pass2_only_count == 0
    assert pass2_only_pct == 0.0


def test_non_triggered_trace_uses_empty_pass2_ids_and_null_overlap_pct():
    retriever = _retriever_without_init()

    trace = Retriever._build_adaptive_trace(
        retriever,
        pass2_triggered=False,
        pass1_entity_ids=["e1", "e2"],
        pass1_relation_ids=["r1"],
        pass2_entity_ids=["should-be-dropped"],
        pass2_relation_ids=["should-be-dropped"],
        conf_pass1=0.82,
        conf_final=0.82,
    )

    assert trace["pass2_triggered"] is False
    assert trace["pass1_entity_ids"] == ["e1", "e2"]
    assert trace["pass1_relation_ids"] == ["r1"]
    assert trace["pass2_entity_ids"] == []
    assert trace["pass2_relation_ids"] == []
    assert trace["entity_overlap_count"] == 0
    assert trace["relation_overlap_count"] == 0
    assert trace["entity_overlap_pct"] is None
    assert trace["relation_overlap_pct"] is None


def test_triggered_trace_computes_overlap_from_pre_merge_pass_sets():
    retriever = _retriever_without_init()

    trace = Retriever._build_adaptive_trace(
        retriever,
        pass2_triggered=True,
        pass1_entity_ids=["e1", "e2"],
        pass1_relation_ids=["r1", "r2"],
        pass2_entity_ids=["e2", "e3"],
        pass2_relation_ids=[],
        conf_pass1=0.31,
        conf_pass2=0.44,
        conf_final=0.52,
    )

    assert trace["pass2_triggered"] is True
    assert trace["entity_overlap_count"] == 1
    assert trace["entity_overlap_pct"] == 1 / 3
    assert trace["relation_overlap_count"] == 0
    assert trace["relation_overlap_pct"] == 0.0


def test_adaptive_research_closes_temporary_graph_when_pass2_fails(monkeypatch):
    from KG.pipeline.retrieval_steps import adaptive

    temporary_graph = Mock()
    monkeypatch.setattr(adaptive, "compute_confidence", Mock(return_value=0.1))
    monkeypatch.setattr(adaptive, "build_adaptive_llm_client", Mock(return_value=object()))
    monkeypatch.setattr(adaptive, "rewrite_query", Mock(return_value=("rewritten", 0.01)))
    monkeypatch.setattr(adaptive, "build_adaptive_graph", Mock(return_value=temporary_graph))

    retriever = _retriever_without_init()
    retriever.cfg = SimpleNamespace(
        tau_confidence=0.5,
        adaptive_threshold_scale=0.8,
    )
    retriever.MGR = object()
    retriever.generate_query_keywords = Mock(
        return_value=SimpleNamespace(low_level_keywords=[], high_level_keywords=[])
    )
    retriever.assemble_context_from_query = Mock(
        side_effect=RuntimeError("pass2 retrieval failed")
    )

    with pytest.raises(RuntimeError, match="pass2 retrieval failed"):
        retriever._adaptive_research(
            question="original",
            ctx_entities=[{"id": "e1"}],
            ctx_rels=[{"rel_id": "r1"}],
            ctx_text="context",
            query_vec=object(),
            request_id="request-1",
            ent_topk=5,
            rel_topk=5,
            ent_threshold=0.5,
            rel_threshold=0.5,
            filter_ent_topk=5,
            filter_rel_topk=5,
            filter_ent_threshold=0.5,
            filter_rel_threshold=0.5,
            query_time=None,
        )

    temporary_graph.close.assert_called_once_with()
