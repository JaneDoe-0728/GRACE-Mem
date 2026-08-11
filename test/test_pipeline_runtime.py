import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from KG.pipeline.factory import PipelineRuntime, build_pipeline


def _runtime() -> PipelineRuntime:
    return PipelineRuntime(
        retriever=object(),
        ingestor=object(),
        graph=Mock(),
        mgr=object(),
    )


def test_pipeline_runtime_preserves_legacy_mapping_contract():
    runtime = _runtime()

    assert tuple(runtime) == ("retriever", "ingestor", "graph", "mgr")
    assert runtime["retriever"] is runtime.retriever
    assert runtime["ingestor"] is runtime.ingestor
    assert runtime["graph"] is runtime.graph
    assert runtime["mgr"] is runtime.mgr
    assert dict(runtime)["mgr"] is runtime.mgr
    with pytest.raises(KeyError):
        runtime["unknown"]


def test_pipeline_runtime_context_manager_closes_graph_once():
    runtime = _runtime()

    with runtime as active:
        assert active is runtime

    runtime.close()
    runtime.graph.close.assert_called_once_with()


def test_pipeline_runtime_closes_graph_when_context_raises():
    runtime = _runtime()

    with pytest.raises(RuntimeError, match="pipeline failed"):
        with runtime:
            raise RuntimeError("pipeline failed")

    runtime.graph.close.assert_called_once_with()


def test_build_pipeline_closes_graph_when_construction_fails(monkeypatch):
    graph = Mock()
    graph_from_env = Mock(return_value=SimpleNamespace(open=Mock(return_value=graph)))

    def module(name: str, **attributes):
        stub = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(stub, key, value)
        return stub

    class FailingEntityManager:
        def __init__(self, **kwargs):
            raise RuntimeError("service construction failed")

    cache = {
        "entities": {},
        "entities_full": {},
        "relationships": {},
        "relationships_full": {},
    }
    stubs = {
        "KG.pipeline.retriever": module("KG.pipeline.retriever", Retriever=object),
        "KG.pipeline.ingestor": module("KG.pipeline.ingestor", Ingestor=object),
        "KG.storage": module("KG.storage", MGR=SimpleNamespace(cache=cache)),
        "KG.llm": module("KG.llm", LLMClient=Mock),
        "KG.graph.falkordb": module("KG.graph.falkordb", graph_from_env=graph_from_env),
        "embeddings": module("embeddings", embedder=SimpleNamespace(embed=Mock())),
        "KG.services": module(
            "KG.services",
            EntityManager=FailingEntityManager,
            RelationshipManager=object,
            Provenance=object(),
        ),
    }
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)

    with pytest.raises(RuntimeError, match="service construction failed"):
        build_pipeline()

    graph.close.assert_called_once_with()
