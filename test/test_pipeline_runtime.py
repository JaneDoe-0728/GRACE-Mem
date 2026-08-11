from unittest.mock import Mock

import pytest

from KG.pipeline.factory import PipelineRuntime


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
