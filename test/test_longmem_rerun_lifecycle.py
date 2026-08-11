from unittest.mock import Mock

import pytest

from experiment.longmem.rerun import LongMemRerun


def test_rerun_close_releases_shared_runtime_once():
    graph = Mock()
    llm = Mock()
    runner = LongMemRerun(llm=llm, graph=graph)

    runner.close()
    runner.close()

    graph.close.assert_called_once_with()
    llm.close.assert_called_once_with()


def test_rerun_close_releases_llm_after_graph_failure():
    graph = Mock()
    llm = Mock()
    graph.close.side_effect = RuntimeError("graph close failed")
    runner = LongMemRerun(llm=llm, graph=graph)

    with pytest.raises(RuntimeError, match="graph close failed"):
        runner.close()

    llm.close.assert_called_once_with()
    assert runner._closed is True


def test_rerun_context_manager_closes_runtime():
    graph = Mock()
    llm = Mock()

    with LongMemRerun(llm=llm, graph=graph):
        pass

    graph.close.assert_called_once_with()
    llm.close.assert_called_once_with()


def test_rerun_from_env_rolls_back_when_graph_open_fails(monkeypatch):
    import KG.graph.falkordb as graph_module
    import KG.llm as llm_module

    graph = Mock()
    llm = Mock()
    graph.open.side_effect = RuntimeError("graph open failed")
    monkeypatch.setattr(graph_module, "graph_from_env", Mock(return_value=graph))
    monkeypatch.setattr(llm_module, "LLMClient", Mock(return_value=llm))

    with pytest.raises(RuntimeError, match="graph open failed"):
        LongMemRerun.from_env()

    graph.close.assert_called_once_with()
    llm.close.assert_called_once_with()
