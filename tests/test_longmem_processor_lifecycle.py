from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiment.longmem.pipeline.processor import MultiDatasetProcessor


def _processor_without_init() -> MultiDatasetProcessor:
    processor = object.__new__(MultiDatasetProcessor)
    processor.current_mgr = None
    processor.current_ingestor = None
    processor.current_retriever = None
    processor.current_ent = None
    processor.current_rel = None
    processor.graph = Mock()
    processor.llm = Mock()
    processor._closed = False
    processor._logger_bindings = []
    return processor


def test_processor_close_releases_shared_runtime_once():
    processor = _processor_without_init()
    manager = Mock()
    graph = processor.graph
    llm = processor.llm
    processor.current_mgr = manager

    processor.close()
    processor.close()

    manager.close.assert_called_once_with()
    graph.close.assert_called_once_with()
    llm.close.assert_called_once_with()
    assert processor.current_mgr is None
    assert processor.graph is None
    assert processor.llm is None


def test_processor_close_releases_remaining_resources_after_failure():
    processor = _processor_without_init()
    manager = Mock()
    graph = processor.graph
    llm = processor.llm
    manager.close.side_effect = RuntimeError("manager close failed")
    processor.current_mgr = manager

    with pytest.raises(RuntimeError, match="manager close failed"):
        processor.close()

    graph.close.assert_called_once_with()
    llm.close.assert_called_once_with()
    assert processor._closed is True


def test_dataset_setup_failure_still_runs_teardown():
    processor = _processor_without_init()
    processor._load_checkpoint = Mock(return_value={})
    processor._output_path = Mock(return_value=Path("output.csv"))
    processor._setup_dataset = Mock(side_effect=RuntimeError("setup failed"))
    processor._teardown_dataset = Mock()
    config = SimpleNamespace(name="dataset", csv_path="dataset.csv")

    with pytest.raises(RuntimeError, match="setup failed"):
        processor.process_dataset(config)

    processor._teardown_dataset.assert_called_once_with(config)


def test_dataset_logger_bindings_are_restored_in_reverse_order():
    processor = _processor_without_init()
    first_original = Mock(name="first_original")
    second_original = Mock(name="second_original")
    first_module = SimpleNamespace(_jlog=first_original)
    second_module = SimpleNamespace(_jlog=second_original)

    processor._bind_module_logger(first_module, Mock(name="first_dataset"))
    processor._bind_module_logger(second_module, Mock(name="second_dataset"))
    processor._restore_module_loggers()

    assert first_module._jlog is first_original
    assert second_module._jlog is second_original
    assert processor._logger_bindings == []
