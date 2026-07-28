"""
Import smoke tests for the benchmark-neutral analysis adapter layer.

Run with:

    python -m pytest test/test_benchmark_analysis_imports.py -v
"""

from __future__ import annotations

from pathlib import Path

from experiment.analysis.base import BenchmarkCapabilities, BenchmarkUnit
from experiment.analysis.base import BenchmarkAdapter
from experiment.analysis.adapters.locomo import LoCoMoAdapter
from experiment.analysis.adapters.longmem import LongMemAdapter
from experiment.analysis.ingestion_analysis import _resolve_ingestion_log_dir
from experiment.analysis.registry import REGISTRY, get_adapter


def test_registry_contains_expected_benchmarks():
    assert REGISTRY["locomo"] is LoCoMoAdapter
    assert REGISTRY["longmem"] is LongMemAdapter


def test_get_adapter_instantiates_locomo():
    adapter = get_adapter("locomo")

    assert isinstance(adapter, BenchmarkAdapter)
    assert isinstance(adapter, LoCoMoAdapter)
    assert adapter.benchmark_name == "locomo"


def test_get_adapter_instantiates_longmem_with_paths():
    adapter = get_adapter(
        "longmem",
        script_data_dir=Path("experiment/longmem/script_data"),
        artifact_dir=Path("experiment/longmem"),
    )

    assert isinstance(adapter, BenchmarkAdapter)
    assert isinstance(adapter, LongMemAdapter)
    assert adapter.benchmark_name == "longmem"


def test_longmem_capabilities_accept_ingestion_logs_in_source_logs_dir(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    logs_dir = run_root / "logs_ds1"
    logs_dir.mkdir()
    (logs_dir / "kg_retriever.jsonl").write_text("", encoding="utf-8")

    source_root = tmp_path / "source"
    source_root.mkdir()
    category_dir = source_root / "knowledge_update"
    category_dir.mkdir()
    artifacts_dir = category_dir / "artifacts_ds1"
    artifacts_dir.mkdir()
    (artifacts_dir / "entities_meta.jsonl").write_text("", encoding="utf-8")
    (artifacts_dir / "summaries_meta.jsonl").write_text("", encoding="utf-8")
    source_logs_dir = category_dir / "logs_ds1"
    source_logs_dir.mkdir()
    (source_logs_dir / "kg_ingestor.jsonl").write_text("", encoding="utf-8")

    adapter = LongMemAdapter(script_data_dir=tmp_path, artifact_dir=source_root)
    unit = BenchmarkUnit(
        unit_id="ds1",
        category="Knowledge Update",
        base_dir=category_dir,
        eval_csv_path=None,
        logs_dir=logs_dir,
        artifacts_dir=artifacts_dir,
        checkpoint_path=None,
    )

    caps = adapter.capabilities_for_unit(unit)

    assert isinstance(caps, BenchmarkCapabilities)
    assert caps.has_ingestion_logs is True
    assert caps.supports_ingestion_analysis is True


def test_ingestion_analysis_falls_back_to_artifacts_dir_for_ingestor_logs(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    logs_dir = run_root / "logs_ds1"
    logs_dir.mkdir()

    source_root = tmp_path / "source"
    source_root.mkdir()
    artifacts_dir = source_root / "artifacts_ds1"
    artifacts_dir.mkdir()
    source_logs_dir = source_root / "logs_ds1"
    source_logs_dir.mkdir()
    (source_logs_dir / "kg_ingestor.jsonl").write_text("", encoding="utf-8")

    assert _resolve_ingestion_log_dir(logs_dir, artifacts_dir) == source_logs_dir
