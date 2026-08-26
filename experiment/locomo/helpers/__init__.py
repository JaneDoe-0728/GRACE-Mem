"""Lazy compatibility facade for LoCoMo helper APIs.

Internal modules import the owning modules directly. This facade preserves the
historical public imports without loading unrelated LLM, aggregation, and graph
dependencies during package initialization.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "llm_post": ("experiment.locomo.helpers.llm", "llm_post"),
    "aggregate_judge_csv_files": (
        "experiment.locomo.analysis.aggregate",
        "aggregate_judge_csv_files",
    ),
    "compute_summary_from_df": (
        "experiment.locomo.analysis.summary",
        "compute_summary_from_df",
    ),
    "compute_summary_from_rows": (
        "experiment.locomo.analysis.summary",
        "compute_summary_from_rows",
    ),
    "ARTIFACTS_SRC": ("experiment.locomo.utils.graph", "ARTIFACTS_SRC"),
    "GRAPH_EXPORT_FILE": ("experiment.locomo.utils.graph", "GRAPH_EXPORT_FILE"),
    "SNAPSHOT_META_FILE": ("experiment.locomo.utils.graph", "SNAPSHOT_META_FILE"),
    "_export_graph": ("experiment.locomo.utils.graph", "_export_graph"),
    "_restore_graph": ("experiment.locomo.utils.graph", "_restore_graph"),
    "export_graph": ("experiment.locomo.utils.graph", "export_graph"),
    "restore_graph_from_export_file": (
        "experiment.locomo.utils.graph",
        "restore_graph_from_export_file",
    ),
    "validate_graph_export": (
        "experiment.locomo.utils.graph",
        "validate_graph_export",
    ),
    "write_graph_export": ("experiment.locomo.utils.graph", "write_graph_export"),
}

for _name in (
    "build_session_records_for_conv",
    "build_session_records_from_json",
    "category_to_label",
    "default_output_stem",
    "default_output_variant_dir",
    "find_evidence_turns_from_sample",
    "index_source_conversations",
    "is_adversarial_category",
    "is_adversarial_item",
    "load_qa_items",
    "load_raw_samples",
    "normalize_dataset_name",
    "normalize_qa_item",
    "resolve_dataset_path",
):
    _EXPORTS[_name] = ("experiment.locomo.helpers.dataset", _name)

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
