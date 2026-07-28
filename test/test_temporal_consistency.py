from __future__ import annotations

from datetime import datetime
import importlib.util
import sys
import types
from pathlib import Path

from KG.utils.query_time_parser import detect_and_parse_time_expressions
from KG.utils.temporal import build_time_context, rewrite_temporal_text


def _load_longmem_qa_eval_stage():
    if "pandas" not in sys.modules:
        pandas_stub = types.ModuleType("pandas")
        pandas_stub.DataFrame = object
        pandas_stub.notna = lambda value: value is not None
        sys.modules["pandas"] = pandas_stub

    module_path = Path(__file__).resolve().parents[1] / "experiment" / "longmem" / "stages" / "qa_eval.py"
    spec = importlib.util.spec_from_file_location("longmem_qa_eval_for_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.QAEvalStage


def test_shared_core_wrapper_and_longmem_rewrite_resolve_identically():
    reference = "2023/04/12 (Wed) 12:00"
    query = "What happened before last Friday?"
    context = build_time_context(
        reference_dt=datetime(2023, 4, 12, 12, 0, 0),
        reference_time_str=reference,
        source="test",
    )

    shared_rewritten, metadata = rewrite_temporal_text(query, context)
    wrapper_rewritten, wrapper_info = detect_and_parse_time_expressions(query, query_time=reference, rewrite_query=True)
    qaeval_cls = _load_longmem_qa_eval_stage()
    longmem_rewritten = qaeval_cls().rewrite_temporal_question(query, query_time=reference)

    assert shared_rewritten == "What happened before 2023-04-07?"
    assert wrapper_rewritten == shared_rewritten
    assert longmem_rewritten == shared_rewritten
    assert metadata["constraints"][0]["resolution"]["display_value"] == "before 2023-04-07"
    assert wrapper_info["constraints"][0]["resolution"]["display_value"] == "before 2023-04-07"
