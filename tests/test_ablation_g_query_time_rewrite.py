from __future__ import annotations

from KG.pipeline import retriever as retriever_module


def test_retrieval_query_time_rewrite_enabled_by_default(monkeypatch):
    monkeypatch.delenv("KG_ABLATION_NO_QUERY_TIME_REWRITE", raising=False)

    rewritten = retriever_module._maybe_rewrite_retrieval_question(
        "What happened last Friday?",
        "2023/04/12 (Wed) 12:00",
        request_id="test-ablation-g-enabled",
    )

    assert rewritten == "What happened on 2023-04-07?"


def test_ablation_g_disables_only_retrieval_query_time_rewrite(monkeypatch):
    question = "What happened last Friday?"
    monkeypatch.setenv("KG_ABLATION_NO_QUERY_TIME_REWRITE", "1")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("retrieval query-time rewrite should be skipped")

    monkeypatch.setattr(retriever_module, "rewrite_temporal_text", fail_if_called)

    rewritten = retriever_module._maybe_rewrite_retrieval_question(
        question,
        "2023/04/12 (Wed) 12:00",
        request_id="test-ablation-g-disabled",
    )

    assert rewritten == question


def test_ablation_g_does_not_disable_query_time_for_containment(monkeypatch):
    monkeypatch.setenv("KG_ABLATION_NO_QUERY_TIME_REWRITE", "1")

    query_dt = retriever_module.parse_query_time("2023/04/12 (Wed) 12:00")

    assert query_dt is not None
    assert query_dt.date().isoformat() == "2023-04-12"
