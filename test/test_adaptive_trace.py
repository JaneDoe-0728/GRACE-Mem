import sys
import types


def _install_retriever_import_stubs() -> None:
    keyword_module = types.ModuleType("KG.llm.prompts.keyword.extraction")
    keyword_module.keyword_extraction_PROMPT = ""
    sys.modules["KG.llm.prompts.keyword.extraction"] = keyword_module

    utils_module = types.ModuleType("KG.utils.utils")
    utils_module.KeywordExtractionResult = object
    sys.modules["KG.utils.utils"] = utils_module

    logger_module = types.ModuleType("KG.utils.logger_config")

    class _DummyTimer:
        def sec(self) -> float:
            return 0.0

    logger_module._StepTimer = _DummyTimer
    logger_module.make_module_jlog = lambda **_: (lambda *args, **kwargs: None)
    sys.modules["KG.utils.logger_config"] = logger_module

    cache_module = types.ModuleType("KG.storage.cache")
    cache_module.build_id_to_meta_maps = lambda *args, **kwargs: ({}, {})
    sys.modules["KG.storage.cache"] = cache_module

    retrieval_module = types.ModuleType("KG.pipeline.retrieval_steps")

    class _DummyComponent:
        def __init__(self, *args, **kwargs):
            pass

    retrieval_module.EntityRelationshipSearcher = _DummyComponent
    retrieval_module.TemporalRelevanceCalculator = _DummyComponent
    retrieval_module.EvidenceBuilder = _DummyComponent
    retrieval_module.ContextFilter = _DummyComponent
    sys.modules["KG.pipeline.retrieval_steps"] = retrieval_module


_install_retriever_import_stubs()

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
