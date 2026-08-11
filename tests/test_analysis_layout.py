from __future__ import annotations

import importlib
import sys

import pandas as pd

from experiment.common.recall import RecallStats, format_ratio
from experiment.longmem.analysis.fact_replay import _build_session_raw_texts


def test_recall_stats_accumulate_shared_metrics() -> None:
    stats = RecallStats()
    stats.add_accuracy(correct=True)
    stats.add_retrieval(gold={"a", "b"}, retrieved={"a", "b", "c"}, correct=True)
    stats.add_accuracy(correct=False)
    stats.add_retrieval(gold={"d"}, retrieved=set(), correct=False)

    assert stats == RecallStats(
        questions=2,
        correct=1,
        gold_total=3,
        gold_hit=2,
        questions_with_gold=2,
        all_gold_hit=1,
        all_gold_hit_correct=1,
    )
    assert format_ratio(1, 2) == "1/2 = 50.0%"
    assert format_ratio(0, 0) == "0/0 = n/a"


def test_fact_replay_role_modes_share_one_loader(tmp_path) -> None:
    source = tmp_path / "question.csv"
    pd.DataFrame(
        [
            {"session_id": "s1", "turn_index": 0, "role": "user", "content": "User fact"},
            {"session_id": "s1", "turn_index": 1, "role": "assistant", "content": "Reply"},
        ]
    ).to_csv(source, index=False)

    assert _build_session_raw_texts(source, source_roles="all") == {
        "s1": "User: User fact\nAssistant: Reply"
    }
    assert _build_session_raw_texts(source, source_roles="user") == {"s1": "User: User fact"}


def test_canonical_analysis_modules_import_without_sys_path_changes() -> None:
    modules = (
        "experiment.locomo.analysis.dataset",
        "experiment.locomo.analysis.flips",
        "experiment.locomo.analysis.gold_recall",
        "experiment.locomo.analysis.turn_filter",
        "experiment.longmem.analysis.fact_replay",
        "experiment.longmem.analysis.gold_recall",
        "experiment.longmem.analysis.judge_flips",
        "experiment.longmem.analysis.summary_scores",
        "tools.agent_filter_trace_viewer.build",
        "tools.manual.agent_filter_smoke",
    )
    for module_name in modules:
        before = list(sys.path)
        importlib.import_module(module_name)
        assert sys.path == before
