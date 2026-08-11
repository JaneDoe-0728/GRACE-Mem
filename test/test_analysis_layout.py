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


def test_legacy_analysis_modules_delegate_private_helpers() -> None:
    pairs = (
        ("locomo_gold_recall_metrics", "experiment.locomo.analysis.gold_recall", "_sample_index"),
        ("gold_recall_metrics", "experiment.longmem.analysis.gold_recall", "_is_correct"),
        ("experiment.locomo.helpers.diff_sample_flips", "experiment.locomo.analysis.flips", "_compute_flips"),
        ("experiment.longmem.summary_score_dist", "experiment.longmem.analysis.summary_scores", "_stats"),
    )
    for legacy_name, canonical_name, attribute in pairs:
        before = list(sys.path)
        legacy = importlib.import_module(legacy_name)
        canonical = importlib.import_module(canonical_name)
        assert getattr(legacy, attribute) is getattr(canonical, attribute)
        assert sys.path == before


def test_fact_replay_legacy_entrypoints_keep_role_defaults(monkeypatch) -> None:
    all_roles = importlib.import_module("experiment.longmem.replay_fact_multi_dataset")
    user_only = importlib.import_module("experiment.longmem.replay_fact_user_only")
    calls: list[tuple[list[str] | None, str]] = []

    def fake_main(argv=None, *, default_source_roles="all"):
        calls.append((argv, default_source_roles))

    monkeypatch.setattr("experiment.longmem.analysis.fact_replay.main", fake_main)
    all_roles.main(["--dry-run"])
    user_only.main(["--dry-run"])

    assert calls == [(["--dry-run"], "all"), (["--dry-run"], "user")]
