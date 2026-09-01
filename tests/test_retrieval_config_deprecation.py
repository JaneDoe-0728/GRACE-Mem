"""Retrieval knobs that are still accepted but no longer decide anything.

A sweep over `reranker_topk` writes a different run_metadata.json and produces
an identical result set. That is how a research conclusion goes wrong, so the
config now says which of its own values were decoration.
"""

from __future__ import annotations

import warnings

from grace_mem.retrieval.config import INERT_FIELDS, RetrieverConfig


def test_a_default_config_reports_no_inert_overrides() -> None:
    assert RetrieverConfig().inert_overrides() == {}


def test_setting_a_retired_knob_is_reported_as_ignored() -> None:
    config = RetrieverConfig(reranker_topk=10, filter_ent_topk=15)

    assert config.inert_overrides() == {"reranker_topk": 10, "filter_ent_topk": 15}


def test_a_knob_that_still_decides_something_is_not_reported() -> None:
    config = RetrieverConfig(summary_rerank_topk=16, rrk_ent_topk=25)

    assert config.inert_overrides() == {}


def test_a_retired_knob_warns_the_first_time_it_is_set() -> None:
    # Deduplication is module-global and load-bearing (LoCoMo builds one
    # Retriever per sample), so this test picks a value no other test uses.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RetrieverConfig(reranker_threshold=-99.0)

    assert len(caught) == 1
    assert issubclass(caught[0].category, FutureWarning)
    assert "reranker_threshold" in str(caught[0].message)


def test_every_inert_field_is_a_real_config_field() -> None:
    names = {f for f in RetrieverConfig().__dict__}

    assert set(INERT_FIELDS) <= names
