"""GREP_AGENT_PARAMS is read once, into one dataclass.

The knobs used to be read where they were used, each with its own inline
default. Two things have to stay true after collecting them: an incomplete
mapping still gets the harness's conservative defaults, and the production
mapping is read exactly as written.
"""

from __future__ import annotations

import warnings

import pytest

from experiment.agent_filter.config import AgentFilterConfig
from experiment.experiment_config import GREP_AGENT_PARAMS

# (params key, config field) for every knob the pipeline actually reads.
KNOBS = [
    ("grep_agent_mode", "mode"),
    ("grep_agent_max_calls", "max_calls"),
    ("grep_agent_max_sids", "max_sids"),
    ("grep_agent_grep_max_lines", "grep_max_lines"),
    ("grep_agent_filter_include_graph_context", "filter_include_graph"),
    ("grep_agent_answer_include_graph_context", "answer_include_graph"),
    ("grep_agent_graph_context_max_chars", "graph_context_max_chars"),
    ("grep_agent_include_pair", "include_pair"),
    ("grep_agent_adjudicate", "adjudicate"),
    ("grep_agent_adjudicate_categories", "adjudicate_categories"),
    ("grep_agent_use_skills", "use_skills"),
    ("grep_agent_vector_search", "vector_search"),
    ("grep_agent_vector_topn", "vector_topn"),
    ("grep_agent_vector_min_score", "vector_min_score"),
]


@pytest.mark.parametrize(("key", "field"), KNOBS)
def test_every_production_knob_reaches_its_field(key: str, field: str) -> None:
    config = AgentFilterConfig.from_params(GREP_AGENT_PARAMS)

    expected = GREP_AGENT_PARAMS[key]
    actual = getattr(config, field)

    assert actual == (bool(expected) if isinstance(actual, bool) else expected)


def test_an_incomplete_mapping_keeps_the_conservative_defaults() -> None:
    config = AgentFilterConfig.from_params({"grep_agent_mode": "filter"})

    assert config.mode == "filter"
    assert config.max_calls == 8
    assert config.max_sids == 16
    assert config.adjudicate is False
    assert config.include_pair is True
    assert config.abstention_hint is False


def test_no_params_at_all_is_the_same_as_an_empty_mapping() -> None:
    assert AgentFilterConfig.from_params(None) == AgentFilterConfig.from_params({})


def test_retired_and_unrelated_keys_are_ignored() -> None:
    config = AgentFilterConfig.from_params({
        "use_grep_agent": True,
        "grep_agent_evidence_floor": 12,
        "grep_agent_verify_rounds": 3,
        "grep_agent_min_keep_aggregation": 4,
        "grep_agent_force_verified_final": 1,
        "grep_agent_adjudicate_keep_all_categories": ("multi_session",),
    })

    assert config == AgentFilterConfig.from_params({})


def test_a_knob_that_still_parses_but_decides_nothing_warns() -> None:
    # It is still read without error -- archived sweeps pass it -- but a caller
    # setting it must not be left believing the provenance gate came back.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = AgentFilterConfig.from_params(
            {"grep_agent_require_verified_additions": True}
        )

    assert config == AgentFilterConfig.from_params({})
    assert len(caught) == 1
    assert issubclass(caught[0].category, FutureWarning)
    assert "grep_agent_require_verified_additions" in str(caught[0].message)


def test_a_layer_without_a_category_list_runs_for_every_category() -> None:
    config = AgentFilterConfig.from_params({})

    assert config.applies_to(None, "multi_session") is True
    assert config.applies_to(("multi_session",), "multi_session") is True
    assert config.applies_to(("multi_session",), "knowledge_update") is False
