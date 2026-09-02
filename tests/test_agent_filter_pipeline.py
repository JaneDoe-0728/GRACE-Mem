"""End-to-end behaviour of one Agent Filter refinement.

Every path a question can take through the harness -- a clean FINAL, a forced
one, each fallback, and the adjudication recovery layer -- is driven here with a
scripted model. These are the guarantees callers already depend on: the context
is never made worse, and the trace explains what happened.
"""

from __future__ import annotations

from agent_filter_fakes import (
    CONTEXT,
    CSV_PATH,
    SEED,
    ExplodingLLM,
    ScriptedLLM,
    corpus,
)

from experiment.agent_filter.harness import refine_context

QUESTION = "What did I do in April?"


def refine(replies, *, context=CONTEXT, question=QUESTION, params=None, **kwargs):
    return refine_context(
        question=question,
        context=context,
        csv_path=CSV_PATH,
        llm=replies if hasattr(replies, "chat") else ScriptedLLM(replies),
        corpus=corpus(),
        params=params or {},
        **kwargs,
    )


# ── The mainline ─────────────────────────────────────────────────────────────

def test_a_final_selection_rebuilds_the_context_from_the_raw_turns() -> None:
    refined, trace = refine(["GREP marathon", "FINAL s1:1:u"])

    assert trace["fallback"] is None
    assert trace["final_sids"] == ["s1:1:u"]
    assert trace["kept"] == ["s1:1:u"]
    assert trace["dropped"] == ["s1:2:u"]
    assert [c["cmd"] for c in trace["commands"]] == ["GREP", "FINAL"]
    assert "Congratulations on the marathon" in refined
    assert "new running shoes" not in refined


def test_turns_returned_by_grep_are_recorded_as_verified() -> None:
    _, trace = refine(["GREP marathon", "FINAL s1:1:u"])

    assert sorted(trace["verified_sids"]) == ["s1:1:a", "s1:1:u"]
    assert trace["evidence_provenance"] == {"s1:1:u": "seed+verified"}


def test_the_selection_is_capped_at_the_evidence_limit() -> None:
    _, trace = refine(["FINAL s1:1:u s1:2:u"], params={"grep_agent_max_sids": 1})

    assert trace["final_before_cap"] == ["s1:1:u", "s1:2:u"]
    assert trace["final_sids"] == ["s1:1:u"]


def test_the_agent_reports_its_own_answer_hypothesis_when_asked_to() -> None:
    _, trace = refine(
        ["HYPOTHESIS: a marathon\nFINAL s1:1:u"],
        params={"grep_agent_emit_hypothesis": 1},
    )

    assert trace["hypothesis"] == "a marathon"


# ── Modes ────────────────────────────────────────────────────────────────────

def test_filter_mode_discards_sids_the_agent_added() -> None:
    _, trace = refine(["FINAL s1:1:u s2:1:u"], params={"grep_agent_mode": "filter"})

    assert trace["final_sids"] == ["s1:1:u"]


def test_filter_mode_without_a_seed_never_runs_the_agent() -> None:
    refined, trace = refine([], context="no evidence summary here",
                            params={"grep_agent_mode": "filter"})

    assert trace["fallback"] == "no_seed"
    assert refined == "no evidence summary here"
    assert trace["commands"] == []


def test_fetch_only_mode_appends_to_the_original_context() -> None:
    refined, trace = refine(["FINAL s2:1:u"], params={"grep_agent_mode": "fetch_only"})

    assert trace["added"] == ["s2:1:u"]
    assert refined.startswith(CONTEXT.rstrip("\n"))
    assert "### Additional Evidence (agent-retrieved)" in refined
    assert "My cat is named Melanie" in refined


def test_fetch_only_mode_falls_back_when_the_agent_adds_nothing() -> None:
    refined, trace = refine(["FINAL s1:1:u"], params={"grep_agent_mode": "fetch_only"})

    assert trace["fallback"] == "no_addition"
    assert refined == CONTEXT


# ── Closing the loop when the agent will not ─────────────────────────────────

def test_a_third_unparseable_reply_ends_the_loop_and_forces_a_final() -> None:
    # Three, not two: a reasoning model that narrates its intent instead of
    # emitting the command needs more than one corrective hint to recover.
    _, trace = refine(["nonsense", "more nonsense", "still nonsense", "FINAL s1:1:u"])

    assert [c["cmd"] for c in trace["commands"]] == [
        "PARSE_FAIL", "PARSE_FAIL", "PARSE_FAIL", "FINAL(forced)",
    ]
    assert trace["final_sids"] == ["s1:1:u"]


def test_a_second_unparseable_reply_gets_another_hint_rather_than_giving_up() -> None:
    _, trace = refine(["nonsense", "more nonsense", "GREP marathon", "FINAL s1:1:u"])

    assert [c["cmd"] for c in trace["commands"]] == [
        "PARSE_FAIL", "PARSE_FAIL", "GREP", "FINAL",
    ]
    assert trace["final_sids"] == ["s1:1:u"]


def test_parse_failures_have_to_be_consecutive_to_end_the_loop() -> None:
    """_MAX_PARSE_FAILURES counts consecutive failures, as its comment says.

    The counter was never reset, so it was cumulative: an agent that failed to
    parse on turns 1, 4 and 7 but issued valid commands in between was abandoned
    at turn 7 with the rest of its budget unspent -- even though the corrective
    hint had demonstrably worked, twice.
    """
    _, trace = refine(
        ["nonsense", "GREP marathon", "more nonsense", "READ s1:1:u 1",
         "still nonsense", "FINAL s1:1:u"],
        params={"grep_agent_max_calls": 10},
    )

    assert [c["cmd"] for c in trace["commands"]] == [
        "PARSE_FAIL", "GREP", "PARSE_FAIL", "READ", "PARSE_FAIL", "FINAL",
    ]
    assert trace["final_sids"] == ["s1:1:u"]


def test_a_forced_final_that_still_fails_keeps_the_original_context() -> None:
    refined, trace = refine(["nonsense", "more nonsense", "still nonsense", "no sids here"])

    assert trace["fallback"] == "no_final"
    assert refined == CONTEXT


def test_repeating_one_command_breaks_the_loop() -> None:
    _, trace = refine(["GREP marathon"] * 3)

    assert [c["cmd"] for c in trace["commands"]] == [
        "GREP", "REPEAT_BREAK", "FINAL(forced)",
    ]
    assert trace["fallback"] == "no_final"


def test_an_exhausted_call_budget_forces_a_final() -> None:
    _, trace = refine(["GREP marathon", "FINAL s1:1:u"],
                      params={"grep_agent_max_calls": 1})

    assert [c["cmd"] for c in trace["commands"]] == ["GREP", "FINAL(forced)"]
    assert trace["final_sids"] == ["s1:1:u"]


def test_a_no_final_run_can_ask_the_answering_model_to_abstain() -> None:
    refined, trace = refine(["no sids here", "still none", "nothing"],
                            params={"grep_agent_abstention_hint": 1})

    assert trace["abstention_hint"] is True
    assert refined.startswith(CONTEXT)
    assert "not available in the conversation history" in refined


def test_a_run_that_keeps_no_seed_falls_back_to_the_original_context() -> None:
    refined, trace = refine(["FINAL s2:1:u"])

    assert trace["fallback"] == "zero_keep"
    assert refined == CONTEXT
    assert trace["context_sids"] == SEED


def test_a_broken_endpoint_falls_back_to_the_original_context() -> None:
    refined, trace = refine(ExplodingLLM())

    assert trace["fallback"] == "exception"
    assert "endpoint down" in trace["error"]
    assert refined == CONTEXT


# ── Recovery layers on top of the agent's selection ──────────────────────────

def test_adjudication_adds_back_a_discarded_seed_it_rules_relevant() -> None:
    _, trace = refine(
        ["FINAL s1:1:u", "s1:2:u KEEP still about the same running story"],
        params={"grep_agent_adjudicate": 1},
    )

    assert trace["adjudication"]["kept"] == ["s1:2:u"]
    assert trace["adjudication"]["reasons"]["s1:2:u"].startswith("KEEP: still about")
    assert trace["final_sids"] == ["s1:1:u", "s1:2:u"]


def test_adjudication_drops_a_seed_it_rules_unrelated() -> None:
    _, trace = refine(
        ["FINAL s1:1:u", "s1:2:u DROP unrelated to April"],
        params={"grep_agent_adjudicate": 1},
    )

    assert trace["adjudication"]["dropped"] == ["s1:2:u"]
    assert trace["final_sids"] == ["s1:1:u"]


def test_adjudication_is_skipped_outside_its_categories() -> None:
    _, trace = refine(["FINAL s1:1:u"], params={
        "grep_agent_adjudicate": 1,
        "grep_agent_adjudicate_categories": ("temporal_reasoning",),
    })

    assert "adjudication" not in trace


# ── The VECTOR tool ──────────────────────────────────────────────────────────

def test_a_vector_hit_counts_as_verified_evidence(monkeypatch, tmp_path) -> None:
    # The prompt calls VECTOR results "leads, not verified evidence", while the
    # runtime trusts them outright. The runtime behaviour is what ships, so it is
    # what is pinned; reconciling the two is a behaviour change, not a refactor.
    from experiment.agent_filter import vector_search

    (tmp_path / "summaries_chroma").mkdir()
    monkeypatch.setattr(
        vector_search, "search_summaries",
        lambda *a, **k: [("s2:1:u", 0.51)],
    )

    _, trace = refine(
        ["VECTOR who is melanie", "FINAL s1:1:u s2:1:u"],
        artifact_dir=tmp_path,
    )

    assert trace["vector_tool"] is True
    assert trace["vector_candidate_sids"] == ["s2:1:u"]
    assert trace["evidence_provenance"] == {"s1:1:u": "seed", "s2:1:u": "verified"}


def test_vector_is_offered_only_when_the_question_has_a_summary_store() -> None:
    llm = ScriptedLLM(["VECTOR who is melanie", "FINAL s1:1:u"])

    _, trace = refine(llm)

    assert trace["vector_tool"] is False
    assert "VECTOR is not available" in llm.prompts[-1][-1]["content"]


def test_a_prebuilt_corpus_needs_neither_a_csv_nor_a_category() -> None:
    # How LoCoMo replay calls in: its corpus is chunk-level and built in memory,
    # and it has no LongMem category to hint from.
    refined, trace = refine_context(
        question=QUESTION,
        context=CONTEXT,
        csv_path="",
        llm=ScriptedLLM(["FINAL s1:1:u"]),
        category=None,
        corpus=corpus(),
        params={},
    )

    assert trace["fallback"] is None
    assert trace["is_abstention"] is False
    assert trace["final_sids"] == ["s1:1:u"]
    assert "Congratulations on the marathon" in refined


# ── The fallback contract covers reading the params, too ─────────────────────

def test_a_malformed_param_falls_back_instead_of_aborting_the_run() -> None:
    """"Any failure falls back to the original context" has to include parsing
    the params.

    from_params does every coercion -- int() on the vector top-N, float() on the
    minimum score, flag() on the switches. It used to run before the try, so one
    bad value propagated out of refine_context; neither the LongMem runner nor
    rerun guards the call, so a QA run aborted where it should have degraded.
    """
    refined, trace = refine(["FINAL s1:1:u"], params={"grep_agent_vector_topn": "not-a-number"})

    assert trace["fallback"] == "exception"
    assert refined == CONTEXT


def test_a_retired_param_under_error_warnings_falls_back_too() -> None:
    """The same guarantee for the FutureWarning that a legacy key raises: under
    -W error it is an exception like any other, and it is raised while reading the
    params."""
    import warnings

    from experiment.agent_filter import config as config_module

    config_module._warned.discard("grep_agent_require_verified_additions")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        refined, trace = refine(
            ["FINAL s1:1:u"],
            params={"grep_agent_require_verified_additions": 1},
        )

    assert trace["fallback"] == "exception"
    assert refined == CONTEXT
