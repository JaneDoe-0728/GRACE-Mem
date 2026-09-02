"""The command protocol the agent speaks, pinned shape by shape.

Local models emit their one command in whichever channel their adapter
prefers -- plain content, Harmony markup, or native tool calls -- and a parse
miss reads downstream as "the agent selected nothing". These cases are the
formats that were observed in real runs, so they are the ones a refactor of the
parser has to keep answering identically.
"""

from __future__ import annotations

from agent_filter_fakes import response, tool_call

from experiment.agent_filter.models import Command
from experiment.agent_filter.protocol import (
    parse_command,
    parse_response,
    response_command_candidates,
    response_diagnostics,
)


def test_plain_commands_parse_with_their_arguments() -> None:
    assert parse_command("GREP marathon") == Command("GREP", "marathon")
    assert parse_command("READ s1:1:u 3") == Command("READ", "s1:1:u 3")
    assert parse_command("VECTOR what shoes do i own") == Command(
        "VECTOR", "what shoes do i own"
    )
    assert parse_command("FINAL s1:1:u s1:2:u") == Command("FINAL", "s1:1:u s1:2:u")


def test_read_without_a_window_size_defaults_to_two() -> None:
    assert parse_command("READ s1:1:u") == Command("READ", "s1:1:u 2")


def test_quoted_arguments_are_unquoted() -> None:
    assert parse_command('GREP "marathon"') == Command("GREP", "marathon")


def test_reasoning_before_the_command_is_ignored() -> None:
    reply = "Let me look for the marathon mention first.\nGREP marathon"

    assert parse_command(reply) == Command("GREP", "marathon")


def test_an_empty_final_yields_to_a_runnable_command_in_the_same_reply() -> None:
    assert parse_command("GREP marathon\nFINAL") == Command("GREP", "marathon")


def test_commands_crammed_onto_one_line_still_end_at_final() -> None:
    # 120B dumps its whole plan on one line; without the split the trailing FINAL
    # is swallowed into the READ argument.
    assert parse_command("READ s1:2:u 5FINAL s1:2:u") == Command("FINAL", "s1:2:u")


def test_harmony_tool_syntax_parses_into_a_command() -> None:
    reply = '<|channel|>commentary to=READ <|constrain|>json<|message|>{"id": "s1:2:u", "k": 3}'

    assert parse_command(reply) == Command("READ", "s1:2:u 3")


def test_harmony_without_a_json_payload_falls_back_to_the_loose_parse() -> None:
    assert parse_command('to=tool.GREP <|constrain|>="marathon"') == Command(
        "GREP", "marathon"
    )


def test_unparseable_text_yields_no_command() -> None:
    assert parse_command("I am not sure which turns to pick.") is None


def test_native_tool_calls_are_normalized_into_command_text() -> None:
    resp = response(None, tool_calls=[tool_call("functions.GREP", '{"pattern": "marathon"}')])

    candidates = response_command_candidates(resp)

    assert [source for source, _ in candidates] == ["tool_calls"]
    assert parse_command(candidates[0][1]) == Command("GREP", "marathon")


def test_the_reasoning_channel_is_the_last_place_a_command_is_looked_for() -> None:
    resp = response("thinking out loud", reasoning="GREP marathon")

    assert [source for source, _ in response_command_candidates(resp)] == [
        "content",
        "reasoning",
    ]


def test_diagnostics_record_an_empty_content_channel() -> None:
    diag = response_diagnostics(response("", reasoning="a long private thought"))

    assert diag["content_empty"] is True
    assert diag["reasoning"] == "a long private thought"
    assert diag["finish_reason"] == "stop"


# ── Reasoning-only replies ───────────────────────────────────────────────────
#
# gpt-oss-20b via LM Studio routinely ends its turn after the analysis channel:
# content and tool_calls come back empty, only reasoning is populated, and
# finish_reason is "stop" -- nothing was truncated, the model narrated its plan
# and stopped. Every one of these strings is a real reasoning field taken from a
# PARSE_FAIL in a recorded run.

def test_a_command_named_in_the_reasoning_is_recovered() -> None:
    resp = response(content="", reasoning="Need to verify candidate. Use READ on answer_530960c1:6:u.")

    parsed = parse_response(resp)

    assert parsed.command == Command("READ", "answer_530960c1:6:u 2")
    assert parsed.source == "reasoning_narrated"


def test_a_grep_pattern_is_taken_from_the_quotes_the_model_used() -> None:
    resp = response(
        content="",
        reasoning='Need to find study in Music and Medicine, number of subjects. '
                  'Use GREP for "Music and Medicine" and "binaural beats".',
    )

    assert parse_response(resp).command == Command("GREP", "Music and Medicine")


def test_the_pattern_may_be_quoted_in_a_later_sentence_than_the_command() -> None:
    resp = response(
        content="",
        reasoning='Need to grep for study in Music and Medicine. '
                  'Use regex "Music and Medicine" and maybe "binaural beats".',
    )

    assert parse_response(resp).command == Command("GREP", "Music and Medicine")


def test_narration_that_names_no_command_stays_a_parse_failure() -> None:
    # "Search." is an intention, not a tool call. Inferring GREP from an English
    # verb would spend a call from the budget on a guess.
    resp = response(content="", reasoning="Need study in Music and Medicine; number of subjects. Search.")

    assert parse_response(resp).command is None


def test_a_command_named_without_its_argument_stays_a_parse_failure() -> None:
    resp = response(content="", reasoning='Need wait time: user said "Over a year". Verify with READ.')

    assert parse_response(resp).command is None


def test_a_well_formed_reply_never_reaches_the_narration_fallback() -> None:
    # The fallback runs only after every ordinary surface has failed, so a reply
    # whose content parses is unaffected by what its reasoning happens to say.
    resp = response(content="GREP marathon", reasoning="I should probably READ s1:1:u instead.")

    parsed = parse_response(resp)

    assert parsed.command == Command("GREP", "marathon")
    assert parsed.source == "content"
