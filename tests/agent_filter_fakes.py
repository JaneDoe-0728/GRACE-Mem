"""Shared doubles for the Agent Filter characterization tests.

The harness talks to an OpenAI-compatible client and to a per-question corpus.
Both are faked here so the tests pin the harness's own behaviour -- command
parsing, the tool loop, and evidence selection -- without a live endpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

from experiment.agent_filter.corpus import Corpus, Turn


def response(
    content: str | None = None,
    *,
    reasoning: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str = "stop",
):
    """One OpenAI-compatible chat completion.

    ``SimpleNamespace`` rather than ``Mock`` because the harness reads
    ``vars(message)`` for its diagnostics, which a Mock cannot answer honestly.
    """
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


def tool_call(name: str, arguments: str):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


class ScriptedLLM:
    """Replies with the scripted turns in order; an exhausted script replies empty."""

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.prompts: list[list[dict]] = []

    def chat(self, *, messages, temperature=0.0, max_tokens=None, **_):
        self.prompts.append([dict(m) for m in messages])
        if not self.replies:
            return response("")
        reply = self.replies.pop(0)
        return reply if hasattr(reply, "choices") else response(reply)


class ExplodingLLM:
    def chat(self, **_):
        raise RuntimeError("endpoint down")


def corpus() -> Corpus:
    """A two-session corpus small enough to reason about turn by turn."""
    return Corpus([
        Turn("s1:1:u", "s1", 0, 0, "user", "2023/03/01", "I ran a marathon in April"),
        Turn("s1:1:a", "s1", 1, 1, "assistant", "2023/03/01", "Congratulations on the marathon"),
        Turn("s1:2:u", "s1", 2, 2, "user", "2023/04/02", "I bought new running shoes"),
        Turn("s1:2:a", "s1", 3, 3, "assistant", "2023/04/02", "Which brand did you pick?"),
        Turn("s2:1:u", "s2", 0, 0, "user", "2023/05/03", "My cat is named Melanie"),
    ])


GRAPH_PREFIX = "=== Entities ===\nMelanie: a cat\n"

CONTEXT = (
    GRAPH_PREFIX
    + "### Evidence Summary\n"
    + "  • [2023/03/01][sid=s1:1:u][score=0.615] User : I ran a marathon in April \n"
    + "  • [2023/04/02][sid=s1:2:u][score=0.412] User : I bought new running shoes \n"
)

SEED = ["s1:1:u", "s1:2:u"]

CSV_PATH = "/data/multi_session/answer_q1.csv"
