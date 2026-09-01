"""The agent's search loop, and the tools it runs.

One conversation, one command per turn, until the agent replies FINAL or the
budget runs out. The loop's whole job is to keep a weak local model moving
towards that FINAL: it retains only the compact command in history (raw
reasoning is diagnostic data, and replaying it would inflate every later
request), it re-prompts once on an unparseable reply, and it breaks the
broken-record pattern rather than burning the budget on it.

The tools are separate from the loop on purpose. The loop decides when to
search; AgentTools decides what searching means. Adding a retrieval mode is a
change to AgentTools alone.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from experiment.agent_filter import vector_search
from experiment.agent_filter.corpus import Corpus
from experiment.agent_filter.models import SID_RE, Command
from experiment.agent_filter.protocol import extract_final_sids, parse_response

_PARSE_FAIL_HINT = (
    "Could not parse a command. Reply with exactly one command as the "
    "last line: GREP <regex> | READ <sid> [k] | FINAL <sid> ..."
)
_REPEAT_HINT = (
    "You already ran that exact command. Try DIFFERENT keywords, "
    "or reply FINAL <sid> ... with your current best selection."
)
# The closing reminder lives permanently in the recent context -- after several
# search rounds models routinely forget how to finish. It states outright that a
# partial FINAL is acceptable: 49 of 73 fallbacks were aggregation questions
# burning every round because they "could not gather every instance and dared not
# submit". The adjudication layer fills gaps anyway, so completeness is not the
# goal here.
_CLOSING_REMINDER = (
    "\n\n(When you have identified the evidence, reply with one line: "
    "FINAL <sid> <sid> ... — copy sids exactly. A PARTIAL set is "
    "acceptable: FINAL the turns you have confirmed so far — a separate "
    "audit step recovers anything you miss. Do not keep searching for "
    "completeness.)"
)
_SALVAGE_MESSAGES = (
    ("STOP searching. Reply NOW with only one line listing the selected evidence "
     "sids, copied EXACTLY from this list (or ones you found via GREP):\n{seeds}\n"
     "FINAL <sid> <sid> ..."),
    ("Output ONLY the single line below, filled in with sids from this list — "
     "no other text, no tool calls:\n{seeds}\nFINAL <sid> <sid> ..."),
)

_VECTOR_UNAVAILABLE = "VECTOR is not available for this question; use GREP or READ."


@dataclass(frozen=True)
class ToolResult:
    """What one command produced: the text the agent reads back, and the sids it
    turned up.

    Both sid lists are the tool's own account of provenance, which is why they
    travel with the result rather than being re-derived from its text.
    """
    text: str
    verified: list[str] = field(default_factory=list)
    vector_candidates: list[str] = field(default_factory=list)


class AgentTools:
    """GREP, READ and VECTOR over one question's corpus.

    VECTOR is available only when the question kept its summaries VDB; without
    one the agent is told so rather than being handed an empty result it would
    read as "nothing exists".
    """

    def __init__(
        self,
        corpus: Corpus,
        *,
        seed: list[str],
        artifact_dir=None,
        grep_max_lines: int = 30,
        vector_enabled: bool = False,
        vector_topn: int = 8,
        vector_min_score: float = 0.30,
    ):
        self.corpus = corpus
        self.artifact_dir = artifact_dir
        self.grep_max_lines = grep_max_lines
        self.vector_enabled = vector_enabled
        self.vector_topn = vector_topn
        self.vector_min_score = vector_min_score
        self._vector_exclude = set(corpus.normalize_sids(seed))

    def execute(self, command: Command) -> ToolResult:
        if command.kind == "GREP":
            return self._grep(command.arg)
        if command.kind == "VECTOR":
            return self._vector(command.arg)
        return self._read(command.arg)

    def _grep(self, pattern: str) -> ToolResult:
        text = self.corpus.grep(pattern, max_lines=self.grep_max_lines)
        return ToolResult(text, verified=self._sids_in(text))

    def _read(self, arg: str) -> ToolResult:
        sid, k = arg.rsplit(" ", 1)
        text = self.corpus.read_window(sid, k=int(k))
        return ToolResult(text, verified=self._sids_in(text))

    def _vector(self, query: str) -> ToolResult:
        if not self.vector_enabled:
            return ToolResult(_VECTOR_UNAVAILABLE)
        hits = vector_search.search_summaries(
            self.artifact_dir, query,
            exclude=self._vector_exclude,
            topn=self.vector_topn,
            min_score=self.vector_min_score,
        )
        text = vector_search.render_hits(self.corpus, query, hits)
        found = self._sids_in(text)
        # A VECTOR hit counts as verified outright, on a par with GREP/READ. The
        # provenance gate is gone: whatever is pulled back is trusted, with no
        # second verification required.
        return ToolResult(text, verified=found, vector_candidates=found)

    def _sids_in(self, text: str) -> list[str]:
        return self.corpus.normalize_sids(SID_RE.findall(text))


class AgentSession:
    """One agent conversation, from the opening prompt to a FINAL sid list.

    The session owns the conversation and everything accumulated across it --
    which sids the tools have confirmed, and which came from VECTOR -- so the
    salvage prompts continue the same conversation rather than starting a second
    one.
    """

    def __init__(
        self,
        *,
        llm,
        tools: AgentTools,
        messages: list[dict],
        trace: dict,
        emit_hypothesis: bool = False,
    ):
        self.llm = llm
        self.tools = tools
        self.messages = messages
        self.trace = trace
        self.emit_hypothesis = emit_hypothesis
        self.verified_sids: set[str] = set()
        self.vector_candidate_sids: set[str] = set()

    def tell(self, text: str) -> None:
        """Add one user message -- how a caller reopens a finished conversation."""
        self.messages.append({"role": "user", "content": text})

    def run(self, budget: int) -> list[str] | None:
        """Run one GREP/READ->FINAL tool loop (shared by the main search and the
        verify top-up search). Returns the FINAL sids, or None when the agent
        never closed."""
        parse_failures = 0
        repeat_count = 0
        prev_cmd: Command | None = None
        for _ in range(budget):
            started = time.perf_counter()
            parsed = self._ask(max_tokens=1024)
            cmd = parsed.command
            if cmd is None:
                parse_failures += 1
                self._record("PARSE_FAIL", parsed, started, arg=parsed.raw_reply[:200])
                if parse_failures >= 2:
                    return None
                self.tell(_PARSE_FAIL_HINT)
                continue

            if cmd.kind == "FINAL":
                if self.emit_hypothesis:
                    self._record_hypothesis(parsed.reply)
                self._record("FINAL", parsed, started, arg=cmd.arg[:500], reply=True)
                return extract_final_sids(cmd.arg, parsed.reply)

            # Broken-record circuit breaker: repeating the same command prompts
            # for closure, and three in a row jumps straight to a forced FINAL
            if cmd == prev_cmd:
                repeat_count += 1
                if repeat_count >= 2:
                    self.trace["commands"].append({
                        "cmd": "REPEAT_BREAK",
                        "arg": f"{cmd.kind} {cmd.arg}"[:200],
                        "ms": _elapsed_ms(started),
                    })
                    return None
                self.tell(_REPEAT_HINT)
                continue
            prev_cmd = cmd
            repeat_count = 0

            result = self.tools.execute(cmd)
            self.verified_sids.update(result.verified)
            self.vector_candidate_sids.update(result.vector_candidates)
            self._record(
                cmd.kind, parsed, started, arg=cmd.arg[:300], reply=True,
                result_chars=len(result.text), result=result.text[:1500],
            )
            self.tell(result.text + _CLOSING_REMINDER)
        return None

    def force_final(self, seed: list[str], attempt: int = 0) -> list[str]:
        """Force closure: attach the candidate sid list for the model to copy
        from, then extract the sids."""
        self.tell(_SALVAGE_MESSAGES[min(attempt, 1)].format(seeds=" ".join(seed)))
        started = time.perf_counter()
        parsed = self._ask(max_tokens=512)
        cmd = parsed.command
        arg = cmd.arg if cmd and cmd.kind == "FINAL" else ""
        self._record("FINAL(forced)", parsed, started,
                     arg=(arg or parsed.reply)[:500], reply=True)
        return extract_final_sids(arg, parsed.reply)

    # ── conversation plumbing ───────────────────────────────────────────
    def _ask(self, *, max_tokens: int):
        resp = self.llm.chat(messages=self.messages, temperature=0.0, max_tokens=max_tokens)
        parsed = parse_response(resp)
        # Only retain the compact command in conversation history. Raw reasoning
        # is diagnostic data, not useful context, and replaying it would inflate
        # the next request's input tokens.
        self.messages.append({
            "role": "assistant",
            "content": f"{parsed.command.kind} {parsed.command.arg}" if parsed.command else "",
        })
        return parsed

    def _record(self, cmd: str, parsed, started: float, *, arg: str,
                reply: bool = False, **extra) -> None:
        entry = {"cmd": cmd, "arg": arg, **extra}
        if reply:
            entry["reply"] = parsed.reply[:1200]
        entry["ms"] = _elapsed_ms(started)
        self.trace["commands"].append({**entry, **parsed.diagnostics})

    def _record_hypothesis(self, reply: str) -> None:
        """The agent's self-reported answer hypothesis (productionizing
        "hypothesis recovery", replacing hyp-v1's after-the-fact 4o-mini
        extraction).

        Capture HYPOTHESIS only to end of line; if a FINAL or sid token follows
        on the same or an adjacent line (the agent writes both together), cut
        before FINAL so the FINAL line's sids are not swallowed into the
        hypothesis (seen in hyp-v1 06db6396 and the 120b filter).
        """
        found = re.search(r"HYPOTHESIS\s*[::]\s*([^\n]+)", reply, re.IGNORECASE)
        hypothesis = found.group(1).strip() if found else ""
        hypothesis = re.split(r"\bFINAL\b", hypothesis, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if hypothesis and hypothesis.upper() != "NONE":
            self.trace["hypothesis"] = hypothesis[:200]


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
