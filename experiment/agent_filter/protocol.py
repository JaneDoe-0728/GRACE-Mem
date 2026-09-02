"""One model reply in, one Command out -- whatever channel it arrived in.

There is no function-calling API here: local models (gpt-oss-20b via LM Studio)
are steadiest against a plain-text one-command-per-line protocol. But the
adapters do not agree on where a reply's visible text goes -- ``content``,
native ``tool_calls``, or ``reasoning`` -- and gpt-oss also emits Harmony's own
channel markup. Every one of those surfaces is normalized here, so the agent
loop never has to know which backend it is talking to.

A parse miss reads downstream as "the agent selected nothing", which is why the
parser is forgiving in the ways real replies have needed.
"""
from __future__ import annotations

import json
import re

from experiment.agent_filter.models import SID_RE, Command, ParsedResponse

# gpt-oss (harmony template) sometimes replies in native tool-call syntax:
#   <|channel|>commentary to=READ <|constrain|>json<|message|>{"id": "...", "k": 2}
_HARMONY_RE = re.compile(
    r"to=(?:\w+\.)?(GREP|READ|VECTOR|FINAL)\b.*?<\|message\|>\s*(\{.*?\})\s*(?:<\|\w+\|>|$)",
    re.IGNORECASE | re.DOTALL,
)
# A messier variant: to=GREP <|constrain|>="pattern" (no <|message|> JSON)
# Namespace prefixes are treated loosely: functions./tool./any <ns>. is stripped
# (the model on box 92 emits to=tool.GREP).
_HARMONY_LOOSE_RE = re.compile(r"to=(?:\w+\.)?(GREP|READ|VECTOR|FINAL)\b(.*)$", re.IGNORECASE)

_CMD_NAMES = {"GREP", "READ", "VECTOR", "FINAL"}

# The plain-text protocol: one command as the last line of the reply.
_GREP_RE = re.compile(r"^\s*GREP\s+(.+?)\s*$", re.IGNORECASE)
_READ_RE = re.compile(r"^\s*READ\s+(\S+)(?:\s+(\d+))?\s*$", re.IGNORECASE)
_VECTOR_RE = re.compile(r"^\s*VECTOR\s+(.+?)\s*$", re.IGNORECASE)
_FINAL_RE = re.compile(r"^\s*FINAL\s*[::]?\s*(.*?)\s*$", re.IGNORECASE)


def response_diagnostics(resp) -> dict:
    """Response-level diagnostics: tells "the model genuinely replied empty" apart
    from "the output landed in the reasoning channel / tool_calls where content
    cannot see it" (common with gpt-oss harmony). Attached to every step trace."""
    try:
        choice = resp.choices[0]
        msg = choice.message
    except (AttributeError, IndexError):
        return {"diag": "no_choices"}
    d: dict = {"finish_reason": getattr(choice, "finish_reason", None)}
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    if reasoning:
        d["reasoning"] = str(reasoning)[:1200]
    tc = getattr(msg, "tool_calls", None)
    if tc:
        d["tool_calls"] = [
            {"name": getattr(getattr(t, "function", None), "name", None),
             "arguments": str(getattr(getattr(t, "function", None), "arguments", ""))[:300]}
            for t in tc
        ]
    content = getattr(msg, "content", None)
    if not (content or "").strip():
        # Empty content: record which fields the message actually carried, so an
        # adapter dropping fields can be ruled out
        d["content_empty"] = True
        d["message_keys"] = sorted(vars(msg).keys())
    return d


def response_command_candidates(resp) -> list[tuple[str, str]]:
    """Return possible command-bearing text from an OpenAI-compatible response.

    gpt-oss/Harmony adapters do not agree on where the visible command goes:
    some put it in ``content``, some expose native ``tool_calls``, and some
    (notably older LM Studio adapters) put it in ``reasoning``.  The command
    parser itself is format-agnostic, so normalize all three surfaces here.
    """
    try:
        message = resp.choices[0].message
    except (AttributeError, IndexError):
        return []

    candidates: list[tuple[str, str]] = []
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        candidates.append(("content", content.strip()))

    tool_calls = getattr(message, "tool_calls", None) or []
    for call in tool_calls:
        function = getattr(call, "function", None)
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
        if not name:
            continue
        name = str(name).split(".")[-1].upper()
        if name not in _CMD_NAMES:
            continue
        if isinstance(arguments, str):
            payload = arguments
        else:
            payload = json.dumps(arguments or {}, ensure_ascii=False)
        candidates.append(("tool_calls", f"to={name} <|message|>{payload}"))

    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        candidates.append(("reasoning", reasoning.strip()))
    return candidates


def _flatten_json(obj) -> tuple[list[str], list[int]]:
    """Recursively collect the strings and integers in a JSON payload (the model's
    schema is not predictable)."""
    strings: list[str] = []
    ints: list[int] = []
    stack = [obj]
    while stack:
        cur = stack.pop(0)
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
        elif isinstance(cur, bool):
            continue
        elif isinstance(cur, int):
            ints.append(cur)
        elif isinstance(cur, str):
            strings.append(cur)
    return strings, ints


def _parse_harmony(reply: str) -> Command | None:
    """Parse a Harmony-format reply into its channels.

    Some backends wrap replies in Harmony's channel markup; unwrapping here
    keeps the callers from having to know which backend produced a reply.
    """
    m = None
    for m in _HARMONY_RE.finditer(reply):
        pass  # take the last tool call
    if m is None:
        return None
    kind = m.group(1).upper()
    try:
        payload = json.loads(m.group(2))
    except json.JSONDecodeError:
        return None

    strings, ints = _flatten_json(payload)
    # If the payload names the command itself (e.g. {"cmd": ["GREP", ...]}), that wins
    for s in strings:
        if s.strip().upper() in _CMD_NAMES:
            kind = s.strip().upper()
    args = [s for s in strings if s.strip().upper() not in _CMD_NAMES]

    if kind in ("GREP", "VECTOR"):
        return Command(kind, _unquote(args[0])) if args else None
    if kind == "READ":
        sid = next((s for s in args if ":" in s), None)
        k = next((i for i in ints if 0 < i <= 10), 2)
        return Command("READ", f"{_unquote(sid)} {k}") if sid else None
    return Command("FINAL", " ".join(s for s in args if ":" in s))


def _parse_harmony_loose(reply: str) -> Command | None:
    """Last resort: JSON-less variants such as `to=GREP <|constrain|>="pattern"`.
    Take the rest of the line after to=CMD and strip the harmony markers and the
    constrain/json noise to get the argument."""
    m = None
    for line in reply.splitlines():
        for m2 in _HARMONY_LOOSE_RE.finditer(line):
            m = m2
    if m is None:
        return None
    kind = m.group(1).upper()
    tail = re.sub(r"<\|[^|]*\|>", " ", m.group(2))
    tail = re.sub(r"\b(?:json|commentary|response)\b", " ", tail, flags=re.IGNORECASE)
    tail = tail.strip().lstrip("=").strip()
    arg = _unquote(tail)
    if kind in ("GREP", "VECTOR"):
        return Command(kind, arg) if arg else None
    if kind == "READ":
        sid = next((s for s in re.split(r"[,\s]+", arg) if ":" in s), None)
        return Command("READ", f"{_unquote(sid)} 2") if sid else None
    return Command("FINAL", arg)


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
        return s[1:-1]
    return s


def parse_command(reply: str) -> Command | None:
    """Parse the command out of the reply's last few lines, tolerating reasoning
    before it. The plain-text protocol is tried first, then harmony's native
    tool-call syntax as a fallback.
    Harmony markers (<|channel|> and friends) are turned into newlines first, so
    a plain-text command wedged in after <|message|> is still caught by the
    line-based parse."""
    sanitized = re.sub(r"<\|[^|]*\|>", "\n", reply)
    # 120B (gpt-oss) harmony dual-channel run-together: several commands crammed
    # onto one line, with the whole span repeated ("GREP MelanieGREP Melanie",
    # "READ 0__2:0 5FINAL 0__2:0"). When the line starts with GREP the entire line
    # is taken as the pattern and the trailing FINAL is swallowed -- the root cause
    # of the 49.6% L1 failure rate (2026-07-06). So: an uppercase command word
    # directly after a lowercase char/digit/quote -> break the line. 20B emits one
    # command per line and is unaffected.
    sanitized = re.sub(r"(?<=[a-z0-9\"'\)\].:])((?:GREP|READ|VECTOR)\s|FINAL\b)",
                       r"\n\1", sanitized)
    for line in reversed(sanitized.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        if m := _FINAL_RE.match(line):
            if not m.group(1).strip():
                # Empty FINAL: if the same reply holds other runnable commands,
                # run those first (120B often dumps the whole plan --
                # GREP..READ..FINAL -- in one go)
                continue
            return Command("FINAL", m.group(1))
        if m := _GREP_RE.match(line):
            return Command("GREP", _unquote(m.group(1)))
        if m := _READ_RE.match(line):
            return Command("READ", f"{_unquote(m.group(1))} {m.group(2) or 2}")
        if m := _VECTOR_RE.match(line):
            return Command("VECTOR", _unquote(m.group(1)))
    return _parse_harmony(reply) or _parse_harmony_loose(reply)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")
_QUOTED_RE = re.compile(r"[\"'`]([^\"'`]{2,})[\"'`]")
_CMD_WORD_RE = re.compile(r"\b(GREP|READ|VECTOR|FINAL)\b", re.IGNORECASE)
# A sid as it appears in prose -- session:message[:u|a] -- rather than the
# bracketed [sid=...] form SID_RE matches in rendered candidate blocks.
_BARE_SID_RE = re.compile(r"\b[\w\-]+:\d+(?::[ua])?\b")


def parse_narrated_command(reasoning: str) -> Command | None:
    """Recover a command the model described instead of issuing.

    gpt-oss-20b via LM Studio routinely ends its turn after the analysis
    channel: `content` and `tool_calls` come back empty and only `reasoning`
    is populated, holding the plan in prose -- "Use READ on s1:6:u.",
    'Use GREP for "Music and Medicine".'. finish_reason is `stop`, so nothing
    was truncated; the model simply narrated and stopped. Measured across five
    runs, every single parse failure had that shape.

    This reads the intent back out, and deliberately only when the model named
    a command itself. "Search for X" is not treated as a GREP: inferring a tool
    call from an English verb is a guess about intent, and a wrong GREP costs a
    call from the budget and misleads the next turn. Of the ten observed
    failures it recovers the three that named a command and leaves the seven
    that only narrated as parse failures, which is what they are.

    Applied to the reasoning channel alone, and only after every ordinary
    surface has failed, so a well-formed reply never reaches it.
    """
    if not reasoning:
        return None
    matches = list(_CMD_WORD_RE.finditer(reasoning))
    if not matches:
        return None
    # The last mention: the model states its conclusion after reasoning toward it.
    match = matches[-1]
    kind = match.group(1).upper()
    after = reasoning[match.end():]

    if kind in ("READ", "FINAL"):
        sids = (
            SID_RE.findall(after)
            or _BARE_SID_RE.findall(after)
            or SID_RE.findall(reasoning)
            or _BARE_SID_RE.findall(reasoning)
        )
        if not sids:
            return None
        return Command("READ", f"{sids[0]} 2") if kind == "READ" else Command("FINAL", " ".join(sids))

    # GREP / VECTOR need a pattern, and the model quotes what it means to search
    # for. Prefer a quote after the command word, then anywhere in the text --
    # "Need to grep for X. Use regex \"X\"" names the pattern only on the second
    # sentence.
    quoted = _QUOTED_RE.search(after) or _QUOTED_RE.search(reasoning)
    return Command(kind, quoted.group(1)) if quoted else None


def parse_response(resp) -> ParsedResponse:
    """Read one chat completion for the single command it carries.

    Each surface is tried in turn and the first that parses wins, so hidden
    reasoning is never fed back to the model as history -- only the compact
    command is.
    """
    diagnostics = response_diagnostics(resp)
    candidates = response_command_candidates(resp)
    raw_reply = candidates[0][1] if candidates else ""
    for source, candidate in candidates:
        command = parse_command(candidate)
        if command is not None:
            diagnostics["command_source"] = source
            return ParsedResponse(raw_reply, candidate, command, source, diagnostics)

    # Nothing issued a command. Before giving up, check whether the model
    # described one in its reasoning -- see parse_narrated_command.
    for source, candidate in candidates:
        if source != "reasoning":
            continue
        command = parse_narrated_command(candidate)
        if command is not None:
            diagnostics["command_source"] = "reasoning_narrated"
            return ParsedResponse(raw_reply, candidate, command, "reasoning_narrated", diagnostics)

    return ParsedResponse(raw_reply, raw_reply, None, None, diagnostics)


def extract_final_sids(arg: str, full_reply: str = "") -> list[str]:
    """Pull the agent's final sid list out of its reply.

    The reply is free text and the model varies how it presents the list -- a
    line after "FINAL:", a bullet list, or both. Accepting all the observed
    shapes matters because a parse miss reads as the agent having selected
    nothing.
    """
    sids = SID_RE.findall(arg) + [
        s for s in re.split(r"[,\s]+", SID_RE.sub("", arg)) if s and ":" in s
    ]
    if not sids and full_reply:
        # Empty FINAL argument: the sids may sit on other lines (a newline
        # and bullet list after "FINAL:", and so on)
        sids = SID_RE.findall(full_reply) + [
            s.strip("*•-,.")
            for s in re.split(r"[,\s]+", SID_RE.sub("", full_reply))
            if ":" in s and re.search(r":\d+", s)
        ]
    return sids
