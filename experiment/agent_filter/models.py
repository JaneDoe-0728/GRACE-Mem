"""The vocabulary every Agent Filter module speaks.

Two identifiers and two value objects, kept here so the parser, the loop, and
the context renderer can share them without importing each other.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# The header the answer context puts above the retrieved evidence, and the sid
# tag each entry carries. Both are the wire format between retrieval, this
# package, and the answering stage.
EVIDENCE_HEADER = "### Evidence Summary"
SID_RE = re.compile(r"\[sid=([^\]\s]+)\]")


@dataclass(frozen=True)
class Command:
    """One agent command: GREP, READ, VECTOR or FINAL, with its argument.

    READ's argument keeps the window size appended ("sid k") because that is the
    shape the protocol emits and the tool consumes.
    """
    kind: str
    arg: str


@dataclass(frozen=True)
class ParsedResponse:
    """One model reply, after the protocol has looked in every channel.

    Attributes:
        raw_reply: The first candidate text, kept for the trace even when
            nothing parsed out of it.
        reply: The candidate the command came from -- what downstream reads for
            a HYPOTHESIS line -- or ``raw_reply`` when no command parsed.
        command: The parsed command, or None when the reply carried none.
        source: Which channel the command came from (content/tool_calls/reasoning).
        diagnostics: Per-response diagnostics for the trace.
    """
    raw_reply: str
    reply: str
    command: Command | None
    source: str | None
    diagnostics: dict = field(default_factory=dict)
