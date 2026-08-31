"""Is the selected evidence enough to answer the question at all?

An independent audit call, blind to the agent's search history: it sees only
the question and the evidence the agent settled on. Its verdict is advisory --
it can send the agent back out for another round, but it can never remove
evidence, and a flaky verdict must not send the loop spinning.
"""
from __future__ import annotations

import re

from experiment.agent_filter.corpus import Corpus
from experiment.agent_filter.prompting.verification import (
    SUFFICIENCY_SYSTEM,
    SUFFICIENCY_USER,
)


def check_sufficiency(
    llm,
    *,
    question: str,
    question_date: str | None,
    corpus: Corpus,
    sids: list[str],
) -> tuple[bool, str]:
    """An independent audit call: is the evidence enough to answer in full?
    Returns (sufficient, missing_desc).
    A parse failure counts as sufficient -- a flaky verifier must not send the
    loop spinning."""
    lines = []
    for s in sids:
        t = corpus.resolve(s)
        if not t:
            continue
        # Must be as complete as the final context (4000 per side): shown a
        # truncated version, the verifier misreads "detail buried deep in a long
        # turn" as missing -- the main driver of the measured 42% false triggers.
        entry = corpus.display_entry(s, max_chars=4000 * len(t))
        lines.append(f"[{t[0].date}] {entry}")
    reply_msgs = [
        {"role": "system", "content": SUFFICIENCY_SYSTEM},
        {"role": "user", "content": SUFFICIENCY_USER.format(
            question=question,
            date_line=f"QUESTION DATE: {question_date}\n" if question_date else "",
            evidence="\n".join(lines) or "(none)",
        )},
    ]
    resp = llm.chat(messages=reply_msgs, temperature=0.0, max_tokens=512)
    reply = (resp.choices[0].message.content or "").strip()
    for line in reply.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\**\s*INSUFFICIENT\s*\**\s*[::]?\s*(.*)$", line, re.IGNORECASE)
        if m:
            return False, (m.group(1) or reply[:300]).strip()
        if re.match(r"^\**\s*SUFFICIENT\b", line, re.IGNORECASE):
            return True, ""
    return True, ""
