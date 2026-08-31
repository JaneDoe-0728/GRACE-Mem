"""Answer-blind per-item adjudication of the seeds FINAL discarded.

The agent's FINAL is an "answer citation": it solves the question first, then
keeps only the smallest turn set containing the answer span, so supporting
evidence that carries no answer is discarded systematically -- the root of the
preference and multi-hop failures.

The remedy is a second opinion that cannot see the answer. This call gets no
part of the agent's conversation, and rules on one thing per discarded seed: is
it topically relevant to the question? It is additive only -- the agent's own
0.84-precision picks are left alone.
"""
from __future__ import annotations

import re

from experiment.agent_filter.corpus import Corpus
from experiment.agent_filter.prompting.adjudication import (
    ADJUDICATE_SYSTEM,
    ADJUDICATE_USER,
)


def adjudicate_candidates(
    llm,
    *,
    question: str,
    question_date: str | None,
    corpus: Corpus,
    pending: list[str],
) -> tuple[list[str], dict]:
    """Rule KEEP/DROP on every seed that FINAL discarded.

    Returns (the KEEP sids, a per-item verdict dict). A candidate given no
    verdict counts as DROP -- adjudication is an add-only recovery, so no
    verdict means nothing is added back.
    """
    lines = []
    for s in pending:
        t = corpus.resolve(s)
        if not t:
            continue
        entry = corpus.display_entry(s, max_chars=700)
        dt = f"[{t[0].date}] " if t[0].date else ""
        lines.append(f"[sid={s}] {dt}{entry}")
    msgs = [
        {"role": "system", "content": ADJUDICATE_SYSTEM},
        {"role": "user", "content": ADJUDICATE_USER.format(
            question=question,
            date_line=f"QUESTION DATE: {question_date}\n" if question_date else "",
            n=len(lines),
            candidates="\n".join(lines) or "(none)",
        )},
    ]
    # Reasoning model: hidden thinking precedes the verdicts, so the token budget
    # has to be generous -- 2048 was measured truncating a 14-verdict output
    # halfway (in the child run, all 129 unjudged items came from this).
    resp = llm.chat(messages=msgs, temperature=0.0, max_tokens=4096)
    reply = (resp.choices[0].message.content or "").strip()
    kept: list[str] = []
    # reply = the adjudication call's raw response (one `<sid> KEEP|DROP <short
    # reason>` per line). The full text is kept so the front end can reconstruct
    # the adjudication trail, and verdict+reason are extracted per item
    # (reasons: sid -> "KEEP|DROP: reason").
    verdicts: dict = {
        "kept": [], "dropped": [], "unjudged": [],
        "reply": reply, "reply_chars": len(reply), "reasons": {},
    }
    judged: set[str] = set()
    for line in reply.splitlines():
        m = re.search(r"\b(KEEP|DROP)\b", line, re.IGNORECASE)
        if not m:
            continue
        for s in pending:
            if s in judged or s not in line:
                continue
            judged.add(s)
            decision = m.group(1).upper()
            # Take the short reason after KEEP/DROP on that line, dropping the sid
            # and the verdict token. The strip set keeps fullwidth punctuation
            # because the model emits it in reasons.  # allow-cjk
            reason = line[m.end():].strip(" \t:-—．。")
            verdicts["reasons"][s] = f"{decision}: {reason}" if reason else decision
            if decision == "KEEP":
                kept.append(s)
                verdicts["kept"].append(s)
            else:
                verdicts["dropped"].append(s)
            break
    verdicts["unjudged"] = [s for s in pending if s not in judged]
    return kept, verdicts
