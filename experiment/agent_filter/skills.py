"""A question-shape driven skill library for the agent.

Search tactics distilled from the error analysis and successful trajectories of
v2-v9, injected into the agent's task message according to the **shape of the
question** (detected by regex, independent of any benchmark category label). Each
skill is a concrete playbook: which anchor to use, how to phrase the GREP, when
to READ, and what to keep when selecting.

Grounded in:
- a successful trajectory: search broad -> drill in -> verify backwards from the
  answer span (the binaural beats case)
- failure mode 1: aggregation questions miscount mentions, or keep only one
  (16 of 39 v2 errors)
- failure mode 2: KU picks up a stale value (the failing contexts were down to
  1-2 entries)
- failure mode 3: a whole-sentence phrase matches nothing, and date conditions
  cannot be grepped (March is not 2023/03)
"""
from __future__ import annotations

import re


def _rx(p: str) -> re.Pattern:
    return re.compile(p, re.IGNORECASE)


# ── Disabled: the counting skill ─────────────────────────────────────────────
# v10's paired-verification attribution: questions this skill fired on held up
# only 73% of the time, against 93-100% for the other skills. Its advice -- "keep
# every instance / more evidence beats cleaner" -- collides head-on with the
# precision engine, the same pattern seen with min-keep and sufficiency in v3-v9.
# The real bottleneck on counting questions is the answering side's ability to
# aggregate, which cannot be fixed from the evidence side (v7's hard prompt failed
# the same way). The text is kept so it can be retested against a stronger model.
_COUNTING_SKILL_DISABLED = (
    "counting",
    _rx(r"\b(how many (?!(days?|weeks?|months?|years?|hours?|minutes?)\b)|how much|"
        r"how often|how frequently|number of|total|in total|altogether|count|sum of)\b"),
    ("SKILL counting/aggregation — the answer is a NUMBER computed from multiple mentions:\n"
    "- Enumerate EVERY mention: grep the countable noun AND its variants "
    "(singular/plural/synonyms, e.g. appointment|visit|checkup). One grep is never "
    "enough — different sessions use different words.\n"
    "- If a timeframe is given, ALSO grep the date stamp form: 'in March' → GREP 2023/03 "
    "(stamps are [YYYY/MM/DD]); combine: GREP 2023/03.*appointment.\n"
    "- READ each hit to check it is a DISTINCT event (same event re-mentioned twice "
    "counts once; two events on different dates count separately).\n"
    "- KEEP every distinct instance in FINAL — dropping one mention breaks the count. "
    "For counting questions, more evidence beats cleaner evidence."),
)

# Each skill is (name, detector, strategy).
# NOTE: counting is not in this list by default (see _COUNTING_SKILL_DISABLED).
# With GREP_AGENT_COUNTING_SKILL=1, select_skills inserts it at the front (for
# ablations and retesting on 120b).
SKILLS: list[tuple[str, re.Pattern, str]] = [
    (
        "latest-value",
        _rx(r"\b(most recent(ly)?|latest|current(ly)?|now|these days|still|"
            r"nowadays|as of|right now|today)\b|\b(did i .*(change|update|switch))\b"),
        ("SKILL latest-value — the fact may have been UPDATED over time:\n"
        "- Grep the target entity/fact across ALL sessions; collect every dated mention "
        "(the same fact stated with different values on different dates).\n"
        "- KEEP BOTH the old and the new mentions in FINAL — the answer needs the most "
        "recent one, and the reader picks it by comparing the [date] stamps. Keeping "
        "only one mention risks keeping the stale value.\n"
        "- If only one mention is found, grep synonyms of the fact before concluding "
        "there was no update."),
    ),
    (
        "temporal-computation",
        _rx(r"\b(how long|how many (days|weeks|months|years)|days? (before|after|between)|"
            r"ago\b|duration|since when|when did|what (day|date))\b"),
        ("SKILL temporal-computation — the answer needs DATES for arithmetic or ordering:\n"
        "- Locate the dated anchor turn for EACH event in the question (two events for "
        "'between', one for 'ago/when').\n"
        "- Date stamps are searchable: GREP 2023/05 finds May-2023 turns; month names in "
        "text ('last Saturday', 'in January') often do NOT appear as stamps — grep the "
        "EVENT keywords first, then read the [date] stamp of the hit.\n"
        "- Prefer turns whose text also states the date explicitly; keep every turn whose "
        "[date] stamp is needed for the computation."),
    ),
    (
        "preference-recommendation",
        _rx(r"\b(recommend|suggest(ion)?s?|what should i|any (tips|ideas|advice)|"
            r"help me (choose|pick|plan)|would i (like|prefer|enjoy))\b"),
        ("SKILL preference/recommendation — the answer must fit the USER's own stated "
        "situation, not generic advice:\n"
        "- The key evidence is what the USER said about themselves: grep "
        "'i (have|use|own|prefer|like|love|enjoy|hate|dislike|am allergic)' combined with "
        "the topic word; also grep the user's named gear/brands/places.\n"
        "- KEEP the user-side turns (sid ending :u) describing their setup, constraints "
        "and tastes — an assistant's earlier suggestions are secondary evidence.\n"
        "- Multiple aspects of their situation may live in different sessions; sweep more "
        "than one keyword before FINAL."),
    ),
    (
        "literal-recall",
        _rx(r"\b(what (is|was|did)|which|who|whom|whose|where (did|do|was)|"
            r"name of|called)\b"),
        ("SKILL literal-recall — the answer is a literal span (name/number/place/title):\n"
        "- Anchor on the RAREST word in the question (proper nouns, unusual terms, "
        "numbers) — one rare word beats a long phrase (multi-word patterns match as an "
        "exact phrase and usually miss).\n"
        "- After locating a promising turn, READ it — the span is often deep inside a "
        "long turn.\n"
        "- Before FINAL, verify: grep a distinctive fragment of the answer you found "
        "(e.g. GREP 38 subjects) to confirm the span really exists in the selected turn.\n"
        "- Select the minimal turn(s) containing the span; drop topical look-alikes."),
    ),
]

# literal-recall's detector is deliberately broad (what/which/who), so it acts as
# the fallback skill. List order is priority order, and at most the first N
# matches are taken.
MAX_SKILLS = 2


def select_skills(question: str) -> list[tuple[str, str]]:
    """Return [(name, strategy)], at most MAX_SKILLS of them, in SKILLS order.

    With GREP_AGENT_COUNTING_SKILL=1 the disabled counting skill is inserted at
    the front (for ablations).
    """
    import os
    active = SKILLS
    if os.getenv("GREP_AGENT_COUNTING_SKILL", "0") not in ("0", "", "false"):
        active = [_COUNTING_SKILL_DISABLED, *SKILLS]
    out = []
    for name, det, strategy in active:
        if det.search(question):
            out.append((name, strategy))
        if len(out) >= MAX_SKILLS:
            break
    return out
