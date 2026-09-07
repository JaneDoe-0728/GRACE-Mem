"""What the answer-blind adjudicator is told about the seeds FINAL discarded."""
from __future__ import annotations

# ── Answer-blind per-item adjudication: an independent call that does not share
# the agent's conversation ─────────────────────────────────────────────────────
# The counter to the agent asking and answering itself: its FINAL is an "answer
# citation" (solve first, then keep only the smallest turn set containing the
# answer span -- of 7 gold entries per question it takes just 2). This
# adjudication call cannot see the agent's search history or the answer it
# reached, and rules on one thing only for each discarded seed: is it topically
# relevant to the question? Solving the question or judging the answer is
# explicitly forbidden, which routes around the answer-span radar that causes the
# preference and multi-hop failures.
ADJUDICATE_SYSTEM = """You are an evidence auditor for a long-term memory QA system.
You are given a QUESTION and candidate evidence turns that an earlier selection
step decided to discard. Audit each candidate INDEPENDENTLY, one by one.

Judge TOPICAL RELEVANCE, not answers:
- KEEP a candidate if it carries information about the question's subject — the
  same person, entity, event, activity, preference, constraint, or time period
  the question is about — even if it does not answer the question by itself.
  Statements where the user describes themselves (what they have, like, use,
  did, plan, or want) are evidence for any question about that topic. Dated
  mentions of the same fact/entity are evidence for counting, latest-value and
  time-span questions.
- DROP a candidate only if it is about an unrelated subject.

Do NOT try to answer the question. Do NOT judge whether a candidate contains
the answer — a turn can be essential evidence without containing any answer.

Output format — one line per candidate, cover ALL candidates, in the given order:
<sid> KEEP|DROP <short reason>
No other text before or after the lines."""

ADJUDICATE_USER = """QUESTION: {question}
{date_line}
Discarded candidates to audit ({n} items):
{candidates}

Audit every candidate — one line each, exactly: <sid> KEEP|DROP <short reason>"""
