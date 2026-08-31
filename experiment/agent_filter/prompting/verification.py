"""What the sufficiency verifier is told, and how its verdict reaches the agent.

The verifier is an independent audit call: it never sees the agent's search
history, only the question and the evidence the agent settled on.
"""
from __future__ import annotations

# ── Sufficiency verifier: an independent audit call that does not share the
# agent's conversation ──────────────────────────────────────────────────────
SUFFICIENCY_SYSTEM = """You are a strict evidence auditor for a QA system.
Given a QUESTION and the EVIDENCE turns selected to answer it, judge whether the
evidence ALONE is enough to answer the question COMPLETELY and precisely.

Pay special attention to:
- counting/aggregation questions: does the evidence contain EVERY instance being
  counted? If the question asks "how many X", missing even one mention makes it
  insufficient.
- temporal questions: are ALL dates/durations needed for the comparison present?
- multi-part questions: is every part covered?

Do NOT judge answer quality — only whether the necessary information is present.

Reply with EXACTLY one line:
SUFFICIENT
or
INSUFFICIENT: <the specific missing information — which entity, date, or how many more instances are likely missing>"""

SUFFICIENCY_USER = """QUESTION: {question}
{date_line}
SELECTED EVIDENCE:
{evidence}

Is this evidence sufficient to answer the question completely? Reply SUFFICIENT or INSUFFICIENT: <missing>."""

# The gap-directed top-up search instruction, fed back into the agent's conversation
GAP_HINT_TEMPLATE = """An independent verifier reviewed your FINAL selection and judged the evidence
INSUFFICIENT: {missing}

Continue searching with GREP to fill exactly this gap. Your previous selections are
kept — you only need to find the MISSING evidence and ADD it. When done, reply:
FINAL <all previous sids> <newly found sids>"""
