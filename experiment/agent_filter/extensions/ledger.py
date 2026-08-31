"""Dated Fact Ledger: compile the evidence into a table of dated facts before
answering (compiling is not judging).

The paradigm is a second mechanism orthogonal to the grep agent's *selection*:
*representation change*. It does not push recall (it respects the Precision
Wall); it moves temporal arithmetic and latest-value judgements out of the LLM's
reasoning at answer time and into a lookup over the compiled table.

Measured on the temporal+KU subset, two replications:
  temporal: grep -> ledger stacked, 81.5/82.3 vs v2's 79.2 (winning 4 of 4)
  KU:       ledger alone, 74.4/74.4 vs v2's 73.1 (stacking cuts away the stale-
            value mentions, so the two cannot be combined here)
"""
from __future__ import annotations

COMPILE_SYSTEM = """You are a fact compiler. From the conversation evidence below, extract every
dated fact as a table row: | date | who/what | fact |
- One row per dated mention. Use the [YYYY/MM/DD] stamps shown on each excerpt.
- Include EVERY dated mention of the same entity/fact (do not merge updates —
  each update is its own row).
- Only extract what is literally in the evidence. Output ONLY the table."""

LEDGER_HEADER = "\n\n### Dated Fact Table (compiled from the evidence above)\n"


def compile_table(llm, evidence: str, *, max_chars: int = 24000, max_tokens: int = 1500) -> str:
    """Compile the evidence block into a dated fact table in a single LLM call.
    Returns an empty string on failure."""
    try:
        resp = llm.chat(messages=[
            {"role": "system", "content": COMPILE_SYSTEM},
            {"role": "user", "content": evidence[:max_chars]},
        ], temperature=0.0, max_tokens=max_tokens)
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def append_ledger(context: str, table: str) -> str:
    if not table:
        return context
    return context + LEDGER_HEADER + table
