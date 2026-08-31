"""Reading the seed evidence out of an answer context, and writing one back.

Agent Filter is handed a rendered answer context and has to return one, so this
module owns both directions of that format: which sids the upstream pipeline
put in the Evidence Summary, with what rerank scores and what graph prefix, and
how the selected turns are rendered back into the same block.
"""
from __future__ import annotations

import re

from experiment.agent_filter.corpus import Corpus
from experiment.agent_filter.models import EVIDENCE_HEADER, SID_RE


def seed_sids_from_context(context: str) -> list[str]:
    """Pull the sids out of the Evidence Summary block in order of appearance,
    deduplicated but order-preserving."""
    idx = context.find(EVIDENCE_HEADER)
    if idx == -1:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in SID_RE.finditer(context[idx:]):
        s = m.group(1).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# The rerank score on each Evidence Summary entry: `[sid=...][score=0.615]`
# (the Score column in the front-end candidate table)
_SID_SCORE_RE = re.compile(r"\[sid=([^\]\s]+)\][^\[]*?\[score=([-\d.]+)\]")


def seed_scores_from_context(context: str) -> dict[str, float]:
    """Extract the rerank score for each seed sid, keeping the first occurrence.
    Entries that fail to parse are skipped."""
    idx = context.find(EVIDENCE_HEADER)
    scan = context[idx:] if idx != -1 else context
    out: dict[str, float] = {}
    for m in _SID_SCORE_RE.finditer(scan):
        sid = m.group(1).strip()
        if sid in out:
            continue
        try:
            out[sid] = float(m.group(2))
        except ValueError:
            continue
    return out


def graph_context_from_context(context: str, *, max_chars: int = 12000) -> str:
    """Return the retrieved graph prefix (Entities/Relationships) if present."""
    idx = context.find(EVIDENCE_HEADER)
    prefix = context[:idx] if idx != -1 else context
    if "=== Entities ===" not in prefix and "=== Relationships ===" not in prefix:
        return ""
    prefix = prefix.strip()
    return prefix if len(prefix) <= max_chars else prefix[:max_chars] + "\n…(graph context truncated)"


def candidates_block(corpus: Corpus, sids: list[str]) -> str:
    """Render the candidate turns the agent will decide over."""
    lines = []
    for s in sids:
        entry = corpus.display_entry(s, max_chars=400)
        if entry is None:
            lines.append(f"[sid={s}] (raw text unavailable)")
        else:
            t = corpus.resolve(s)[0]
            lines.append(f"[sid={s}] [{t.date}] {entry}")
    return "\n".join(lines) if lines else "(none)"


def rebuild_context(
    context: str,
    corpus: Corpus,
    final_sids: list[str],
    *,
    include_pair: bool = True,
    include_prefix: bool = True,
) -> tuple[str, list[str]]:
    """Rebuild the Evidence Summary block. With include_pair=True a selected sid
    brings its pair partner along (the other half of the same user<->assistant
    exchange) -- the agent sometimes picks the wrong side of the right pair, and
    presenting them together keeps the crucial evidence. Returns
    (context, context_sids)."""
    idx = context.find(EVIDENCE_HEADER)
    if include_prefix:
        head = context[:idx].rstrip("\n") if idx != -1 else context.rstrip("\n")
    else:
        head = ""
    lines = [head, EVIDENCE_HEADER] if head else [EVIDENCE_HEADER]

    entries: list[str] = []  # sids (pair base or split sid), order-preserving dedup
    seen: set[str] = set()
    for s in final_sids:
        key = s.rsplit(":", 1)[0] if include_pair and (s.endswith((":u", ":a"))) else s
        if key not in seen:
            seen.add(key)
            entries.append(key)

    context_sids: list[str] = []
    for key in entries:
        turns = corpus.resolve(key)
        if not turns:
            continue
        context_sids.extend(t.sid for t in turns)
        sid_tags = "".join(f"[sid={t.sid}]" for t in turns)
        entry = corpus.display_entry(key, max_chars=4000 * len(turns))
        dt_str = f"[{turns[0].date}]" if turns[0].date else ""
        lines.append(f"  • {dt_str}{sid_tags}[score=--] {entry} ")
    return "\n".join(lines), context_sids


def append_fetched_evidence(
    context: str,
    corpus: Corpus,
    added: list[str],
    seed_sids: list[str],
) -> tuple[str, list[str]]:
    """Hang the agent's finds on the end of the original context, untouched.

    fetch_only's guarantee is that the information is never less than baseline,
    so the original text is left word for word -- baseline evidence may be plain
    text with no sid, which a rebuild would wrongly delete. Returns
    (context, context_sids).
    """
    lines = ["", "### Additional Evidence (agent-retrieved)"]
    evidence_sids = list(seed_sids)
    for sid in added:
        turn = corpus.resolve(sid)[0]
        entry = corpus.display_entry(sid)
        dt = f"[{turn.date}]" if turn.date else ""
        lines.append(f"  • {dt}[sid={sid}][score=--] {entry} ")
        evidence_sids.append(sid)
    return context.rstrip("\n") + "\n".join(lines), evidence_sids
