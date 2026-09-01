"""One typed reading of GREP_AGENT_PARAMS.

Every knob arrived as a `p.get("grep_agent_...", default)` scattered through the
pipeline, which put the defaults in a dozen places and the coercions in a dozen
more -- and made "what can this run be configured to do?" a question you
answered by grepping. Reading the mapping once, here, answers it by reading one
dataclass.

The defaults below are the harness's own, which are deliberately not the same as
experiment_config's: a caller passing an incomplete mapping gets the
conservative reading (adjudication off), while a benchmark run passes
GREP_AGENT_PARAMS in full.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

# Categories may be restricted per layer; None means "every category".
Categories = tuple[str, ...] | None

#: Keys still accepted from a GREP_AGENT_PARAMS mapping that no longer reach any
#: decision. Key -> why. Old traces, sweep scripts and archived configs still
#: carry them, so reading such a mapping must not fail -- but it must not look
#: like the setting took effect either.
INERT_PARAMS = {
    "grep_agent_require_verified_additions": (
        "the provenance gate was removed on 2026-07-22; a VECTOR hit counts as "
        "verified like GREP/READ, and only hallucinated sids are dropped"
    ),
}

_warned: set[str] = set()


@dataclass(frozen=True)
class AgentFilterConfig:
    """What one refinement is allowed to do."""

    # ── The agent's own budget ──────────────────────────────────────────
    # filter       -> may only keep or drop the retrieved candidates
    # filter_fetch -> may also add corpus sids it found itself
    # fetch_only   -> may only add; the original context is left word for word
    mode: str = "filter_fetch"
    max_calls: int = 8
    max_sids: int = 16
    grep_max_lines: int = 30
    emit_hypothesis: bool = False
    use_skills: bool = False

    # ── Graph facts: switchable on the filter prompt and the answer context
    # independently, for ablations ──────────────────────────────────────
    filter_include_graph: bool = False
    answer_include_graph: bool = True
    graph_context_max_chars: int = 12000

    # ── VECTOR, the semantic search the agent drives itself ─────────────
    vector_search: bool = True
    vector_topn: int = 8
    vector_min_score: float = 0.30

    # ── Answer-blind adjudication of the discarded seeds ────────────────
    adjudicate: bool = False
    adjudicate_categories: Categories = None

    # ── What to do when the agent will not close ────────────────────────
    abstention_hint: bool = False

    # ── Rebuilding the context ──────────────────────────────────────────
    include_pair: bool = True

    @classmethod
    def from_params(cls, params: dict | None) -> AgentFilterConfig:
        """Read a GREP_AGENT_PARAMS-shaped mapping, keeping the harness defaults
        for anything it does not name."""
        p = params or {}
        for key in INERT_PARAMS.keys() & p.keys():
            if key in _warned:
                continue
            _warned.add(key)
            warnings.warn(
                f"{key} no longer affects anything and was ignored "
                f"({INERT_PARAMS[key]})",
                FutureWarning,
                stacklevel=2,
            )

        def flag(key: str, default: bool) -> bool:
            # Several of these ship as 0/1 rather than False/True.
            value = p.get(key, default)
            return bool(int(value)) if isinstance(value, str) else bool(value)

        return cls(
            mode=p.get("grep_agent_mode", cls.mode),
            max_calls=int(p.get("grep_agent_max_calls", cls.max_calls)),
            max_sids=int(p.get("grep_agent_max_sids", cls.max_sids)),
            grep_max_lines=int(p.get("grep_agent_grep_max_lines", cls.grep_max_lines)),
            emit_hypothesis=flag("grep_agent_emit_hypothesis", cls.emit_hypothesis),
            use_skills=flag("grep_agent_use_skills", cls.use_skills),
            filter_include_graph=flag(
                "grep_agent_filter_include_graph_context", cls.filter_include_graph),
            answer_include_graph=flag(
                "grep_agent_answer_include_graph_context", cls.answer_include_graph),
            graph_context_max_chars=int(p.get(
                "grep_agent_graph_context_max_chars", cls.graph_context_max_chars)),
            vector_search=flag("grep_agent_vector_search", cls.vector_search),
            vector_topn=int(p.get("grep_agent_vector_topn", cls.vector_topn)),
            vector_min_score=float(p.get("grep_agent_vector_min_score", cls.vector_min_score)),
            adjudicate=flag("grep_agent_adjudicate", cls.adjudicate),
            adjudicate_categories=p.get(
                "grep_agent_adjudicate_categories", cls.adjudicate_categories),
            abstention_hint=flag("grep_agent_abstention_hint", cls.abstention_hint),
            include_pair=flag("grep_agent_include_pair", cls.include_pair),
        )

    def applies_to(self, categories: Categories, category: str | None) -> bool:
        """A layer restricted to `categories` runs only for those; None = all."""
        return categories is None or category in categories
