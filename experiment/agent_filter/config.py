"""One typed reading of GREP_AGENT_PARAMS.

Every knob arrived as a `p.get("grep_agent_...", default)` scattered through the
pipeline, which put the defaults in a dozen places and the coercions in a dozen
more -- and made "what can this run be configured to do?" a question you
answered by grepping. Reading the mapping once, here, answers it by reading one
dataclass.

The defaults below are the harness's own, which are deliberately not the same as
experiment_config's: a caller passing an incomplete mapping gets the
conservative reading (adjudication off, no verify rounds, no padding), while a
benchmark run passes GREP_AGENT_PARAMS in full.
"""
from __future__ import annotations

from dataclasses import dataclass

# Categories may be restricted per layer; None means "every category".
Categories = tuple[str, ...] | None


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

    # ── The sufficiency verifier, and the gap search its verdict triggers ─
    verify_rounds: int = 0
    verify_max_calls: int = 4
    verify_categories: Categories = None
    gap_vector_topn: int = 0
    gap_vector_min_score: float = 0.30

    # ── Answer-blind adjudication of the discarded seeds ────────────────
    adjudicate: bool = False
    adjudicate_categories: Categories = None
    # Recall-recovery only: every discarded seed comes back without an LLM DROP.
    adjudicate_keep_all_categories: Categories = None

    # ── What to do when the agent will not close ────────────────────────
    force_verified_final: bool = False
    force_verified_min: int = 12
    abstention_hint: bool = False

    # ── Rebuilding the context ──────────────────────────────────────────
    min_keep_aggregation: int = 0
    include_pair: bool = True

    @classmethod
    def from_params(cls, params: dict | None) -> AgentFilterConfig:
        """Read a GREP_AGENT_PARAMS-shaped mapping, keeping the harness defaults
        for anything it does not name."""
        p = params or {}

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
            verify_rounds=int(p.get("grep_agent_verify_rounds", cls.verify_rounds)),
            verify_max_calls=int(p.get("grep_agent_verify_max_calls", cls.verify_max_calls)),
            verify_categories=p.get("grep_agent_verify_categories", cls.verify_categories),
            gap_vector_topn=int(p.get("grep_agent_gap_vector_topn", cls.gap_vector_topn)),
            gap_vector_min_score=float(p.get(
                "grep_agent_gap_vector_min_score", cls.gap_vector_min_score)),
            adjudicate=flag("grep_agent_adjudicate", cls.adjudicate),
            adjudicate_categories=p.get(
                "grep_agent_adjudicate_categories", cls.adjudicate_categories),
            adjudicate_keep_all_categories=p.get(
                "grep_agent_adjudicate_keep_all_categories",
                cls.adjudicate_keep_all_categories),
            force_verified_final=flag(
                "grep_agent_force_verified_final", cls.force_verified_final),
            force_verified_min=int(p.get(
                "grep_agent_force_verified_min", cls.force_verified_min)),
            abstention_hint=flag("grep_agent_abstention_hint", cls.abstention_hint),
            min_keep_aggregation=int(p.get(
                "grep_agent_min_keep_aggregation", cls.min_keep_aggregation)),
            include_pair=flag("grep_agent_include_pair", cls.include_pair),
        )

    def applies_to(self, categories: Categories, category: str | None) -> bool:
        """A layer restricted to `categories` runs only for those; None = all."""
        return categories is None or category in categories
