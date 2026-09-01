"""From the agent's FINAL sid list to the context the answering model sees.

This is the evidence retention policy, and it exists in one copy. Splitting it
into select() and finalize() keeps the mode policy, the cap and the provenance
record separate from the recovery layer that runs on top of the agent's choice.

The order matters and is load-bearing:

    mode policy -> provenance -> cap -> adjudication
                -> kept/added/dropped -> rebuild

Every layer after the cap is additive: nothing here removes evidence the agent
chose, and if a layer leaves no upstream summary standing the original context
is handed back untouched.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from experiment.agent_filter.adjudication import adjudicate_candidates
from experiment.agent_filter.config import AgentFilterConfig
from experiment.agent_filter.context import append_fetched_evidence, rebuild_context
from experiment.agent_filter.corpus import Corpus


@dataclass
class Selection:
    """The agent's choice, normalized against the corpus and capped.

    A `fallback` means the run cannot produce a refined context and the caller
    should hand back the original; `final` is empty in that case.
    """
    final: list[str] = field(default_factory=list)
    seed_norm: list[str] = field(default_factory=list)
    fallback: str | None = None


class EvidenceFinalizer:
    def __init__(self, config: AgentFilterConfig):
        self.config = config

    def select(
        self,
        *,
        corpus: Corpus,
        final_raw,
        seed: list[str],
        verified_sids: set,
        vector_candidate_sids: set,
        trace: dict,
    ) -> Selection:
        """Apply the mode policy and the evidence cap, and record provenance."""
        seed_norm = corpus.normalize_sids(seed)
        if not (final_raw and corpus.normalize_sids(final_raw)):
            trace["fallback"] = "no_final"
            trace["verified_sids"] = corpus.normalize_sids(list(verified_sids))
            return Selection(seed_norm=seed_norm, fallback="no_final")

        final = corpus.normalize_sids(final_raw)
        if self.config.mode == "filter":
            final = [s for s in final if s in set(seed_norm)]
        elif self.config.mode == "fetch_only":
            # Add without cutting: keep the baseline context's serendipity while
            # taking the recall the agent digs up.
            # Measured on LoCoMo: the agent's all-gold-hit rate rose 19.8pp, but
            # cutting useful non-gold content cancelled the gain -- hence this mode
            # decouples the two.
            final = seed_norm + [s for s in final if s not in set(seed_norm)]

        # The provenance gate was removed on 2026-07-22: a VECTOR hit now counts as
        # verified just like GREP/READ (see AgentTools), and evidence pulled back by
        # GREP/READ/VECTOR is trusted across the board. The only unverified things
        # left are hallucinated sids, which rebuild_context drops naturally because
        # the corpus cannot resolve them.
        trace["verified_sids"] = corpus.normalize_sids(list(verified_sids))
        trace["vector_candidate_sids"] = corpus.normalize_sids(list(vector_candidate_sids))
        seed_set = set(seed_norm)
        trace["evidence_provenance"] = {
            s: (
                "seed+verified" if s in seed_set and s in verified_sids
                else "seed" if s in seed_set
                else "verified" if s in verified_sids
                else "unverified"
            )
            for s in final
        }
        trace["final_before_cap"] = list(final)  # pre-truncation, for diagnosing top-k truncation
        final = final[:self.config.max_sids]
        if not final:
            trace["fallback"] = "empty_final"
            return Selection(seed_norm=seed_norm, fallback="empty_final")
        return Selection(final=final, seed_norm=seed_norm)

    def finalize(
        self,
        *,
        selection: Selection,
        context: str,
        corpus: Corpus,
        question: str,
        question_date: str | None,
        category: str | None,
        llm,
        trace: dict,
    ) -> tuple[str, dict]:
        """Run the recovery layer, then render the context it agreed on."""
        final = self._adjudicate(
            selection, corpus=corpus, question=question,
            question_date=question_date, category=category, llm=llm, trace=trace,
        )

        seed_set = set(selection.seed_norm)
        trace["final_sids"] = final
        trace["kept"] = [s for s in final if s in seed_set]
        trace["added"] = [s for s in final if s not in seed_set]
        trace["dropped"] = [s for s in selection.seed_norm if s not in set(final)]

        # Safety net: if the selector dropped every upstream summary, keep the
        # original 16-summary context instead of answering from agent-fetched
        # evidence alone. This is distinct from a no_final/exception fallback:
        # the agent completed, but produced zero retained seeds.
        if not trace["kept"]:
            trace["fallback"] = "zero_keep"
            trace["context_sids"] = selection.seed_norm
            return context, trace

        if self.config.mode == "fetch_only":
            if not trace["added"]:
                trace["fallback"] = "no_addition"
                return context, trace
            refined, context_sids = append_fetched_evidence(
                context, corpus, trace["added"], selection.seed_norm,
            )
            trace["context_sids"] = context_sids
            return refined, trace

        refined, context_sids = rebuild_context(
            context, corpus, final,
            include_pair=self.config.include_pair,
            include_prefix=self.config.answer_include_graph,
        )
        trace["context_sids"] = context_sids
        return refined, trace

    # ── the recovery layer ──────────────────────────────────────────────
    def _adjudicate(
        self, selection: Selection, *, corpus: Corpus, question: str,
        question_date: str | None, category: str | None, llm, trace: dict,
    ) -> list[str]:
        """Add back the discarded seeds an answer-blind auditor rules relevant.

        The agent's FINAL is an "answer citation" -- solve first, then keep only
        the smallest turn set containing the answer span -- so supporting
        evidence is discarded systematically. The auditor cannot see the agent's
        conversation, and so does not know the answer it reached; it judges
        topical relevance alone. Additive only: the agent's own 0.84-precision
        picks are never touched.
        """
        cfg = self.config
        final = list(selection.final)
        if not cfg.adjudicate or len(final) >= cfg.max_sids:
            return final
        if not cfg.applies_to(cfg.adjudicate_categories, category):
            return final

        pending = [s for s in selection.seed_norm if s not in set(final)]
        if not pending:
            return final

        started = time.perf_counter()
        try:
            kept, verdicts = adjudicate_candidates(
                llm, question=question, question_date=question_date,
                corpus=corpus, pending=pending,
            )
            verdicts["ms"] = round((time.perf_counter() - started) * 1000)
            trace["adjudication"] = verdicts
            return (final + [s for s in kept if s not in set(final)])[:cfg.max_sids]
        except Exception as exc:  # an adjudication crash must not disturb the main flow
            trace["adjudication"] = {"error": str(exc)[:200]}
            return final
