"""Is the selected evidence enough to answer the question at all?

An independent audit call, blind to the agent's search history: it sees only
the question and the evidence the agent settled on. Its verdict is advisory --
it can send the agent back out for another round, but it can never remove
evidence, and a flaky verdict must not send the loop spinning.
"""
from __future__ import annotations

import re

from experiment.agent_filter import vector_search
from experiment.agent_filter.config import AgentFilterConfig
from experiment.agent_filter.corpus import Corpus
from experiment.agent_filter.llm_factory import verify_llm
from experiment.agent_filter.loop import AgentSession
from experiment.agent_filter.prompting.verification import (
    GAP_HINT_TEMPLATE,
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


class SufficiencyRepairer:
    """Send the agent back out for what the verifier says is missing.

    Additive and monotonic by construction: a round may only add sids to the
    selection, and it stops at the first round that is ruled sufficient, adds
    nothing, or fails. A verifier crash ends the repair rather than the run.
    """

    def __init__(self, *, llm, corpus: Corpus, config: AgentFilterConfig, artifact_dir=None):
        self.llm = llm
        self.corpus = corpus
        self.config = config
        self.artifact_dir = artifact_dir

    def repair(
        self,
        *,
        session: AgentSession,
        question: str,
        question_date: str | None,
        category: str | None,
        seed: list[str],
        selected: list[str],
        trace: dict,
    ) -> list[str]:
        cfg = self.config
        # Selective: skip non-aggregation categories, where verify only dilutes.
        rounds = cfg.verify_rounds if cfg.applies_to(cfg.verify_categories, category) else 0
        final = list(selected)
        trace["sufficiency"] = []
        for round_index in range(rounds):
            try:
                sufficient, missing = check_sufficiency(
                    verify_llm(self.llm), question=question, question_date=question_date,
                    corpus=self.corpus, sids=final,
                )
            except Exception as exc:  # a verifier crash must not disturb the main flow
                trace["sufficiency"].append({"round": round_index, "error": str(exc)[:200]})
                break
            trace["sufficiency"].append(
                {"round": round_index, "sufficient": sufficient, "missing": missing[:300]}
            )
            if sufficient:
                break

            session.tell(self._gap_message(question, missing, final, trace))
            extra_raw = session.run(cfg.verify_max_calls)
            if not extra_raw:
                break
            extra = self.corpus.normalize_sids(extra_raw)
            if cfg.mode == "filter":
                extra = [s for s in extra if s in set(self.corpus.normalize_sids(seed))]
            # Monotonic: a verify round may only add, never touch what is chosen
            added_now = [s for s in extra if s not in set(final)]
            trace["sufficiency"][-1]["added"] = added_now
            if not added_now:
                break
            final = (final + added_now)[:cfg.max_sids]
        return final

    def _gap_message(self, question: str, missing: str, final: list[str], trace: dict) -> str:
        """The gap hint, plus the semantic neighbours GREP alone would not reach.

        The grep repair arm comes back empty roughly 87% of the time because of
        the paraphrase gap, so the gap description is embedded and searched
        against the summaries VDB, and the neighbours are handed to the agent to
        confirm.
        """
        message = GAP_HINT_TEMPLATE.format(missing=missing)
        if self.artifact_dir is None or self.config.gap_vector_topn <= 0:
            return message

        candidates = vector_search.search_summaries(
            self.artifact_dir,
            f"{question}\n{missing}",
            exclude=set(final),
            topn=self.config.gap_vector_topn,
            min_score=self.config.gap_vector_min_score,
        )
        trace["sufficiency"][-1]["vector_cands"] = [sid for sid, _ in candidates]
        if not candidates:
            return message
        lines = []
        for sid, score in candidates:
            entry = self.corpus.display_entry(sid, max_chars=200) or "(text unavailable)"
            lines.append(f"[sid={sid}] (score={score:.2f}) {entry}")
        return message + (
            "\n\nA semantic search for the missing information surfaced these "
            "candidate turns (NOT yet verified — check with READ/GREP before "
            "including):\n" + "\n".join(lines)
        )
