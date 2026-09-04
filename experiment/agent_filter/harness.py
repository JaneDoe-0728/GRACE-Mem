"""Agent Filter's entry point: one question in, one refined answer context out.

The flow this module orchestrates, and nothing else:

    prepare   the seed sids inside the Evidence Summary, the corpus behind them,
              and the prompt the agent starts from
        │
    search    the agent verifies candidates with GREP/READ/VECTOR and hunts down
              the evidence retrieval missed, until it replies FINAL
        │
    finalize  the selection policy decides what the answering model sees

Safety net at every step: if the agent fails, emits invalid output, or blows the
budget, the original context is handed back untouched. Each stage lives in its
own module -- protocol, loop, adjudication, finalization -- and this file is the
sequence they run in.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

from experiment.agent_filter.config import AgentFilterConfig
from experiment.agent_filter.context import (
    candidates_block,
    graph_context_from_context,
    seed_scores_from_context,
    seed_sids_from_context,
)
from experiment.agent_filter.corpus import Corpus, load_corpus
from experiment.agent_filter.finalization import EvidenceFinalizer
from experiment.agent_filter.llm_factory import agent_llm
from experiment.agent_filter.loop import AgentSession, AgentTools
from experiment.agent_filter.prompting.agent import (
    ABSTENTION_HINT,
    CATEGORY_HINTS,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    VECTOR_TOOL_BLOCK,
    active_hypothesis_block,
)

_FILTER_MODE_RULE = (
    "\nIMPORTANT: you may only KEEP or DROP candidates; do not add new sids in FINAL."
)


@dataclass
class _Preparation:
    """What one question looks like before the agent has said anything."""
    corpus: Corpus
    seed: list[str]
    category: str | None
    graph_context: str
    hint: str
    vector_ok: bool


def refine_context(
    *,
    question: str,
    context: str,
    csv_path: str | Path,
    llm,
    question_date: str | None = None,
    category: str | None = None,
    params: dict | None = None,
    artifact_dir: str | Path | None = None,
    corpus: Corpus | None = None,
) -> tuple[str, dict]:
    """Run the grep agent and return (refined_context, trace). Any failure falls
    back to the original context.
    The corpus may be prebuilt externally (e.g. a LoCoMo chunk-level corpus); when
    it is not supplied it is loaded from csv_path."""
    trace: dict = {"enabled": True, "mode": None, "commands": [], "fallback": None}
    try:
        # Inside the try, not before it: from_params does every coercion (int/float
        # on the vector knobs, flag()) and raises the INERT_PARAMS FutureWarning, so
        # one malformed param -- or a legacy key under -W error -- used to escape
        # this function and abort the QA run instead of falling back.
        config = AgentFilterConfig.from_params(params)
        trace["mode"] = config.mode
        prep = _prepare(
            question=question, context=context, csv_path=csv_path, category=category,
            corpus=corpus, artifact_dir=artifact_dir, config=config, trace=trace,
        )
        if not prep.seed and config.mode == "filter":
            trace["fallback"] = "no_seed"
            return context, trace

        session = _open_session(
            question=question, question_date=question_date, prep=prep,
            llm=llm, artifact_dir=artifact_dir, config=config, trace=trace,
        )
        final_raw = session.run(config.max_calls)
        if not (final_raw and prep.corpus.normalize_sids(final_raw)):
            # No FINAL / empty FINAL / unparseable sids -> prompt once for closure.
            # A second prompt is useless (measured: 129 of 138 times the model still
            # returns a tool call). Narrowing the context down to the verified hits
            # also tested negative (salvage group 54 -> 43%): a narrow but
            # "trustworthy" context actually tempts the answering model to invent,
            # whereas the noise of all 16 entries acts as cover on hard questions.
            if final_raw:
                trace["final_raw_unresolved"] = final_raw[:20]
            final_raw = session.force_final(prep.seed)
        if not (final_raw and prep.corpus.normalize_sids(final_raw)):
            return _without_a_final(
                context=context, prep=prep, session=session, config=config,
                trace=trace,
            )

        finalizer = EvidenceFinalizer(config)
        selection = finalizer.select(
            corpus=prep.corpus, final_raw=final_raw, seed=prep.seed,
            verified_sids=session.verified_sids,
            vector_candidate_sids=session.vector_candidate_sids,
            trace=trace,
        )
        if selection.fallback:
            return context, trace

        return finalizer.finalize(
            selection=selection, context=context, corpus=prep.corpus,
            question=question, question_date=question_date, category=prep.category,
            llm=llm, trace=trace,
        )

    except Exception:
        trace["fallback"] = "exception"
        trace["error"] = traceback.format_exc()[-2000:]
        return context, trace


def _prepare(
    *,
    question: str,
    context: str,
    csv_path: str | Path,
    category: str | None,
    corpus: Corpus | None,
    artifact_dir: str | Path | None,
    config: AgentFilterConfig,
    trace: dict,
) -> _Preparation:
    """Read the question, its context and its corpus into what the agent needs."""
    if corpus is None:
        corpus = load_corpus(csv_path)
    seed = seed_sids_from_context(context)
    trace["seed_sids"] = seed
    trace["seed_scores"] = seed_scores_from_context(context)  # sid -> rerank score
    graph_context = graph_context_from_context(
        context, max_chars=config.graph_context_max_chars,
    )
    trace["graph_context_available"] = bool(graph_context)
    trace["filter_graph_context"] = config.filter_include_graph
    trace["answer_graph_context"] = config.answer_include_graph

    if category is None:
        category = Path(csv_path).parent.name
    # _abs abstention questions (the answer is not in the corpus) are recorded
    # for the trace: every retention experiment has had to be read separately for
    # them, because narrowing an abstention context tempts the model to abandon
    # the abstention and answer.
    trace["is_abstention"] = bool(csv_path) and Path(csv_path).stem.endswith("_abs")

    # The skill library (driven by question shape) takes precedence; only on a
    # miss does it fall back to the category hint
    hint = ""
    if config.use_skills:
        from experiment.agent_filter.prompting.skills import select_skills
        matched = select_skills(question)
        trace["skills"] = [name for name, _ in matched]
        hint = "\n\n".join(strategy for _, strategy in matched)
    if not hint:
        hint = CATEGORY_HINTS.get(category, "")

    # VECTOR tool: enabled only when this question's summaries VDB is present
    # (the agent decides for itself when to search semantically -- unlike the
    # disproven gap-repair approach where the verifier pushed candidates, here
    # the agent pulls).
    vector_ok = (
        config.vector_search
        and artifact_dir is not None
        and (Path(artifact_dir) / "summaries_chroma").exists()
    )
    trace["vector_tool"] = vector_ok
    trace["evidence_provenance"] = {}
    return _Preparation(
        corpus=corpus, seed=seed, category=category, graph_context=graph_context,
        hint=hint, vector_ok=vector_ok,
    )


def _open_session(
    *,
    question: str,
    question_date: str | None,
    prep: _Preparation,
    llm,
    artifact_dir: str | Path | None,
    config: AgentFilterConfig,
    trace: dict,
) -> AgentSession:
    """Write the opening prompt and hand it to a session with its tools."""
    system = SYSTEM_PROMPT.format(
        max_calls=config.max_calls,
        vector_tool=VECTOR_TOOL_BLOCK if prep.vector_ok else "",
        hypothesis_line=active_hypothesis_block() if config.emit_hypothesis else "",
    )
    if config.mode == "filter":
        system += _FILTER_MODE_RULE
    user = USER_TEMPLATE.format(
        question=question,
        date_line=f"QUESTION DATE: {question_date}\n" if question_date else "",
        hint_line=f"{prep.hint}\n" if prep.hint else "",
        graph_context=(
            "GRAPH FACTS (Entities/Relationships; use as supporting evidence):\n"
            + prep.graph_context
            if config.filter_include_graph and prep.graph_context else ""
        ),
        candidates=candidates_block(prep.corpus, prep.seed),
    )
    return AgentSession(
        llm=llm,
        tools=AgentTools(
            prep.corpus,
            seed=prep.seed,
            artifact_dir=artifact_dir,
            grep_max_lines=config.grep_max_lines,
            vector_enabled=prep.vector_ok,
            vector_topn=config.vector_topn,
            vector_min_score=config.vector_min_score,
        ),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        trace=trace,
        emit_hypothesis=config.emit_hypothesis,
    )


def _without_a_final(
    *,
    context: str,
    prep: _Preparation,
    session: AgentSession,
    config: AgentFilterConfig,
    trace: dict,
) -> tuple[str, dict]:
    """The agent would not close. Hand back the full protective context.

    Narrowing here has been disproved twice: raising max-calls to force a
    submitted selection (-6), and rebuilding from the sids the agent did confirm
    -- a bare 1-2 sid context strips the answering model of the full-context
    noise cover and it starts inventing.

    Uncertainty signal: an agent refusing to FINAL means it found no confirmable
    evidence for an answer (_abs abstention questions fall back 70% of the time),
    which is itself the strongest evidence for abstention. The hint is attached
    only to this full-context path -- "narrowed context + hint" tested negative
    (ordinary questions 46.7 -> 33.3).
    """
    trace["fallback"] = "no_final"
    trace["verified_sids"] = prep.corpus.normalize_sids(list(session.verified_sids))
    if config.abstention_hint:
        trace["abstention_hint"] = True
        return context + ABSTENTION_HINT, trace
    return context, trace


def maybe_refine_context(
    *,
    question: str,
    context: str,
    csv_path: str | Path | None,
    llm,
    question_date: str | None = None,
    category: str | None = None,
    log_dir=None,
    artifact_dir: str | Path | None = None,
) -> str:
    """The single mount point in the qa_eval flow, shared by both the processor and
    rerun paths.
    A no-op when GREP_AGENT_PARAMS.use_grep_agent is off; any failure falls back to
    the original context."""
    from experiment.experiment_config import GREP_AGENT_PARAMS

    if not GREP_AGENT_PARAMS.get("use_grep_agent"):
        return context
    if not csv_path or not Path(csv_path).exists():
        print(f"[QA] Grep agent skipped: source csv not found ({csv_path})")
        return context

    print("[QA] Grep agent refining evidence...")
    refined, trace = refine_context(
        question=question,
        context=context,
        csv_path=csv_path,
        llm=agent_llm(llm),
        question_date=question_date,
        category=category,
        params=GREP_AGENT_PARAMS,
        artifact_dir=artifact_dir,
    )
    if trace.get("fallback"):
        print(f"[QA] Grep agent fallback: {trace['fallback']} (context unchanged)")
    else:
        print(
            f"[QA] Grep agent: kept={len(trace.get('kept', []))} "
            f"added={len(trace.get('added', []))} dropped={len(trace.get('dropped', []))} "
            f"({len(trace.get('commands', []))} tool calls)"
        )
    if log_dir is not None:
        try:
            from grace_mem.utils.analysis_log import append_analysis_record
            append_analysis_record(log_dir, "grep_agent", {"question": question, **trace})
        except Exception as exc:  # a logging failure must not affect answering
            print(f"[QA] Grep agent trace logging failed: {exc}")
    return refined
