"""Where the benchmark harness mounts Agent Filter.

`grace_mem.agent_filter` is the mechanism and takes its settings as an argument.
This module is the one place that knows those settings live in
`experiment_config.GREP_AGENT_PARAMS`, which is what keeps the core free of any
import back into the benchmark harness.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from grace_mem.agent_filter.harness import refine_context
from grace_mem.agent_filter.runtime.llm import agent_llm

if TYPE_CHECKING:
    from grace_mem.agent_filter.retrieval.corpus import Corpus


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
    corpus: Corpus | None = None,
) -> str:
    """The single mount point in the qa_eval flow, shared by every benchmark.

    A no-op when GREP_AGENT_PARAMS.use_grep_agent is off; any failure falls back to
    the original context.

    The corpus arrives one of two ways. LongMem has a per-question script_data CSV
    and passes `csv_path`. LoCoMo has no such CSV -- its units are chunks cut out
    of locomo10.json -- so it builds the corpus itself and passes it directly.
    """
    from experiment.experiment_config import GREP_AGENT_PARAMS

    if not GREP_AGENT_PARAMS.get("use_grep_agent"):
        return context
    if corpus is None and (not csv_path or not Path(csv_path).exists()):
        print(f"[QA] Grep agent skipped: source csv not found ({csv_path})")
        return context

    print("[QA] Grep agent refining evidence...")
    refined, trace = refine_context(
        question=question,
        context=context,
        csv_path=csv_path or "",
        corpus=corpus,
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
