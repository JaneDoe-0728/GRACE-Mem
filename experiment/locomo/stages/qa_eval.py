"""QA evaluation stage: ask each question and record how the answer was reached.

Produces one row per question -- the answer, the retrieved context, latency,
and the trace of what retrieval did -- which is the input to both judging and
error analysis.

The module holds mutable global state (`retriever`, `retrieval_mode`, the
`_gold_*` and `_replay_*` maps) that workers set before calling
`evaluate_items`. That is unusual and worth knowing about: it exists because
this file doubles as a standalone script and as a stage the worker drives, and
the ablation modes below need to reach deep into evaluation without threading a
config through every call. It also means one process can only be in one
retrieval mode at a time.

The retrieval modes are the ablation surface, each isolating one contribution:

    gold_summary_only                 skip retrieval, feed the gold session
                                      summaries -- an upper bound on what
                                      perfect summary retrieval could achieve
    gold_raw_text_only                same, but raw text, which separates the
                                      summarizer's contribution from retrieval's
    replay_summary_raw_text_from_run  reuse a prior run's retrieved summary_ids
                                      but substitute raw text, holding retrieval
                                      fixed while varying what it returns
    replay_summary_fact_from_run      same, with facts extracted from that text

The replay modes are what make comparisons fair: they hold the retrieved set
constant across variants, so a difference in accuracy cannot be attributed to
retrieval having found different things.
"""

import re
import csv
import json
import time
import argparse
import pandas as pd
import sys
import os
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

try:
    from grace_mem.storage import MGR
    from grace_mem.pipeline.retrieval_steps import TemporalRelevanceCalculator
    from grace_mem.utils.query_time_parser import detect_and_parse_time_expressions
    from grace_mem.utils.temporal import time_rewrite_ablation_enabled
except Exception as e:
    raise RuntimeError(
        "Failed to import GRACE-Mem modules. Ensure PYTHONPATH includes your project root. Original error: %r" % (e,)
    )

from experiment.locomo.helpers.llm import llm_post, llm_post_json
from experiment.locomo.utils.error_analysis import append_analysis_record, compact_json, derive_drop_reasons
from experiment.locomo.utils.io import EVAL_COLUMNS

# retriever is set by the caller (e.g. locomo_pipeline.py) after build_pipeline().
# When running qa_eval.py as a standalone script, set it via build_pipeline() in main().
retriever = None

# Ablation state: set by workers before calling evaluate_items().
# retrieval_mode="gold_summary_only"              → use _gold_session_summaries
# retrieval_mode="gold_raw_text_only"             → use _gold_session_raw_texts
# retrieval_mode="replay_summary_raw_text_from_run" → reuse prior summary_ids, replace with raw session text
# retrieval_mode="replay_summary_fact_from_run"   → reuse prior summary_ids, extract facts from raw session text
retrieval_mode: str = ""
_gold_session_summaries: dict = {}  # {"session_1_summary": "<text>", ...}
_gold_session_raw_texts: dict = {}  # {"session_1": [{"speaker":..,"text":..}, ...], ...}
_replay_question_context: dict[str, dict[str, Any]] = {}
_replay_entity_meta_by_name: dict[str, dict[str, Any]] = {}
_replay_relationship_meta_by_label: dict[str, dict[str, Any]] = {}

_TEMPORAL_TYPES = {"Date", "Event", "Activity"}

# ---------------------------------------------------------------------------
# Fact extraction prompt used by concise replay modes.
# ---------------------------------------------------------------------------

_FACT_EXTRACTION_SYSTEM_PROMPT = """
Extract SIGNIFICANT facts from text. Be selective, but preserve details that are likely to answer future questions.

LANGUAGE REQUIREMENT:
Detect the input language. All extracted facts, names, descriptions, and output must be in the SAME language as the input. Do not translate.

Extract ONLY "world" and "assistant" type facts.

══════════════════════════════════════════════════════════════════════════
WHAT TO EXTRACT
══════════════════════════════════════════════════════════════════════════

Extract facts worth remembering long-term, including:

✅ Personal info: names, relationships, roles, background
✅ Long-term preferences, habits, interests, favorites
✅ Significant events, milestones, decisions, achievements, changes
✅ Plans, goals, deadlines, commitments
✅ Expertise, skills, certifications, experience
✅ Important context: projects, problems, constraints
✅ Reasons, motivations, lessons learned, takeaways, emotional significance
✅ QA-critical details: exact names, titles, authors, pet names, organizations, programs, dates, places, and event names

Do NOT extract greetings, filler, process chatter, repeated info, or trivial one-off details.

Preserve answerable details even if they look small. For example, book titles, authors, pet names, favorite media, reasons, motivations, and lessons learned should be kept if they help answer future questions.

══════════════════════════════════════════════════════════════════════════
ATTRIBUTION AND CONSOLIDATION
══════════════════════════════════════════════════════════════════════════

Speaker attribution is critical.

A fact must be attributed to the person who experienced, owns, likes, believes, plans, said, or felt it.
Do NOT transfer a preference, feeling, lesson, motivation, or experience from one speaker to another.
If one person asks and another answers, the answer usually belongs to the answering speaker.

Resolve references when possible:

* "my roommate" + "Emily" → "Emily (user's roommate)"
* "my dog" + "Luna" → "Luna (user's dog)"

Consolidate related statements only when no answerable detail is lost.
Do NOT consolidate if it would lose:

* who did, said, felt, liked, planned, or experienced something
* names, titles, dates, places, or event names
* reasons, motivations, lessons learned, or takeaways
* details likely to be queried later

Do NOT merge facts across speakers unless the text clearly says they refer to the same real-world item or event.

══════════════════════════════════════════════════════════════════════════
FACT KINDS AND TYPES
══════════════════════════════════════════════════════════════════════════

fact_kind:

* "event": a specific datable occurrence or action
* "conversation": a stable or ongoing fact, including preferences, habits, traits, beliefs, goals, relationships, background, motivations, and important context

fact_type:

* "world": user's life, other people, external events, ordinary conversations between people
* "assistant": interactions with the AI assistant, including user requests, assistant recommendations, assistant help, or user feedback about the assistant

Only use fact_type="assistant" when the text is explicitly about an interaction between the user and the AI assistant.
For ordinary conversations between people, use fact_type="world" even if one person gives advice, encouragement, or support.

══════════════════════════════════════════════════════════════════════════
OUTPUT RULES
══════════════════════════════════════════════════════════════════════════

Each fact must include:

* what: concise core fact, 1-2 sentences max
* when: temporal info if mentioned, otherwise "N/A"
* where: location if relevant, otherwise "N/A"
* who: people involved and relationships, otherwise "N/A"
* why: motivation, cause, significance, emotional reason, or purpose if mentioned; otherwise "N/A"
* fact_kind: "event" or "conversation"
* fact_type: "world" or "assistant"
* occurred_start: date for event facts only; null for conversation facts
* occurred_end: date for event facts only; null for conversation facts

Use "Event Date" as the reference for relative dates.
"Yesterday", "last week", and similar expressions are relative to Event Date, not today.

For event facts:

* set occurred_start and occurred_end when a date is available
* use the same date for occurred_start and occurred_end if the event happened on one day

For conversation facts:

* keep occurred_start and occurred_end as null

Return only valid JSON.
Do not include explanations, markdown, or extra text.

══════════════════════════════════════════════════════════════════════════
EXAMPLES
══════════════════════════════════════════════════════════════════════════

Example 1:
Input:
"Hey! I'm planning my wedding - want a small outdoor ceremony. Just got back from Emily's wedding, she married Sarah at a rooftop garden. It was nice weather. I grabbed a coffee."

Output:
{
"facts": [
{
"what": "User is planning a wedding and wants a small outdoor ceremony.",
"when": "N/A",
"where": "N/A",
"who": "user",
"why": "N/A",
"fact_kind": "conversation",
"fact_type": "world",
"occurred_start": null,
"occurred_end": null
},
{
"what": "Emily married Sarah at a rooftop garden.",
"when": "N/A",
"where": "rooftop garden",
"who": "Emily, Sarah",
"why": "N/A",
"fact_kind": "event",
"fact_type": "world",
"occurred_start": null,
"occurred_end": null
}
]
}

Example 2:
Input:
"Alice has 5 years of Kubernetes experience and holds CKA certification. She's been leading the infrastructure team since March. By the way, she prefers dark roast coffee."

Output:
{
"facts": [
{
"what": "Alice has 5 years of Kubernetes experience and holds CKA certification.",
"when": "N/A",
"where": "N/A",
"who": "Alice",
"why": "N/A",
"fact_kind": "conversation",
"fact_type": "world",
"occurred_start": null,
"occurred_end": null
},
{
"what": "Alice has led the infrastructure team since March.",
"when": "since March",
"where": "N/A",
"who": "Alice",
"why": "N/A",
"fact_kind": "conversation",
"fact_type": "world",
"occurred_start": null,
"occurred_end": null
},
{
"what": "Alice prefers dark roast coffee.",
"when": "N/A",
"where": "N/A",
"who": "Alice",
"why": "N/A",
"fact_kind": "conversation",
"fact_type": "world",
"occurred_start": null,
"occurred_end": null
}
]
}

Example 3:
Input:
"Caroline: I loved Becoming Nicole by Amy Ellis Nutt. It is about a trans girl and her family. It taught me self-acceptance and the importance of finding support.
Melanie: That sounds inspiring."

Output:
{
"facts": [
{
"what": "Caroline loved 'Becoming Nicole' by Amy Ellis Nutt.",
"when": "N/A",
"where": "N/A",
"who": "Caroline",
"why": "The book is important to Caroline.",
"fact_kind": "conversation",
"fact_type": "world",
"occurred_start": null,
"occurred_end": null
},
{
"what": "'Becoming Nicole' is about a trans girl and her family.",
"when": "N/A",
"where": "N/A",
"who": "Caroline",
"why": "Explains why the book resonated with Caroline.",
"fact_kind": "conversation",
"fact_type": "world",
"occurred_start": null,
"occurred_end": null
},
{
"what": "Caroline learned self-acceptance and the importance of finding support from 'Becoming Nicole'.",
"when": "N/A",
"where": "N/A",
"who": "Caroline",
"why": "Captures Caroline's takeaway from the book.",
"fact_kind": "conversation",
"fact_type": "world",
"occurred_start": null,
"occurred_end": null
}
]
}

Respond with:
{"facts": [{"what": "...", "when": "...", "where": "...", "who": "...", "why": "...", "fact_kind": "event|conversation", "fact_type": "world|assistant", "occurred_start": null, "occurred_end": null}]}
"""



_FACT_CHUNK_SIZE = 2500  # chars; sessions longer than this are split
_fact_cache: dict[str, list[str]] = {}  # session_id → extracted facts (persists across questions)


def _split_at_sentence(text: str, max_chars: int) -> list[str]:
    """Split text into balanced chunks, each at most max_chars, breaking at sentence boundaries.

    Instead of greedy splitting (fill each chunk to max), we first compute the number of
    chunks needed, then divide the text evenly and snap each cut point to the nearest
    sentence boundary (newline preferred, then '. '/'! '/'? ').
    """
    import math

    if len(text) <= max_chars:
        return [text]

    n_chunks = math.ceil(len(text) / max_chars)
    target = len(text) // n_chunks  # ideal chars per chunk

    def find_cut(remaining: str, ideal: int) -> int:
        # Search in a window around the ideal cut point, never exceeding max_chars
        """Find the sentence boundary nearest a target offset.

        Cuts are snapped to a boundary so a chunk never ends mid-sentence, which
        would leave the fact extractor working on a fragment.
        """
        lo = max(ideal - ideal // 4, 1)
        hi = min(ideal + ideal // 4, max_chars, len(remaining) - 1)
        window = remaining[lo:hi]
        # Prefer newline (dialogue turn boundaries)
        pos = window.rfind("\n")
        if pos >= 0:
            return lo + pos + 1
        for punct in (". ", "! ", "? "):
            pos = window.rfind(punct)
            if pos >= 0:
                return lo + pos + 2  # include punctuation + space
        # No boundary found in window — fall back to ideal (capped at max_chars)
        return min(ideal, max_chars)

    chunks: list[str] = []
    remaining = text
    for _ in range(n_chunks - 1):
        cut = find_cut(remaining, target)
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
        # Recompute target for the remaining text and remaining chunk count
        left = n_chunks - len(chunks)
        target = len(remaining) // left if left > 0 else len(remaining)
    if remaining:
        chunks.append(remaining)
    return chunks


def _extract_facts_from_chunk(chunk: str, chunk_idx: int, total_chunks: int,
                               event_date_str: str, event_date_iso: str) -> list[str]:
    """Extract salient facts from one chunk of raw session text.

    Used by the concise replay ablations, which substitute extracted facts for
    raw text while holding the retrieved set fixed -- isolating how much the
    verbosity of the evidence costs, separately from what was retrieved.

    Falls back to returning the chunk unchanged on any failure, so an
    extraction problem degrades the ablation to the raw-text condition rather
    than losing the question.
    """
    import json as _json
    user_message = (
        f"Extract facts from the following text chunk.\n\n"
        f"Chunk: {chunk_idx}/{total_chunks}\n"
        f"Event Date: {event_date_str} ({event_date_iso})\n"
        f"Context: none\n\n"
        f"Text:\n{chunk}"
    )
    raw_response = llm_post(
        [
            {"role": "system", "content": _FACT_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=2048,
    )
    start = raw_response.find("{")
    end = raw_response.rfind("}") + 1
    if start == -1 or end == 0:
        return []
    result = _json.loads(raw_response[start:end])
    facts_raw = result.get("facts") or []
    rendered: list[str] = []
    for f in facts_raw:
        if not isinstance(f, dict):
            continue
        what = (f.get("what") or "").strip()
        if not what:
            continue
        parts = [what]
        when = (f.get("when") or "").strip()
        if when and when.upper() != "N/A":
            parts.append(f"When: {when}")
        who = (f.get("who") or "").strip()
        if who and who.upper() != "N/A":
            parts.append(f"Involving: {who}")
        why = (f.get("why") or "").strip()
        if why and why.upper() != "N/A":
            parts.append(why)
        rendered.append(" | ".join(parts))
    return rendered


def _extract_facts_for_evidence(raw_text: str, date_time: str | None) -> list[str]:
    """
    Run fact extraction on raw session text.
    Long sessions are split at sentence boundaries and extracted chunk-by-chunk.
    Falls back to [raw_text] on any error.
    """
    from datetime import datetime

    if date_time:
        try:
            event_dt = datetime.fromisoformat(date_time)
            event_date_str = event_dt.strftime("%A, %B %d, %Y")
            event_date_iso = event_dt.isoformat()
        except ValueError:
            event_date_str = date_time
            event_date_iso = date_time
    else:
        now = datetime.utcnow()
        event_date_str = now.strftime("%A, %B %d, %Y")
        event_date_iso = now.isoformat()

    try:
        chunks = _split_at_sentence(raw_text, _FACT_CHUNK_SIZE)
        total = len(chunks)
        all_facts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            all_facts.extend(_extract_facts_from_chunk(
                chunk, i, total, event_date_str, event_date_iso
            ))
        return all_facts if all_facts else [raw_text]
    except Exception:
        return [raw_text]

from experiment.locomo.helpers.dataset import default_output_stem, load_qa_items, normalize_dataset_name, resolve_dataset_path

try:
    from experiment.experiment_config import RETRIEVAL_PARAMS, RERANKER_PARAMS
except Exception:
    RETRIEVAL_PARAMS = {
        "ent_topk": 20,
        "rel_topk": 10,
        "ent_threshold": 0.2,
        "rel_threshold": 0.2,
        "filter_ent_topk": 10,
        "filter_rel_topk": 10,
        "filter_ent_threshold": 0.4,
        "filter_rel_threshold": 0.4,
        "summary_topk_per_item": 6,
        "summary_vec_threshold": 0.4,
    }
    RERANKER_PARAMS = {
        "use_reranker": True,
        "reranker_threshold": -3.0,
        "reranker_topk": 3,
    }

def scrub(piece: str) -> str:
    """Remove provider-specific tags/noise if any."""
    if not piece:
        return ""
    s = str(piece)
    import re
    _TAG_RE = re.compile(r"<\|\s*[^>|]+\|>")
    _END_TOKENS = ("<|end|>", "<|stop|>", "<|im_end|>")
    s = _TAG_RE.sub("", s)
    s = s.replace("<think>", "").replace("</think>", "")
    for t in _END_TOKENS:
        s = s.replace(t, "")
    toks = s.strip()
    if toks in {"analysis", "final", "message"}:
        return ""
    return toks


def gold_summary_answer(item: dict) -> dict:
    """Answer using only gold session summaries — no KG retrieval (ablation mode).

    Session IDs are parsed from D{N}: patterns in the evidence list (locomo style).
    When no D-patterns are found (locomo-plus), all available summaries are concatenated.
    """
    evidence_list = item.get("evidence", [])
    session_ids: set[int] = set()
    for evi in evidence_list:
        for m in re.findall(r"D(\d+):", str(evi)):
            session_ids.add(int(m))

    warnings: list[str] = []
    summaries: list[str] = []
    if session_ids:
        for sid in sorted(session_ids):
            key = f"session_{sid}_summary"
            text = _gold_session_summaries.get(key, "")
            if text:
                summaries.append(text)
            else:
                warnings.append("missing_gold_session_summary")
        if len(summaries) > 1:
            warnings.append("multi_gold_session_summary_concat")
    else:
        # locomo-plus: evidence is raw text with no D{N}: markers — use all summaries.
        all_texts = [_gold_session_summaries[k] for k in sorted(_gold_session_summaries) if _gold_session_summaries.get(k)]
        if all_texts:
            summaries = all_texts
            if len(summaries) > 1:
                warnings.append("multi_gold_session_summary_concat")
        else:
            warnings.append("missing_gold_session_summary")

    sep = "\n\n---\n\n"
    if summaries:
        context = sep.join(f"[Gold Session Summary]\n{s}" for s in summaries)
    else:
        context = "(no gold session summary available)"

    print(context)

    question = item.get("question", "")
    messages = [
        {"role": "system", "content": f"---Retrieved Context---\n{context}\n------------------"},
        {"role": "user", "content": (
            "Please answer based on the retrieved knowledge graph context above. "
            "Be concise and accurate.\n\n"
            f"Question: {question}\n\nAnswer:"
        )},
    ]
    t0 = time.time()
    answer_raw = llm_post(messages, temperature=0.0, max_tokens=1024)
    elapsed = time.time() - t0
    answer = scrub(answer_raw) or "(no assistant reply)"

    gold_ids_list = sorted(session_ids) if session_ids else []
    print(f"[GOLD_SUMMARY] {question[:60]!r} -> {answer[:80]!r} ({elapsed:.2f}s)")

    log_dir = Path(os.environ.get("KG_TRACE_PRETTY_LOG_DIR", "logs"))
    append_analysis_record(log_dir, "retrieval_summary", {
        "question": question,
        "request_id": None,
        "stop_reason": "gold_summary_only",
        "retrieval_mode": "gold_summary_only",
        "gold_session_ids": gold_ids_list,
        "gold_summary_context_chars": len(context),
        "warnings": warnings,
        "latency_sec": round(elapsed, 3),
        "answer": answer,
    })

    request_id = f"gold_summary_only:{','.join(str(s) for s in gold_ids_list)}" if gold_ids_list else "gold_summary_only"
    return {
        "answer": answer,
        "retrieved_context": context,
        "evidence": context,
        "latency_sec": round(elapsed, 3),
        "retrieval_request_id": request_id,
        "retrieval_stop_reason": "gold_summary_only",
        "retrieval_failure_type": ",".join(warnings) if warnings else "",
        "retrieval_confidence": None,
        "retrieval_tau": None,
        "selected_evidence_count": len(summaries),
        "selected_evidence_ids": compact_json(gold_ids_list),
        "selected_evidence_preview": compact_json([s[:100] for s in summaries[:3]]),
        "final_entity_names": compact_json([]),
        "final_relationship_names": compact_json([]),
        "anomaly_flags": compact_json(warnings),
        "pass2_triggered": False,
        "pass1_entity_ids": "[]",
        "pass2_entity_ids": "[]",
        "pass1_relation_ids": "[]",
        "pass2_relation_ids": "[]",
        "entity_overlap_count": 0,
        "relation_overlap_count": 0,
        "entity_overlap_pct": None,
        "relation_overlap_pct": None,
    }


def gold_raw_text_answer(item: dict) -> dict:
    """Answer using only gold session raw conversation turns — no KG retrieval (ablation mode).

    Session IDs are parsed from D{N}: patterns in the evidence list (locomo style).
    When no D-patterns are found (locomo-plus), all available session texts are concatenated.
    """
    evidence_list = item.get("evidence", [])
    session_ids: set[int] = set()
    for evi in evidence_list:
        for m in re.findall(r"D(\d+):", str(evi)):
            session_ids.add(int(m))

    warnings: list[str] = []
    texts: list[str] = []
    if session_ids:
        for sid in sorted(session_ids):
            key = f"session_{sid}"
            turns = _gold_session_raw_texts.get(key, [])
            if turns:
                formatted = "\n".join(f"{t.get('speaker','?')}: {t.get('text','')}" for t in turns)
                texts.append(formatted)
            else:
                warnings.append("missing_gold_session_raw_text")
        if len(texts) > 1:
            warnings.append("multi_gold_session_raw_text_concat")
    else:
        # locomo-plus: no D{N}: markers — use all sessions.
        all_keys = sorted(k for k in _gold_session_raw_texts if not k.endswith("_date_time"))
        for key in all_keys:
            turns = _gold_session_raw_texts.get(key, [])
            if isinstance(turns, list) and turns:
                formatted = "\n".join(f"{t.get('speaker','?')}: {t.get('text','')}" for t in turns)
                texts.append(formatted)
        if texts:
            if len(texts) > 1:
                warnings.append("multi_gold_session_raw_text_concat")
        else:
            warnings.append("missing_gold_session_raw_text")

    sep = "\n\n---\n\n"
    if texts:
        context = sep.join(f"[Gold Session Raw Text]\n{t}" for t in texts)
    else:
        context = "(no gold session raw text available)"

    question = item.get("question", "")
    messages = [
        {"role": "system", "content": f"---Retrieved Context---\n{context}\n------------------"},
        {"role": "user", "content": (
            "Please answer based on the retrieved knowledge graph context above. "
            "Be concise and accurate.\n\n"
            f"Question: {question}\n\nAnswer:"
        )},
    ]
    t0 = time.time()
    answer_raw = llm_post(messages, temperature=0.0, max_tokens=1024)
    elapsed = time.time() - t0
    answer = scrub(answer_raw) or "(no assistant reply)"

    gold_ids_list = sorted(session_ids) if session_ids else []
    print(f"[GOLD_RAW_TEXT] {question[:60]!r} -> {answer[:80]!r} ({elapsed:.2f}s)")

    log_dir = Path(os.environ.get("KG_TRACE_PRETTY_LOG_DIR", "logs"))
    append_analysis_record(log_dir, "retrieval_summary", {
        "question": question,
        "request_id": None,
        "stop_reason": "gold_raw_text_only",
        "retrieval_mode": "gold_raw_text_only",
        "gold_session_ids": gold_ids_list,
        "gold_raw_text_context_chars": len(context),
        "warnings": warnings,
        "latency_sec": round(elapsed, 3),
        "answer": answer,
    })

    request_id = f"gold_raw_text_only:{','.join(str(s) for s in gold_ids_list)}" if gold_ids_list else "gold_raw_text_only"
    return {
        "answer": answer,
        "retrieved_context": context,
        "evidence": context,
        "latency_sec": round(elapsed, 3),
        "retrieval_request_id": request_id,
        "retrieval_stop_reason": "gold_raw_text_only",
        "retrieval_failure_type": ",".join(warnings) if warnings else "",
        "retrieval_confidence": None,
        "retrieval_tau": None,
        "selected_evidence_count": len(texts),
        "selected_evidence_ids": compact_json(gold_ids_list),
        "selected_evidence_preview": compact_json([t[:100] for t in texts[:3]]),
        "final_entity_names": compact_json([]),
        "final_relationship_names": compact_json([]),
        "anomaly_flags": compact_json(warnings),
        "pass2_triggered": False,
        "pass1_entity_ids": "[]",
        "pass2_entity_ids": "[]",
        "pass1_relation_ids": "[]",
        "pass2_relation_ids": "[]",
        "entity_overlap_count": 0,
        "relation_overlap_count": 0,
        "entity_overlap_pct": None,
        "relation_overlap_pct": None,
    }


def _normalize_replay_question_key(question: str) -> str:
    return str(question or "").strip()


def _parse_replay_summary_id(summary_id: str) -> tuple[str | None, str | None]:
    match = re.match(r"^(?:\d+__)?(\d+):(\d+)$", str(summary_id or "").strip())
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _render_session_raw_text(session_id: str) -> tuple[str | None, str | None]:
    """Render a session's turns as the speaker-prefixed text the model sees.

    Speaker prefixes are kept because many LoCoMo questions turn on who said
    something, and stripped of them the evidence cannot answer those.
    """
    key = f"session_{session_id}"
    turns = _gold_session_raw_texts.get(key, [])
    if not isinstance(turns, list) or not turns:
        return None, None
    date_time = _gold_session_raw_texts.get(f"{key}_date_time")
    date_time_str = str(date_time or "").strip() or None
    lines = []
    # Ablation G: disable query-side time rewriting entirely
    rewrite_enabled = not time_rewrite_ablation_enabled()
    for t in turns:
        text = t.get("text", "")
        if rewrite_enabled and date_time_str and text:
            # Resolve relative time expressions ("yesterday", "last week", …) to absolute
            # dates using the session timestamp as the reference, so the model does not
            # confuse "when the conversation happened" with "when the event happened".
            try:
                text, _ = detect_and_parse_time_expressions(text, query_time=date_time_str, rewrite_query=True)
            except Exception:
                pass
        lines.append(f"{t.get('speaker', '?')}: {text}")
    rendered = "\n".join(lines)
    return rendered, date_time_str


def _parse_replay_relationship_label(label: str) -> tuple[str, str, str]:
    """Parse a "source -> target" relationship label back into its endpoints.

    The inverse of the renderer in summary_scoring. Replay depends on the pair
    round-tripping, so a change to either side breaks replay against artifacts
    already on disk.
    """
    text = str(label or "").strip()
    if not text:
        return "", "", ""
    head, sep, tail = text.partition(" | ")
    left, arrow, right = head.partition(" -> ")
    if not arrow:
        return text, "", tail.strip() if sep else ""
    return left.strip(), right.strip(), tail.strip() if sep else ""


def _render_replay_temporal_tag(item_type: str, prov: Any, request_id: str | None) -> str:
    dt_str, _ = TemporalRelevanceCalculator.get_newest_dialogue_datetime(prov or {}, request_id)
    return f" [t:{dt_str}]" if dt_str and item_type in _TEMPORAL_TYPES else ""


def _build_replay_entities(entity_names: list[Any]) -> list[dict[str, Any]]:
    """Reconstruct entity metadata from a previous run's recorded names."""
    entities: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for entity_name in entity_names:
        name = str(entity_name or "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        meta = _replay_entity_meta_by_name.get(name, {}) or {}
        entities.append({
            "name": name,
            "type": str(meta.get("type") or "").strip(),
            "desc": str(meta.get("description") or "").strip(),
            "prov": meta.get("prov") or {},
        })
    return entities


def _build_replay_relationships(relationship_names: list[Any]) -> list[dict[str, Any]]:
    """Reconstruct relationship metadata from a previous run's recorded labels."""
    relationships: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for relationship_name in relationship_names:
        label = str(relationship_name or "").strip()
        if not label or label in seen_labels:
            continue
        seen_labels.add(label)
        meta = _replay_relationship_meta_by_label.get(label, {}) or {}
        src_name, tgt_name, rel_desc = _parse_replay_relationship_label(label)
        relationships.append({
            "source_name": str(meta.get("source_entity") or src_name).strip(),
            "target_name": str(meta.get("target_entity") or tgt_name).strip(),
            "rel_desc": str(meta.get("description") or rel_desc).strip(),
            "type": str(meta.get("type") or "").strip(),
            "prov": meta.get("prov") or {},
        })
    return relationships


def _render_replay_context_text(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    *,
    request_id: str | None,
) -> str:
    """Render a replayed context so it is byte-comparable with the original run.

    Replay ablations only mean something if the two contexts differ in exactly
    the intended way, so the surrounding formatting -- markers, ordering,
    separators -- has to match the live path's rendering exactly.
    """
    lines: list[str] = []

    if entities:
        lines.append("=== Entities ===")
        for ent in entities:
            name = ent.get("name", "")
            ent_type = ent.get("type", "")
            desc = ent.get("desc", "")
            temporal_tag = _render_replay_temporal_tag(ent_type, ent.get("prov") or {}, request_id)
            lines.append(f"- {name} ({ent_type}): {desc}{temporal_tag}")

    if relationships:
        lines.append("\n=== Relationships ===")
        for rel in relationships:
            src_name = rel.get("source_name", "")
            tgt_name = rel.get("target_name", "")
            rel_desc = rel.get("rel_desc", "")
            temporal_tag = _render_replay_temporal_tag(rel.get("type", ""), rel.get("prov") or {}, request_id)
            lines.append(f"- {src_name} -> {tgt_name}: {rel_desc}{temporal_tag}")

    return "\n".join(lines) if lines else ""


def replay_summary_raw_text_from_run_answer(item: dict) -> dict:
    """Replay a previous run's selected evidence ids, but replace summaries with raw session text."""
    question = str(item.get("question", "")).strip()
    replay_record = _replay_question_context.get(_normalize_replay_question_key(question), {})
    source_request_id = str(replay_record.get("request_id") or "").strip()

    warnings: list[str] = []
    if not replay_record:
        warnings.append("missing_replay_question_record")

    selected_evidence_ids = replay_record.get("selected_evidence_ids") or []
    if not isinstance(selected_evidence_ids, list):
        selected_evidence_ids = []

    final_entity_names = replay_record.get("final_entity_names") or []
    if not isinstance(final_entity_names, list):
        final_entity_names = []

    final_relationship_names = replay_record.get("final_relationship_names") or []
    if not isinstance(final_relationship_names, list):
        final_relationship_names = []

    seen_summary_ids: set[str] = set()
    raw_blocks: list[tuple[str, str | None, str]] = []
    for summary_id in selected_evidence_ids:
        summary_id_str = str(summary_id or "").strip()
        if not summary_id_str or summary_id_str in seen_summary_ids:
            continue
        seen_summary_ids.add(summary_id_str)
        session_id, _message_id = _parse_replay_summary_id(summary_id_str)
        if session_id is None:
            warnings.append("unparseable_replay_summary_id")
            continue
        raw_text, date_time = _render_session_raw_text(session_id)
        if not raw_text:
            warnings.append("missing_replay_raw_text")
            continue
        raw_blocks.append((summary_id_str, date_time, raw_text))

    context_text = _render_replay_context_text(
        _build_replay_entities(final_entity_names),
        _build_replay_relationships(final_relationship_names),
        request_id=source_request_id or None,
    )

    lines: list[str] = []
    if context_text:
        lines.append(context_text)
    if raw_blocks:
        if lines:
            lines.append("")
        lines.append("### Evidence Raw Text")
        for summary_id, date_time, raw_text in raw_blocks:
            dt_prefix = f"[{date_time}]" if date_time else ""
            lines.append(f"  • {dt_prefix}[sid={summary_id}] {raw_text}")
    else:
        warnings.append("missing_replay_selected_evidence")

    context = "\n".join(lines) if lines else "(no replay context available)"
    messages = [
        {"role": "system", "content": f"---Retrieved Context---\n{context}\n------------------"},
        {"role": "user", "content": (
            "Please answer based on the retrieved knowledge graph context above. "
            "Be concise and accurate.\n\n"
            f"Question: {question}\n\nAnswer:"
        )},
    ]
    t0 = time.time()
    answer_raw = llm_post(messages, temperature=0.0, max_tokens=1024)
    elapsed = time.time() - t0
    answer = scrub(answer_raw) or "(no assistant reply)"

    request_id = (
        f"replay_summary_raw_text_from_run:{source_request_id}"
        if source_request_id
        else "replay_summary_raw_text_from_run"
    )
    print(f"[REPLAY_RAW] {question[:60]!r} -> {answer[:80]!r} ({elapsed:.2f}s)")

    log_dir = Path(os.environ.get("KG_TRACE_PRETTY_LOG_DIR", "logs"))
    append_analysis_record(log_dir, "retrieval_summary", {
        "question": question,
        "request_id": request_id,
        "source_request_id": source_request_id,
        "stop_reason": "replay_summary_raw_text_from_run",
        "retrieval_mode": "replay_summary_raw_text_from_run",
        "selected_evidence_count": len(raw_blocks),
        "selected_evidence_ids": selected_evidence_ids,
        "final_entity_names": final_entity_names,
        "final_relationship_names": final_relationship_names,
        "warnings": warnings,
        "latency_sec": round(elapsed, 3),
        "answer": answer,
    })

    return {
        "answer": answer,
        "retrieved_context": context,
        "evidence": context,
        "latency_sec": round(elapsed, 3),
        "retrieval_request_id": request_id,
        "retrieval_stop_reason": "replay_summary_raw_text_from_run",
        "retrieval_failure_type": ",".join(sorted(set(warnings))) if warnings else "",
        "retrieval_confidence": replay_record.get("conf_final"),
        "retrieval_tau": replay_record.get("tau_confidence"),
        "selected_evidence_count": len(raw_blocks),
        "selected_evidence_ids": compact_json(selected_evidence_ids),
        "selected_evidence_preview": compact_json([text[:100] for _, _, text in raw_blocks[:3]]),
        "final_entity_names": compact_json(final_entity_names),
        "final_relationship_names": compact_json(final_relationship_names),
        "anomaly_flags": compact_json(sorted(set(warnings))),
        "pass2_triggered": bool(replay_record.get("pass2_triggered", False)),
        "pass1_entity_ids": compact_json(replay_record.get("pass1_entity_ids") or []),
        "pass2_entity_ids": compact_json(replay_record.get("pass2_entity_ids") or []),
        "pass1_relation_ids": compact_json(replay_record.get("pass1_relation_ids") or []),
        "pass2_relation_ids": compact_json(replay_record.get("pass2_relation_ids") or []),
        "entity_overlap_count": 0,
        "relation_overlap_count": 0,
        "entity_overlap_pct": None,
        "relation_overlap_pct": None,
    }


def replay_summary_fact_from_run_answer(item: dict) -> dict:
    """Replay a previous run's selected evidence ids, extract facts from raw session text."""
    question = str(item.get("question", "")).strip()
    replay_record = _replay_question_context.get(_normalize_replay_question_key(question), {})
    source_request_id = str(replay_record.get("request_id") or "").strip()

    warnings: list[str] = []
    if not replay_record:
        warnings.append("missing_replay_question_record")

    selected_evidence_ids = replay_record.get("selected_evidence_ids") or []
    if not isinstance(selected_evidence_ids, list):
        selected_evidence_ids = []

    final_entity_names = replay_record.get("final_entity_names") or []
    if not isinstance(final_entity_names, list):
        final_entity_names = []

    final_relationship_names = replay_record.get("final_relationship_names") or []
    if not isinstance(final_relationship_names, list):
        final_relationship_names = []

    seen_summary_ids: set[str] = set()
    fact_blocks: list[tuple[str, str | None, list[str]]] = []
    for summary_id in selected_evidence_ids:
        summary_id_str = str(summary_id or "").strip()
        if not summary_id_str or summary_id_str in seen_summary_ids:
            continue
        seen_summary_ids.add(summary_id_str)
        session_id, _message_id = _parse_replay_summary_id(summary_id_str)
        if session_id is None:
            warnings.append("unparseable_replay_summary_id")
            continue
        raw_text, date_time = _render_session_raw_text(session_id)
        if not raw_text:
            warnings.append("missing_replay_raw_text")
            continue
        if session_id not in _fact_cache:
            _fact_cache[session_id] = _extract_facts_for_evidence(raw_text, date_time)
        facts = _fact_cache[session_id]
        fact_blocks.append((summary_id_str, date_time, facts))

    context_text = _render_replay_context_text(
        _build_replay_entities(final_entity_names),
        _build_replay_relationships(final_relationship_names),
        request_id=source_request_id or None,
    )

    lines: list[str] = []
    if context_text:
        lines.append(context_text)
    if fact_blocks:
        if lines:
            lines.append("")
        lines.append("### Evidence Facts")
        for summary_id, date_time, facts in fact_blocks:
            dt_prefix = f"[{date_time}]" if date_time else ""
            lines.append(f"  • {dt_prefix}[sid={summary_id}]")
            for fact in facts:
                lines.append(f"    • {fact}")
    else:
        warnings.append("missing_replay_selected_evidence")

    context = "\n".join(lines) if lines else "(no replay context available)"
    messages = [
        {"role": "system", "content": f"---Retrieved Context---\n{context}\n------------------"},
        {"role": "user", "content": (
            "Please answer based on the retrieved knowledge graph context above. "
            "Be concise and accurate.\n\n"
            f"Question: {question}\n\nAnswer:"
        )},
    ]
    t0 = time.time()
    answer_raw = llm_post(messages, temperature=0.0, max_tokens=1024)
    elapsed = time.time() - t0
    answer = scrub(answer_raw) or "(no assistant reply)"

    request_id = (
        f"replay_summary_fact_from_run:{source_request_id}"
        if source_request_id
        else "replay_summary_fact_from_run"
    )
    print(f"[REPLAY_FACT] {question[:60]!r} -> {answer[:80]!r} ({elapsed:.2f}s)")

    log_dir = Path(os.environ.get("KG_TRACE_PRETTY_LOG_DIR", "logs"))
    append_analysis_record(log_dir, "retrieval_summary", {
        "question": question,
        "request_id": request_id,
        "source_request_id": source_request_id,
        "stop_reason": "replay_summary_fact_from_run",
        "retrieval_mode": "replay_summary_fact_from_run",
        "selected_evidence_count": len(fact_blocks),
        "selected_evidence_ids": selected_evidence_ids,
        "final_entity_names": final_entity_names,
        "final_relationship_names": final_relationship_names,
        "warnings": warnings,
        "latency_sec": round(elapsed, 3),
        "answer": answer,
        "retrieved_context": context,
    })

    return {
        "answer": answer,
        "retrieved_context": context,
        "evidence": context,
        "latency_sec": round(elapsed, 3),
        "retrieval_request_id": request_id,
        "retrieval_stop_reason": "replay_summary_fact_from_run",
        "retrieval_failure_type": ",".join(sorted(set(warnings))) if warnings else "",
        "retrieval_confidence": replay_record.get("conf_final"),
        "retrieval_tau": replay_record.get("tau_confidence"),
        "selected_evidence_count": len(fact_blocks),
        "selected_evidence_ids": compact_json(selected_evidence_ids),
        "selected_evidence_preview": compact_json([facts[:1] for _, _, facts in fact_blocks[:3]]),
        "final_entity_names": compact_json(final_entity_names),
        "final_relationship_names": compact_json(final_relationship_names),
        "anomaly_flags": compact_json(sorted(set(warnings))),
        "pass2_triggered": bool(replay_record.get("pass2_triggered", False)),
        "pass1_entity_ids": compact_json(replay_record.get("pass1_entity_ids") or []),
        "pass2_entity_ids": compact_json(replay_record.get("pass2_entity_ids") or []),
        "pass1_relation_ids": compact_json(replay_record.get("pass1_relation_ids") or []),
        "pass2_relation_ids": compact_json(replay_record.get("pass2_relation_ids") or []),
        "entity_overlap_count": 0,
        "relation_overlap_count": 0,
        "entity_overlap_pct": None,
        "relation_overlap_pct": None,
    }


def _extract_latest_t_tag(kg_context: str) -> str | None:
    """Return the most recent date string found in [t:...] tags in the KG context."""
    from grace_mem.utils.query_time_parser import parse_query_time
    dates = []
    for m in re.finditer(r'\[t:([^\]]+)\]', kg_context):
        dt = parse_query_time(m.group(1).strip())
        if dt:
            dates.append((dt, m.group(1).strip()))
    if not dates:
        return None
    _, date_str = max(dates, key=lambda x: x[0])
    return date_str


def rag_answer(
    query: str,
    *,
    ent_topk: int = RETRIEVAL_PARAMS.get("ent_topk", 20),
    rel_topk: int = RETRIEVAL_PARAMS.get("rel_topk", 10),
    ent_threshold: float = RETRIEVAL_PARAMS.get("ent_threshold", 0.2),
    rel_threshold: float = RETRIEVAL_PARAMS.get("rel_threshold", 0.2),
    filter_ent_topk: int = RETRIEVAL_PARAMS.get("filter_ent_topk", 10),
    filter_rel_topk: int = RETRIEVAL_PARAMS.get("filter_rel_topk", 10),
    filter_ent_threshold: float = RETRIEVAL_PARAMS.get("filter_ent_threshold", 0.4),
    filter_rel_threshold: float = RETRIEVAL_PARAMS.get("filter_rel_threshold", 0.4),
    summary_topk_per_item: int = RETRIEVAL_PARAMS.get("summary_topk_per_item", 6),
    summary_vec_threshold: float = RETRIEVAL_PARAMS.get("summary_vec_threshold", 0.4),
):
    """
    Pure retrieval + answer (no writes).
    Returns dict with: answer, retrieved_context, evidence
    """
    def _json_ids(value):
        return json.dumps(list(value or []), ensure_ascii=False)

    def _trace_field(trace: dict | None, key: str, default):
        if not trace:
            return default
        value = trace.get(key, default)
        return default if value is None and default is not None else value

    # 1) Build KG context using current retriever
    kg_context = retriever.build_kg_context(
        question=query,
        ent_topk=ent_topk,
        rel_topk=rel_topk,
        ent_threshold=ent_threshold,
        rel_threshold=rel_threshold,
        filter_ent_topk=filter_ent_topk,
        filter_rel_topk=filter_rel_topk,
        filter_ent_threshold=filter_ent_threshold,
        filter_rel_threshold=filter_rel_threshold,
        summary_topk_per_item=summary_topk_per_item,
        summary_vec_threshold=summary_vec_threshold,
    )
    trace = getattr(retriever, "last_retrieval_trace", None) or {}
    log_dir = Path(os.environ.get("KG_TRACE_PRETTY_LOG_DIR", "logs"))

    # 2) Call LLM
    conversation_date = _extract_latest_t_tag(kg_context)
    date_note = (
        f"\nNote: These conversations took place around {conversation_date}. "
        "For questions about durations or how long ago something happened, "
        "calculate from this date, not from today."
    ) if conversation_date else ""
    messages = [
        {"role": "system", "content": f"---Retrieved Context---\n{kg_context}\n------------------"},
        {"role": "user", "content": (
            "Please answer based on the retrieved knowledge graph context above. "
            f"Be concise and accurate.{date_note}\n\n"
            f"Question: {query}\n\nAnswer:"
        )},
    ]
    t0 = time.time()
    answer_raw = llm_post(messages, temperature=0.0, max_tokens=1024)
    elapsed = time.time() - t0
    answer = scrub(answer_raw) or "(no assistant reply)"
    print(f"[RAG] {query[:60]!r} -> {answer[:80]!r} ({elapsed:.2f}s)")
    selected_evidence = trace.get("selected_evidence") or []
    summary_record = {
        "request_id": trace.get("request_id"),
        "question": query,
        "low_level_keywords": trace.get("low_level_keywords", []),
        "high_level_keywords": trace.get("high_level_keywords", []),
        "stop_reason": trace.get("stop_reason"),
        "branches": trace.get("branches", {}),
        "pass2_triggered": bool(trace.get("pass2_triggered", False)),
        "rewritten_query": trace.get("rewritten_query"),
        "conf_pass1": trace.get("conf_pass1"),
        "conf_pass2": trace.get("conf_pass2"),
        "conf_final": trace.get("conf_final"),
        "tau_confidence": trace.get("tau_confidence"),
        "pass1_entity_ids": trace.get("pass1_entity_ids", []),
        "pass2_entity_ids": trace.get("pass2_entity_ids", []),
        "pass1_relation_ids": trace.get("pass1_relation_ids", []),
        "pass2_relation_ids": trace.get("pass2_relation_ids", []),
        "final_entity_count": trace.get("final_entity_count", 0),
        "final_relationship_count": trace.get("final_relationship_count", 0),
        "final_entity_names": trace.get("final_entity_names", []),
        "final_relationship_names": trace.get("final_relationship_names", []),
        "selected_evidence_count": trace.get("selected_evidence_count", 0),
        "selected_evidence_ids": [item.get("summary_id") for item in selected_evidence if item.get("summary_id")],
        "selected_evidence_preview": [item.get("preview") for item in selected_evidence[:3]],
        "has_temporal_evidence": bool(trace.get("has_temporal_evidence", False)),
        "latency_sec": round(elapsed, 3),
        "answer": answer,
    }
    append_analysis_record(log_dir, "retrieval_summary", summary_record)
    for drop_reason in derive_drop_reasons(summary_record):
        append_analysis_record(log_dir, "drop_reasons", drop_reason)
    return {
        "answer": answer,
        "retrieved_context": kg_context,
        "evidence": kg_context,  # Full context includes evidence
        "latency_sec": round(elapsed, 3),
        "retrieval_request_id": trace.get("request_id"),
        "retrieval_stop_reason": trace.get("stop_reason"),
        "retrieval_failure_type": trace.get("stop_reason"),
        "retrieval_confidence": trace.get("conf_final"),
        "retrieval_tau": trace.get("tau_confidence"),
        "selected_evidence_count": trace.get("selected_evidence_count", 0),
        "selected_evidence_ids": compact_json(summary_record["selected_evidence_ids"]),
        "selected_evidence_preview": compact_json(summary_record["selected_evidence_preview"]),
        "final_entity_names": compact_json(trace.get("final_entity_names", [])),
        "final_relationship_names": compact_json(trace.get("final_relationship_names", [])),
        "anomaly_flags": compact_json([]),
        "pass2_triggered": bool(_trace_field(trace, "pass2_triggered", False)),
        "pass1_entity_ids": _json_ids(_trace_field(trace, "pass1_entity_ids", [])),
        "pass2_entity_ids": _json_ids(_trace_field(trace, "pass2_entity_ids", [])),
        "pass1_relation_ids": _json_ids(_trace_field(trace, "pass1_relation_ids", [])),
        "pass2_relation_ids": _json_ids(_trace_field(trace, "pass2_relation_ids", [])),
        "entity_overlap_count": _trace_field(trace, "entity_overlap_count", 0),
        "relation_overlap_count": _trace_field(trace, "relation_overlap_count", 0),
        "entity_overlap_pct": trace.get("entity_overlap_pct"),
        "relation_overlap_pct": trace.get("relation_overlap_pct"),
    }

def load_questions(
    dataset_json_path: str | Path,
    *,
    sample_index: int = 7,
    include_adversarial: bool = True,
):
    """Load one sample's questions, normalized, honouring the adversarial filter."""
    qa_list = load_qa_items(
        dataset_json_path,
        sample_index=sample_index,
        include_adversarial=include_adversarial,
    )
    print(
        f"[Schema] dataset={Path(dataset_json_path).name} "
        f"picked sample_index={sample_index}, qa_count={len(qa_list)}"
    )
    return qa_list

def pick_gold_answer(item: dict) -> str:
    # Prefer 'answer'; fallback to 'adversarial_answer' for category 5 items
    """Choose the gold answer to score against, honouring the adversarial variant."""
    if "answer" in item and item["answer"] not in (None, "", []):
        return str(item["answer"])
    if "adversarial_answer" in item and item["adversarial_answer"] not in (None, "", []):
        return str(item["adversarial_answer"])
    return ""


def prediction_fallback(error: Exception) -> dict:
    """Produce an answer when generation returned nothing usable.

    An empty prediction would be judged wrong, which is the correct outcome but
    the wrong diagnosis -- it looks like a retrieval failure rather than a
    generation one. The fallback makes that distinction visible in the results.
    """
    return {
        "answer": f"(ERROR: {error})",
        "retrieved_context": "",
        "evidence": "",
        "retrieval_request_id": "",
        "retrieval_stop_reason": "prediction_exception",
        "retrieval_failure_type": "retrieval_exception",
        "retrieval_confidence": None,
        "retrieval_tau": None,
        "selected_evidence_count": 0,
        "selected_evidence_ids": "[]",
        "selected_evidence_preview": "[]",
        "final_entity_names": "[]",
        "final_relationship_names": "[]",
        "anomaly_flags": "[]",
        "pass2_triggered": False,
        "pass1_entity_ids": "[]",
        "pass2_entity_ids": "[]",
        "pass1_relation_ids": "[]",
        "pass2_relation_ids": "[]",
        "entity_overlap_count": 0,
        "relation_overlap_count": 0,
        "entity_overlap_pct": None,
        "relation_overlap_pct": None,
    }


def evaluate_item(
    item: dict,
    *,
    simplify_evidence: bool = False,
) -> dict | None:
    """Evaluate one question: retrieve, answer, and record the trace.

    The retrieval path is selected by the module-level `retrieval_mode` -- see
    the module docstring for what each ablation isolates.
    """
    question = str(item.get("question", "")).strip()
    if not question:
        return None

    gold_answer = pick_gold_answer(item)
    gold_evidence = ";".join(item.get("evidence", []) or [])
    try:
        if retrieval_mode == "gold_summary_only":
            prediction = gold_summary_answer(item)
        elif retrieval_mode == "gold_raw_text_only":
            prediction = gold_raw_text_answer(item)
        elif retrieval_mode == "replay_summary_raw_text_from_run":
            prediction = replay_summary_raw_text_from_run_answer(item)
        elif retrieval_mode == "replay_summary_fact_from_run":
            prediction = replay_summary_fact_from_run_answer(item)
        else:
            prediction = rag_answer(question)
    except Exception as exc:
        prediction = prediction_fallback(exc)

    if simplify_evidence:
        gold_evidence = simplify_gold_evidence(gold_evidence)

    return {
        "question": question,
        "gold_answer": gold_answer,
        "gold_evidence_source": gold_evidence,
        "model_answer": prediction.get("answer", ""),
        "retrieved_context": prediction.get("retrieved_context", ""),
        "rendered_evidence": prediction.get("evidence", ""),
        "retrieval_request_id": prediction.get("retrieval_request_id", ""),
        "retrieval_stop_reason": prediction.get("retrieval_stop_reason", ""),
        "retrieval_failure_type": prediction.get("retrieval_failure_type", ""),
        "retrieval_confidence": prediction.get("retrieval_confidence"),
        "retrieval_tau": prediction.get("retrieval_tau"),
        "selected_evidence_count": prediction.get("selected_evidence_count", 0),
        "selected_evidence_ids": prediction.get("selected_evidence_ids", "[]"),
        "selected_evidence_preview": prediction.get("selected_evidence_preview", "[]"),
        "final_entity_names": prediction.get("final_entity_names", "[]"),
        "final_relationship_names": prediction.get("final_relationship_names", "[]"),
        "anomaly_flags": prediction.get("anomaly_flags", "[]"),
        "pass2_triggered": prediction.get("pass2_triggered", False),
        "pass1_entity_ids": prediction.get("pass1_entity_ids", "[]"),
        "pass2_entity_ids": prediction.get("pass2_entity_ids", "[]"),
        "pass1_relation_ids": prediction.get("pass1_relation_ids", "[]"),
        "pass2_relation_ids": prediction.get("pass2_relation_ids", "[]"),
        "entity_overlap_count": prediction.get("entity_overlap_count", 0),
        "relation_overlap_count": prediction.get("relation_overlap_count", 0),
        "entity_overlap_pct": prediction.get("entity_overlap_pct"),
        "relation_overlap_pct": prediction.get("relation_overlap_pct"),
    }


def evaluate_items(
    qa_items: list[dict] | tuple[dict, ...],
    *,
    simplify_evidence: bool = False,
) -> list[dict]:
    # Optional filter: KG_QUESTION_FILTER env var (newline-separated substrings)
    """Evaluate every question in a sample and return the result rows."""
    _filter_raw = os.environ.get("KG_QUESTION_FILTER", "").strip()
    _filter_strs = [s.strip().lower() for s in _filter_raw.splitlines() if s.strip()] if _filter_raw else []

    rows = []
    for item in qa_items:
        if _filter_strs:
            q = str(item.get("question", "")).lower()
            if not any(f in q for f in _filter_strs):
                continue
        row = evaluate_item(item, simplify_evidence=simplify_evidence)
        if row is not None:
            rows.append(row)
    return rows

# --- Post-processing helpers ---

def simplify_gold_evidence(evi: str) -> str:
    """D5:4;D8:3;D2:1 -> '5,8,2' (sorted unique). Preserve raw text when not dia_id-based."""
    if not isinstance(evi, str):
        return ""
    ids = re.findall(r"D(\d+):", evi)
    if not ids:
        return evi.strip()
    uniq_sorted = sorted(set(ids), key=lambda x: int(x))
    return ",".join(uniq_sorted)

def extract_session_ids(text: str) -> str:
    """... [session=6, message=1] ... -> '6'; final -> '1,6,9' (sorted unique)"""
    if not isinstance(text, str):
        return ""
    sessions = re.findall(r"\[session=(\d+),", text)
    uniq_sorted = sorted(set(sessions), key=lambda x: int(x))
    return ",".join(uniq_sorted)

def coverage_percent(gold_ids_csv: str, rendered_ids_csv: str) -> float:
    """
    Compute coverage: |gold ∩ rendered| / |gold| * 100
    Returns 0.0 when gold is empty.
    """
    if not gold_ids_csv:
        return 0.0
    gold = {int(x) for x in gold_ids_csv.split(",") if x.strip().isdigit()}
    rendered = {int(x) for x in rendered_ids_csv.split(",") if x.strip().isdigit()}
    if not gold:
        return 0.0
    hit = len(gold & rendered)
    return round(100.0 * hit / len(gold), 2)

class QAEvalStage:
    """Class interface for standalone or embedded QA evaluation runs.

    The module-level `retriever` variable remains the adapter's call surface.
    Pass retriever explicitly here to avoid relying on module-level mutation.
    """

    def __init__(
        self,
        *,
        retriever,
        dataset_json,
        sample_index: int,
        include_adversarial: bool = False,
        simplify_evidence: bool = True,
    ) -> None:
        self.retriever = retriever
        self.dataset_json = dataset_json
        self.sample_index = sample_index
        self.include_adversarial = include_adversarial
        self.simplify_evidence = simplify_evidence

    def run(self) -> list[dict]:
        import experiment.locomo.stages.qa_eval as _self
        _self.retriever = self.retriever
        qa_items = load_questions(
            self.dataset_json,
            sample_index=self.sample_index,
            include_adversarial=self.include_adversarial,
        )
        return evaluate_items(qa_items, simplify_evidence=self.simplify_evidence)


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation for a conversational QA dataset sample")
    parser.add_argument("--dataset", choices=["locomo", "locomo-plus"], default="locomo")
    parser.add_argument("--dataset-json", default=None, help="Defaults are resolved from --dataset")
    parser.add_argument("--sample-index", type=int, default=3)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--adv", action="store_true", help="Include adversarial questions")
    args = parser.parse_args()

    dataset = normalize_dataset_name(args.dataset)
    dataset_json_path = resolve_dataset_path(
        dataset=dataset,
        kind="qa_json",
        explicit_path=args.dataset_json,
    )
    if args.output_csv:
        output_csv = Path(args.output_csv)
    elif dataset == "locomo":
        output_csv = Path(__file__).resolve().parent / "data" / f"sample{args.sample_index}_eval.csv"
    else:
        output_csv = Path(__file__).resolve().parent / "data" / f"{default_output_stem(dataset)}_sample{args.sample_index}_eval.csv"

    # Standalone mode owns one pipeline runtime for retrieval and graph access.
    import experiment.locomo.stages.qa_eval as _self
    from grace_mem.pipeline.factory import build_pipeline

    with build_pipeline(retriever_config=RERANKER_PARAMS) as runtime:
        _self.retriever = runtime.retriever
        MGR.initialize()
        qa_items = load_questions(
            dataset_json_path,
            sample_index=args.sample_index,
            include_adversarial=args.adv,
        )

        rows = evaluate_items(qa_items, simplify_evidence=False)

        df = pd.DataFrame(rows, columns=EVAL_COLUMNS)

        # === Cleaning + coverage ===
        df_clean = df.copy()
        df_clean["gold_evidence_source"] = df_clean["gold_evidence_source"].apply(simplify_gold_evidence)
        df_clean["rendered_evidence"] = df_clean["rendered_evidence"].apply(extract_session_ids)
        df_clean["coverage_percent"] = [
            coverage_percent(g, r) for g, r in zip(df_clean["gold_evidence_source"], df_clean["rendered_evidence"])
        ]

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

        print(f"[DONE] Wrote CLEAN CSV: {output_csv}")

if __name__ == "__main__":
    main()
