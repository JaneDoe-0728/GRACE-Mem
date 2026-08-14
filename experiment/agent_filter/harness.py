"""Grep agent mini-harness (inline delivery, plain-text command protocol).

Flow:
  seed = the 16 sids inside the Evidence Summary (vector+rerank coarse filter)
  -> the agent verifies the candidates with GREP / READ and hunts down the
     literal evidence that was missed
  -> FINAL sids -> rebuild the Evidence Summary block from raw turn text
Safety net: if the agent fails, emits invalid output, or blows the budget, the
original context is handed back untouched.

No function-calling API: local models (gpt-oss-20b via LM Studio) are steadiest
against a plain-text one-command-per-line protocol, parsed with regexes.
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback
from pathlib import Path

from experiment.agent_filter.corpus import Corpus, load_corpus
from experiment.agent_filter.prompts import (
    ABSTENTION_HINT,
    CATEGORY_HINTS,
    GAP_HINT_TEMPLATE,
    SUFFICIENCY_SYSTEM,
    SUFFICIENCY_USER,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
)

_EVIDENCE_HEADER = "### Evidence Summary"
_SID_RE = re.compile(r"\[sid=([^\]\s]+)\]")

_GREP_RE = re.compile(r"^\s*GREP\s+(.+?)\s*$", re.IGNORECASE)
# Detects aggregation/latest-value questions (a question-driven trigger for the
# retention strategy, independent of any dataset category label)
_AGG_QUESTION_RE = re.compile(
    r"\b(how many|how much|how often|how long|how frequently|total|count|sum|"
    r"number of|most recent(ly)?|latest|currently|current|in total|altogether)\b",
    re.IGNORECASE,
)
_READ_RE = re.compile(r"^\s*READ\s+(\S+)(?:\s+(\d+))?\s*$", re.IGNORECASE)
_VECTOR_RE = re.compile(r"^\s*VECTOR\s+(.+?)\s*$", re.IGNORECASE)
_FINAL_RE = re.compile(r"^\s*FINAL\s*[::]?\s*(.*?)\s*$", re.IGNORECASE)
_COVERAGE_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]{2,}", re.IGNORECASE)
_COVERAGE_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "were", "have", "has",
    "had", "for", "her", "his", "their", "they", "them", "she", "you",
    "your", "what", "when", "where", "which", "does", "did", "about",
    "into", "also", "just", "very", "more", "some", "user", "assistant",
}


def seed_sids_from_context(context: str) -> list[str]:
    """Pull the sids out of the Evidence Summary block in order of appearance,
    deduplicated but order-preserving."""
    idx = context.find(_EVIDENCE_HEADER)
    if idx == -1:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in _SID_RE.finditer(context[idx:]):
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
    idx = context.find(_EVIDENCE_HEADER)
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
    idx = context.find(_EVIDENCE_HEADER)
    prefix = context[:idx] if idx != -1 else context
    if "=== Entities ===" not in prefix and "=== Relationships ===" not in prefix:
        return ""
    prefix = prefix.strip()
    return prefix if len(prefix) <= max_chars else prefix[:max_chars] + "\n…(graph context truncated)"


def _resp_diag(resp) -> dict:
    """Response-level diagnostics: tells "the model genuinely replied empty" apart
    from "the output landed in the reasoning channel / tool_calls where content
    cannot see it" (common with gpt-oss harmony). Attached to every step trace."""
    try:
        choice = resp.choices[0]
        msg = choice.message
    except (AttributeError, IndexError):
        return {"diag": "no_choices"}
    d: dict = {"finish_reason": getattr(choice, "finish_reason", None)}
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    if reasoning:
        d["reasoning"] = str(reasoning)[:1200]
    tc = getattr(msg, "tool_calls", None)
    if tc:
        d["tool_calls"] = [
            {"name": getattr(getattr(t, "function", None), "name", None),
             "arguments": str(getattr(getattr(t, "function", None), "arguments", ""))[:300]}
            for t in tc
        ]
    content = getattr(msg, "content", None)
    if not (content or "").strip():
        # Empty content: record which fields the message actually carried, so an
        # adapter dropping fields can be ruled out
        d["content_empty"] = True
        d["message_keys"] = sorted(vars(msg).keys())
    return d


def _response_command_candidates(resp) -> list[tuple[str, str]]:
    """Return possible command-bearing text from an OpenAI-compatible response.

    gpt-oss/Harmony adapters do not agree on where the visible command goes:
    some put it in ``content``, some expose native ``tool_calls``, and some
    (notably older LM Studio adapters) put it in ``reasoning``.  The command
    parser itself is format-agnostic, so normalize all three surfaces here.
    """
    try:
        message = resp.choices[0].message
    except (AttributeError, IndexError):
        return []

    candidates: list[tuple[str, str]] = []
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        candidates.append(("content", content.strip()))

    tool_calls = getattr(message, "tool_calls", None) or []
    for call in tool_calls:
        function = getattr(call, "function", None)
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
        if not name:
            continue
        name = str(name).split(".")[-1].upper()
        if name not in _CMD_NAMES:
            continue
        if isinstance(arguments, str):
            payload = arguments
        else:
            payload = json.dumps(arguments or {}, ensure_ascii=False)
        candidates.append(("tool_calls", f"to={name} <|message|>{payload}"))

    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        candidates.append(("reasoning", reasoning.strip()))
    return candidates


# gpt-oss (harmony template) sometimes replies in native tool-call syntax:
#   <|channel|>commentary to=READ <|constrain|>json<|message|>{"id": "...", "k": 2}
_HARMONY_RE = re.compile(
    r"to=(?:\w+\.)?(GREP|READ|VECTOR|FINAL)\b.*?<\|message\|>\s*(\{.*?\})\s*(?:<\|\w+\|>|$)",
    re.IGNORECASE | re.DOTALL,
)
# A messier variant: to=GREP <|constrain|>="pattern" (no <|message|> JSON)
# Namespace prefixes are treated loosely: functions./tool./any <ns>. is stripped
# (the model on box 92 emits to=tool.GREP).
_HARMONY_LOOSE_RE = re.compile(r"to=(?:\w+\.)?(GREP|READ|VECTOR|FINAL)\b(.*)$", re.IGNORECASE)


_CMD_NAMES = {"GREP", "READ", "VECTOR", "FINAL"}


def _flatten_json(obj) -> tuple[list[str], list[int]]:
    """Recursively collect the strings and integers in a JSON payload (the model's
    schema is not predictable)."""
    strings: list[str] = []
    ints: list[int] = []
    stack = [obj]
    while stack:
        cur = stack.pop(0)
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
        elif isinstance(cur, bool):
            continue
        elif isinstance(cur, int):
            ints.append(cur)
        elif isinstance(cur, str):
            strings.append(cur)
    return strings, ints


def _parse_harmony(reply: str) -> tuple[str, str] | None:
    """Parse a Harmony-format reply into its channels.

    Some backends wrap replies in Harmony's channel markup; unwrapping here
    keeps the callers from having to know which backend produced a reply.
    """
    m = None
    for m in _HARMONY_RE.finditer(reply):
        pass  # take the last tool call
    if m is None:
        return None
    kind = m.group(1).upper()
    try:
        payload = json.loads(m.group(2))
    except json.JSONDecodeError:
        return None

    strings, ints = _flatten_json(payload)
    # If the payload names the command itself (e.g. {"cmd": ["GREP", ...]}), that wins
    for s in strings:
        if s.strip().upper() in _CMD_NAMES:
            kind = s.strip().upper()
    args = [s for s in strings if s.strip().upper() not in _CMD_NAMES]

    if kind in ("GREP", "VECTOR"):
        return (kind, _unquote(args[0])) if args else None
    if kind == "READ":
        sid = next((s for s in args if ":" in s), None)
        k = next((i for i in ints if 0 < i <= 10), 2)
        return ("READ", f"{_unquote(sid)} {k}") if sid else None
    return ("FINAL", " ".join(s for s in args if ":" in s))


def _parse_harmony_loose(reply: str) -> tuple[str, str] | None:
    """Last resort: JSON-less variants such as `to=GREP <|constrain|>="pattern"`.
    Take the rest of the line after to=CMD and strip the harmony markers and the
    constrain/json noise to get the argument."""
    m = None
    for line in reply.splitlines():
        for m2 in _HARMONY_LOOSE_RE.finditer(line):
            m = m2
    if m is None:
        return None
    kind = m.group(1).upper()
    tail = re.sub(r"<\|[^|]*\|>", " ", m.group(2))
    tail = re.sub(r"\b(?:json|commentary|response)\b", " ", tail, flags=re.IGNORECASE)
    tail = tail.strip().lstrip("=").strip()
    arg = _unquote(tail)
    if kind in ("GREP", "VECTOR"):
        return (kind, arg) if arg else None
    if kind == "READ":
        sid = next((s for s in re.split(r"[,\s]+", arg) if ":" in s), None)
        return ("READ", f"{_unquote(sid)} 2") if sid else None
    return ("FINAL", arg)


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
        return s[1:-1]
    return s


def _parse_command(reply: str) -> tuple[str, str] | None:
    """Parse the command out of the reply's last few lines, tolerating reasoning
    before it. The plain-text protocol is tried first, then harmony's native
    tool-call syntax as a fallback.
    Harmony markers (<|channel|> and friends) are turned into newlines first, so
    a plain-text command wedged in after <|message|> is still caught by the
    line-based parse."""
    sanitized = re.sub(r"<\|[^|]*\|>", "\n", reply)
    # 120B (gpt-oss) harmony dual-channel run-together: several commands crammed
    # onto one line, with the whole span repeated ("GREP MelanieGREP Melanie",
    # "READ 0__2:0 5FINAL 0__2:0"). When the line starts with GREP the entire line
    # is taken as the pattern and the trailing FINAL is swallowed -- the root cause
    # of the 49.6% L1 failure rate (2026-07-06). So: an uppercase command word
    # directly after a lowercase char/digit/quote -> break the line. 20B emits one
    # command per line and is unaffected.
    sanitized = re.sub(r"(?<=[a-z0-9\"'\)\].:])((?:GREP|READ|VECTOR)\s|FINAL\b)",
                       r"\n\1", sanitized)
    for line in reversed(sanitized.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        if m := _FINAL_RE.match(line):
            if not m.group(1).strip():
                # Empty FINAL: if the same reply holds other runnable commands,
                # run those first (120B often dumps the whole plan --
                # GREP..READ..FINAL -- in one go)
                continue
            return ("FINAL", m.group(1))
        if m := _GREP_RE.match(line):
            return ("GREP", _unquote(m.group(1)))
        if m := _READ_RE.match(line):
            return ("READ", f"{_unquote(m.group(1))} {m.group(2) or 2}")
        if m := _VECTOR_RE.match(line):
            return ("VECTOR", _unquote(m.group(1)))
    return _parse_harmony(reply) or _parse_harmony_loose(reply)


def _candidates_block(corpus: Corpus, sids: list[str]) -> str:
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


def _coverage_tokens(text: str) -> set[str]:
    """Extract small, dependency-free lexical evidence signatures.

    This is intentionally generic: it does not classify the question or
    assign special handling to temporal/counting/category labels.
    """
    return {
        tok.lower()
        for tok in _COVERAGE_TOKEN_RE.findall(text or "")
        if tok.lower() not in _COVERAGE_STOPWORDS
    }


def _portfolio_pad(
    *,
    question: str,
    corpus: Corpus,
    seed_sids: list[str],
    selected_sids: list[str],
    target_size: int,
    seed_scores: dict[str, float],
) -> tuple[list[str], dict]:
    """Add a diverse evidence portfolio using generic lexical MMR signals.

    Candidates remain restricted to upstream seed sids.  At each step the
    candidate with the best combination of question overlap, new lexical
    coverage, distance from already selected evidence, and upstream score is
    added.  No question category or question-shape rule is used.
    """
    target_size = max(0, target_size)
    selected = list(dict.fromkeys(selected_sids))
    selected_set = set(selected)
    seed_order = list(dict.fromkeys(seed_sids))
    q_tokens = _coverage_tokens(question)
    signatures = {
        s: _coverage_tokens(corpus.display_entry(s, max_chars=4000) or "")
        for s in seed_order
    }
    groups = {s: s.split(":", 1)[0] for s in seed_order}
    covered = set().union(*(signatures.get(s, set()) for s in selected))
    covered_groups = {groups[s] for s in selected if s in groups}
    additions: list[str] = []
    scores: dict[str, float] = {}

    while len(selected) < target_size:
        remaining = [s for s in seed_order if s not in selected_set]
        if not remaining:
            break

        best = None
        best_key = None
        for sid in remaining:
            tokens = signatures.get(sid, set())
            if not tokens:
                continue
            overlap = len(tokens & q_tokens)
            novelty = len(tokens - covered)
            group_gain = 1.0 if groups.get(sid) not in covered_groups else 0.0
            # Penalize redundancy against the most similar selected item.
            max_similarity = 0.0
            for picked in selected:
                other = signatures.get(picked, set())
                union = tokens | other
                if union:
                    max_similarity = max(max_similarity, len(tokens & other) / len(union))
            diversity = 1.0 - max_similarity
            rerank = float(seed_scores.get(sid, 0.0))
            # Lexical question overlap is the relevance guard; novelty and
            # diversity prevent the portfolio from collapsing to near copies.
            score = 2.0 * overlap + 0.25 * novelty + diversity + 1.5 * group_gain + 0.5 * rerank
            key = (score, group_gain, novelty, overlap, rerank, -seed_order.index(sid))
            if best_key is None or key > best_key:
                best_key = key
                best = sid

        if best is None:
            break
        selected.append(best)
        selected_set.add(best)
        additions.append(best)
        scores[best] = round(float(best_key[0]), 4)
        covered.update(signatures.get(best, set()))
        covered_groups.add(groups.get(best, best))

    return selected, {
        "added": additions,
        "candidate_count": len(seed_order),
        "selected_count": len(selected),
        "covered_token_count": len(covered),
        "covered_group_count": len(covered_groups),
        "scores": scores,
    }


def _vector_search(
    artifact_dir,
    corpus: Corpus,
    query: str,
    *,
    exclude: set[str],
    topn: int,
    min_score: float,
) -> str:
    """Execution side of the VECTOR command: embed the query, search that
    question's summaries VDB, and return an inline candidate list (same format as
    GREP; the agent still has to verify them with READ/GREP)."""
    from experiment.agent_filter.gap_vector import vector_gap_candidates

    cands = vector_gap_candidates(
        artifact_dir, query, exclude=exclude, topn=topn, min_score=min_score,
    )
    if not cands:
        return (f"vector {query!r}: 0 hits above threshold. "
                "Try rephrasing the query, or fall back to GREP with rare literal words.")
    lines = [f"vector {query!r}: {len(cands)} semantically similar turns "
             "(NOT verified — check with READ/GREP before including)"]
    for s, sc in cands:
        turns = corpus.resolve(s)
        entry = corpus.display_entry(s, max_chars=200) or "(text unavailable)"
        dt = f"[{turns[0].date}] " if turns and turns[0].date else ""
        lines.append(f"[sid={s}] (score={sc:.2f}) {dt}{entry}")
    return "\n".join(lines)


def _rebuild_context(
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
    idx = context.find(_EVIDENCE_HEADER)
    if include_prefix:
        head = context[:idx].rstrip("\n") if idx != -1 else context.rstrip("\n")
    else:
        head = ""
    lines = [head, _EVIDENCE_HEADER] if head else [_EVIDENCE_HEADER]

    entries: list[str] = []  # sids (pair base or split sid), order-preserving dedup
    seen: set[str] = set()
    for s in final_sids:
        key = s.rsplit(":", 1)[0] if include_pair and (s.endswith(":u") or s.endswith(":a")) else s
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


def _check_sufficiency(
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


def _adjudicate_candidates(
    llm,
    *,
    question: str,
    question_date: str | None,
    corpus: Corpus,
    pending: list[str],
) -> tuple[list[str], dict]:
    """Answer-blind per-item adjudication: an independent call (no agent search
    history, so it cannot see the answer the agent already reached) rules
    KEEP/DROP on every seed that FINAL discarded. The criterion is topical
    relevance to the question, not "contains the answer".
    Returns (the KEEP sids, a per-item verdict dict). A candidate given no
    verdict counts as DROP -- adjudication is an add-only recovery, so no
    verdict means nothing is added back."""
    from experiment.agent_filter.prompts import (
        ADJUDICATE_SYSTEM,
        ADJUDICATE_USER,
    )
    lines = []
    for s in pending:
        t = corpus.resolve(s)
        if not t:
            continue
        entry = corpus.display_entry(s, max_chars=700)
        dt = f"[{t[0].date}] " if t[0].date else ""
        lines.append(f"[sid={s}] {dt}{entry}")
    msgs = [
        {"role": "system", "content": ADJUDICATE_SYSTEM},
        {"role": "user", "content": ADJUDICATE_USER.format(
            question=question,
            date_line=f"QUESTION DATE: {question_date}\n" if question_date else "",
            n=len(lines),
            candidates="\n".join(lines) or "(none)",
        )},
    ]
    # Reasoning model: hidden thinking precedes the verdicts, so the token budget
    # has to be generous -- 2048 was measured truncating a 14-verdict output
    # halfway (in the child run, all 129 unjudged items came from this).
    resp = llm.chat(messages=msgs, temperature=0.0, max_tokens=4096)
    reply = (resp.choices[0].message.content or "").strip()
    kept: list[str] = []
    # reply = the adjudication call's raw response (one `<sid> KEEP|DROP <short
    # reason>` per line). The full text is kept so the front end can reconstruct
    # the adjudication trail, and verdict+reason are extracted per item
    # (reasons: sid -> "KEEP|DROP: reason").
    verdicts: dict = {
        "kept": [], "dropped": [], "unjudged": [],
        "reply": reply, "reply_chars": len(reply), "reasons": {},
    }
    judged: set[str] = set()
    for line in reply.splitlines():
        m = re.search(r"\b(KEEP|DROP)\b", line, re.IGNORECASE)
        if not m:
            continue
        for s in pending:
            if s in judged or s not in line:
                continue
            judged.add(s)
            decision = m.group(1).upper()
            # Take the short reason after KEEP/DROP on that line, dropping the sid
            # and the verdict token. The strip set keeps fullwidth punctuation
            # because the model emits it in reasons.  # allow-cjk
            reason = line[m.end():].strip(" \t:-—．。")
            verdicts["reasons"][s] = f"{decision}: {reason}" if reason else decision
            if decision == "KEEP":
                kept.append(s)
                verdicts["kept"].append(s)
            else:
                verdicts["dropped"].append(s)
            break
    verdicts["unjudged"] = [s for s in pending if s not in judged]
    return kept, verdicts


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
    p = params or {}
    mode = p.get("grep_agent_mode", "filter_fetch")
    max_calls = int(p.get("grep_agent_max_calls", 8))
    max_sids = int(p.get("grep_agent_max_sids", 16))
    grep_max_lines = int(p.get("grep_agent_grep_max_lines", 30))
    include_filter_graph = bool(p.get("grep_agent_filter_include_graph_context", False))
    include_answer_graph = bool(p.get("grep_agent_answer_include_graph_context", True))

    trace: dict = {"enabled": True, "mode": mode, "commands": [], "fallback": None}
    try:
        if corpus is None:
            corpus = load_corpus(csv_path)
        seed = seed_sids_from_context(context)
        trace["seed_sids"] = seed
        trace["seed_scores"] = seed_scores_from_context(context)  # sid -> rerank score
        graph_context = graph_context_from_context(
            context,
            max_chars=int(p.get("grep_agent_graph_context_max_chars", 12000)),
        )
        trace["graph_context_available"] = bool(graph_context)
        trace["filter_graph_context"] = include_filter_graph
        trace["answer_graph_context"] = include_answer_graph
        if not seed and mode == "filter":
            trace["fallback"] = "no_seed"
            return context, trace

        if category is None:
            category = Path(csv_path).parent.name
        # _abs abstention questions (the answer is not in the corpus):
        # force_verified_final must keep the full protective context for these and
        # never narrow -- fvf-73 measured that narrowing (even with plenty of
        # verified evidence) tempts the model to abandon the abstention and answer.
        is_abstention = bool(csv_path) and Path(csv_path).stem.endswith("_abs")
        trace["is_abstention"] = is_abstention
        # The skill library (driven by question shape) takes precedence; only on a
        # miss does it fall back to the category hint
        hint = ""
        if p.get("grep_agent_use_skills", False):
            from experiment.agent_filter.skills import select_skills
            matched = select_skills(question)
            trace["skills"] = [n for n, _ in matched]
            hint = "\n\n".join(s for _, s in matched)
        if not hint:
            hint = CATEGORY_HINTS.get(category, "")

        # VECTOR tool: enabled only when this question's summaries VDB is present
        # (the agent decides for itself when to search semantically -- unlike the
        # disproven gap_vector approach where the verifier pushed candidates, here
        # the agent pulls).
        vector_ok = (
            bool(p.get("grep_agent_vector_search", True))
            and artifact_dir is not None
            and (Path(artifact_dir) / "summaries_chroma").exists()
        )
        trace["vector_tool"] = vector_ok
        # Provenance: VECTOR results are discovery-only. A sid is verified only
        # after it appears in a raw GREP or READ result.
        verified_sids: set[str] = set()
        vector_candidate_sids: set[str] = set()
        trace["evidence_provenance"] = {}
        from experiment.agent_filter.prompts import (
            VECTOR_TOOL_BLOCK, _active_hypothesis_block,
        )
        emit_hyp = bool(int(p.get("grep_agent_emit_hypothesis", 0)))
        system = SYSTEM_PROMPT.format(
            max_calls=max_calls,
            vector_tool=VECTOR_TOOL_BLOCK if vector_ok else "",
            hypothesis_line=_active_hypothesis_block() if emit_hyp else "",
        )
        if mode == "filter":
            system += "\nIMPORTANT: you may only KEEP or DROP candidates; do not add new sids in FINAL."
        user = USER_TEMPLATE.format(
            question=question,
            date_line=f"QUESTION DATE: {question_date}\n" if question_date else "",
            hint_line=f"{hint}\n" if hint else "",
            graph_context=(
                "GRAPH FACTS (Entities/Relationships; use as supporting evidence):\n"
                + graph_context
                if include_filter_graph and graph_context else ""
            ),
            candidates=_candidates_block(corpus, seed),
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        def _extract_final(arg: str, full_reply: str = "") -> list[str]:
            """Pull the agent's final sid list out of its reply.

            The reply is free text and the model varies how it presents the list -- a
            line after "FINAL:", a bullet list, or both. Accepting all the observed
            shapes matters because a parse miss reads as the agent having selected
            nothing.
            """
            sids = _SID_RE.findall(arg) + [
                s for s in re.split(r"[,\s]+", _SID_RE.sub("", arg)) if s and ":" in s
            ]
            if not sids and full_reply:
                # Empty FINAL argument: the sids may sit on other lines (a newline
                # and bullet list after "FINAL:", and so on)
                sids = _SID_RE.findall(full_reply) + [
                    s.strip("*•-,.")
                    for s in re.split(r"[,\s]+", _SID_RE.sub("", full_reply))
                    if ":" in s and re.search(r":\d+", s)
                ]
            return sids

        def _parse_response(resp):
            """Parse a response without feeding hidden reasoning back to the model."""
            candidates = _response_command_candidates(resp)
            raw_reply = candidates[0][1] if candidates else ""
            for source, candidate in candidates:
                cmd = _parse_command(candidate)
                if cmd is not None:
                    return raw_reply, candidate, cmd, source
            return raw_reply, raw_reply, None, None

        def _run_loop(budget: int) -> list[str] | None:
            """Run one GREP/READ->FINAL tool loop (shared by the main search and the
            verify top-up search)."""
            parse_failures = 0
            repeat_count = 0
            prev_cmd: tuple[str, str] | None = None
            for _ in range(budget):
                _t0 = time.perf_counter()
                resp = llm.chat(messages=messages, temperature=0.0, max_tokens=1024)
                diag = _resp_diag(resp)
                raw_reply, reply, cmd, source = _parse_response(resp)
                if source:
                    diag["command_source"] = source
                # Only retain the compact command in conversation history.
                # Raw reasoning is diagnostic data, not useful context, and
                # replaying it would inflate the next request's input tokens.
                messages.append({
                    "role": "assistant",
                    "content": f"{cmd[0]} {cmd[1]}" if cmd else "",
                })
                if cmd is None:
                    parse_failures += 1
                    trace["commands"].append({"cmd": "PARSE_FAIL", "arg": raw_reply[:200],
                                              "ms": round((time.perf_counter() - _t0) * 1000),
                                              **diag})
                    if parse_failures >= 2:
                        return None
                    messages.append({"role": "user", "content":
                        "Could not parse a command. Reply with exactly one command as the "
                        "last line: GREP <regex> | READ <sid> [k] | FINAL <sid> ..."})
                    continue

                kind, arg = cmd
                if kind == "FINAL":
                    if emit_hyp:
                        # The agent's self-reported answer hypothesis (productionizing
                        # "hypothesis recovery", replacing hyp-v1's after-the-fact
                        # 4o-mini extraction). Look for a HYPOTHESIS: line in the
                        # reply; failing that, fall back to the whole reasoning block
                        # (reasoning_content or reply) for downstream use.
                        # Capture HYPOTHESIS only to end of line; if a FINAL or sid
                        # token follows on the same or an adjacent line (the agent
                        # writes both together), cut before FINAL so the FINAL line's
                        # sids are not swallowed into the hypothesis (seen in hyp-v1
                        # 06db6396 and the 120b filter).
                        hm = re.search(r"HYPOTHESIS\s*[::]\s*([^\n]+)", reply, re.IGNORECASE)
                        hyp = hm.group(1).strip() if hm else ""
                        hyp = re.split(r"\bFINAL\b", hyp, maxsplit=1, flags=re.IGNORECASE)[0].strip()
                        if hyp and hyp.upper() != "NONE":
                            trace["hypothesis"] = hyp[:200]
                    trace["commands"].append({"cmd": "FINAL", "arg": arg[:500],
                                              "reply": reply[:1200],
                                              "ms": round((time.perf_counter() - _t0) * 1000),
                                              **diag})
                    return _extract_final(arg, reply)

                # Broken-record circuit breaker: repeating the same command prompts
                # for closure, and three in a row jumps straight to a forced FINAL
                if cmd == prev_cmd:
                    repeat_count += 1
                    if repeat_count >= 2:
                        trace["commands"].append({"cmd": "REPEAT_BREAK", "arg": f"{kind} {arg}"[:200],
                                                  "ms": round((time.perf_counter() - _t0) * 1000)})
                        return None
                    messages.append({"role": "user", "content":
                        "You already ran that exact command. Try DIFFERENT keywords, "
                        "or reply FINAL <sid> ... with your current best selection."})
                    continue
                prev_cmd = cmd
                repeat_count = 0
                if kind == "GREP":
                    result = corpus.grep(arg, max_lines=grep_max_lines)
                    verified_sids.update(corpus.normalize_sids(_SID_RE.findall(result)))
                elif kind == "VECTOR":
                    if vector_ok:
                        result = _vector_search(
                            artifact_dir, corpus, arg,
                            exclude=set(corpus.normalize_sids(seed)),
                            topn=int(p.get("grep_agent_vector_topn", 8)),
                            min_score=float(p.get("grep_agent_vector_min_score", 0.30)),
                        )
                        _vhits = corpus.normalize_sids(_SID_RE.findall(result))
                        vector_candidate_sids.update(_vhits)
                        # A VECTOR hit counts as verified outright, on a par with
                        # GREP/READ. The provenance gate is gone: whatever is pulled
                        # back is trusted, with no second verification required.
                        verified_sids.update(_vhits)
                    else:
                        result = "VECTOR is not available for this question; use GREP or READ."
                else:  # READ
                    sid, k = arg.rsplit(" ", 1)
                    result = corpus.read_window(sid, k=int(k))
                    verified_sids.update(corpus.normalize_sids(_SID_RE.findall(result)))
                trace["commands"].append({"cmd": kind, "arg": arg[:300], "result_chars": len(result),
                                          "reply": reply[:1200], "result": result[:1500],
                                          "ms": round((time.perf_counter() - _t0) * 1000),
                                          **diag})
                # The closing reminder lives permanently in the recent context --
                # after several search rounds models routinely forget how to finish.
                # It states outright that a partial FINAL is acceptable: 49 of 73
                # fallbacks were aggregation questions burning every round because
                # they "could not gather every instance and dared not submit". The
                # adjudication layer fills gaps anyway, so completeness is not the
                # goal here.
                result += ("\n\n(When you have identified the evidence, reply with one line: "
                           "FINAL <sid> <sid> ... — copy sids exactly. A PARTIAL set is "
                           "acceptable: FINAL the turns you have confirmed so far — a separate "
                           "audit step recovers anything you miss. Do not keep searching for "
                           "completeness.)")
                messages.append({"role": "user", "content": result})
            return None

        final_raw = _run_loop(max_calls)

        _SALVAGE_MSGS = [
            "STOP searching. Reply NOW with only one line listing the selected evidence "
            "sids, copied EXACTLY from this list (or ones you found via GREP):\n{seeds}\n"
            "FINAL <sid> <sid> ...",
            "Output ONLY the single line below, filled in with sids from this list — "
            "no other text, no tool calls:\n{seeds}\nFINAL <sid> <sid> ...",
        ]

        def _ask_final(attempt: int) -> list[str]:
            """Force closure: attach the candidate sid list for the model to copy
            from, then extract the sids."""
            messages.append({"role": "user", "content":
                _SALVAGE_MSGS[min(attempt, 1)].format(seeds=" ".join(seed))})
            _t0 = time.perf_counter()
            resp = llm.chat(messages=messages, temperature=0.0, max_tokens=512)
            diag = _resp_diag(resp)
            raw_reply, reply, cmd, source = _parse_response(resp)
            if source:
                diag["command_source"] = source
            messages.append({
                "role": "assistant",
                "content": f"{cmd[0]} {cmd[1]}" if cmd else "",
            })
            arg = cmd[1] if cmd and cmd[0] == "FINAL" else ""
            out = _extract_final(arg, reply)
            trace["commands"].append({"cmd": "FINAL(forced)", "arg": (arg or reply)[:500],
                                      "reply": reply[:1200],
                                      "ms": round((time.perf_counter() - _t0) * 1000),
                                      **diag})
            return out

        if not (final_raw and corpus.normalize_sids(final_raw)):
            # No FINAL / empty FINAL / unparseable sids -> prompt once for closure.
            # A second prompt is useless (measured: 129 of 138 times the model still
            # returns a tool call). Narrowing the context down to the verified hits
            # also tested negative (salvage group 54 -> 43%): a narrow but
            # "trustworthy" context actually tempts the answering model to invent,
            # whereas the noise of all 16 entries acts as cover on hard questions.
            if final_raw:
                trace["final_raw_unresolved"] = final_raw[:20]
            final_raw = _ask_final(0)

        # Uncertainty signal: an agent refusing to FINAL on its own means it found
        # no confirmable evidence for an answer (_abs abstention questions fall back
        # 70% of the time). The hint is a conditional abstention prompt, attached
        # only to the full-context fallback path -- "narrowed context + hint" tested
        # negative (ordinary questions 46.7 -> 33.3).
        abstain = False
        if not (final_raw and corpus.normalize_sids(final_raw)):
            # Forced verified->FINAL: when the agent will not close on its own, take
            # the verified sids it actually confirmed via GREP/READ, treat them as
            # the FINAL, and run the full finalize pipeline (adjudicate + floor +
            # rebuild) rather than falling back to the whole raw context marked as a
            # fallback. Rationale:
            #   - Raising max-calls disproved "force the agent to submit a narrowed
            #     context" (-6): a bare 1-2 sid FINAL strips the answering model of
            #     the full-context noise cover and it starts inventing.
            #   - verified->FINAL is different: aggregation questions routinely
            #     verify >16 entries (the agent has GREPed dozens of turns), so after
            #     the cap the size is about the same as the full context, merely
            #     reordered verified-first. Questions that searched up empty have
            #     verified=0 -> fall back to the full seed set, keeping the cover.
            #     Neither end produces a bare narrowed context.
            #   - Going through finalize_from_raw preserves adjudication's recovery
            #     of topically relevant seeds plus the floor pad back to 12,
            #     provenance stays intact, and the no_final marker disappears.
            # Gate: narrow via finalize only when the agent's verified evidence is
            # plentiful enough. Insufficient verified evidence -- including _abs
            # questions that searched up empty, and aggregation questions that only
            # scraped together a few entries -- keeps the full protective context
            # (fvf-73: all the harm from narrowing landed on low-verified _abs
            # questions, while every question with lots of verified evidence was safe
            # or improved). The threshold defaults to 12, i.e. "at least a full
            # context's worth of confirmed evidence" before it is trusted to replace
            # the full set. Note: the evidence_floor blind pad was retired on
            # 2026-07-20, so its value is no longer borrowed here and the fallback is
            # written as a literal 12.
            verified_norm = corpus.normalize_sids(list(verified_sids))
            fvf_min = int(p.get("grep_agent_force_verified_min", 12))
            # Narrowing requires: the flag on, a non-abstention question, and enough
            # verified evidence. Fail any one of those -> keep the full context.
            if (int(p.get("grep_agent_force_verified_final", 0))
                    and not is_abstention
                    and len(verified_norm) >= fvf_min):
                trace["forced_verified_final"] = verified_norm
                return finalize_from_raw(
                    final_raw=verified_norm,
                    context=context, corpus=corpus, seed=seed,
                    verified_sids=verified_sids,
                    vector_candidate_sids=vector_candidate_sids,
                    question=question, question_date=question_date, category=category,
                    llm=llm, p=p, artifact_dir=artifact_dir, trace=trace,
                )
            trace["fallback"] = "no_final"
            trace["verified_sids"] = corpus.normalize_sids(list(verified_sids))
            if bool(p.get("grep_agent_abstention_hint", 0)):
                trace["abstention_hint"] = True
                return context + ABSTENTION_HINT, trace
            return context, trace

        final = corpus.normalize_sids(final_raw)
        if mode == "filter":
            seed_set = set(corpus.normalize_sids(seed))
            final = [s for s in final if s in seed_set]
        elif mode == "fetch_only":
            # Add without cutting: keep the baseline context's serendipity while
            # taking the recall the agent digs up.
            # Measured on LoCoMo: the agent's all-gold-hit rate rose 19.8pp, but
            # cutting useful non-gold content cancelled the gain -- hence this mode
            # decouples the two.
            seed_norm_ = corpus.normalize_sids(seed)
            final = seed_norm_ + [s for s in final if s not in set(seed_norm_)]

        # The provenance gate was removed on 2026-07-22: a VECTOR hit now counts as
        # verified just like GREP/READ (see the VECTOR branch), and evidence pulled
        # back by GREP/READ/VECTOR is trusted across the board. The only unverified
        # things left are hallucinated sids, which _rebuild_context drops naturally
        # because the corpus cannot resolve them.
        trace["verified_sids"] = corpus.normalize_sids(list(verified_sids))
        trace["vector_candidate_sids"] = corpus.normalize_sids(list(vector_candidate_sids))
        seed_set = set(corpus.normalize_sids(seed))
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
        final = final[:max_sids]
        if not final:
            trace["fallback"] = "empty_final"
            return context, trace

        # ── Sufficiency loop: when the verifier rules the evidence insufficient,
        # search again carrying a gap hint. Additive only, never removes. ──
        verify_rounds = int(p.get("grep_agent_verify_rounds", 0))
        verify_budget = int(p.get("grep_agent_verify_max_calls", 4))
        verify_cats = p.get("grep_agent_verify_categories")
        if verify_cats is not None and category not in verify_cats:
            verify_rounds = 0  # Selective: skip non-aggregation categories, where verify only dilutes
        trace["sufficiency"] = []
        for vr in range(verify_rounds):
            try:
                ok, missing = _check_sufficiency(
                    _verify_llm(llm), question=question, question_date=question_date,
                    corpus=corpus, sids=final,
                )
            except Exception as exc:  # a verifier crash must not disturb the main flow
                trace["sufficiency"].append({"round": vr, "error": str(exc)[:200]})
                break
            trace["sufficiency"].append({"round": vr, "sufficient": ok, "missing": missing[:300]})
            if ok:
                break
            gap_msg = GAP_HINT_TEMPLATE.format(missing=missing)
            # Vector top-up search: grep's repair arm comes back empty roughly 87% of
            # the time because of the paraphrase gap, so embed the gap description,
            # search the summaries VDB, and hand the semantic neighbours to the agent
            # to confirm.
            gap_topn = int(p.get("grep_agent_gap_vector_topn", 0))
            if artifact_dir is not None and gap_topn > 0:
                from experiment.agent_filter.gap_vector import vector_gap_candidates
                cands = vector_gap_candidates(
                    artifact_dir,
                    f"{question}\n{missing}",
                    exclude=set(final),
                    topn=gap_topn,
                    min_score=float(p.get("grep_agent_gap_vector_min_score", 0.30)),
                )
                trace["sufficiency"][-1]["vector_cands"] = [s for s, _ in cands]
                if cands:
                    lines = []
                    for s, sc in cands:
                        entry = corpus.display_entry(s, max_chars=200) or "(text unavailable)"
                        lines.append(f"[sid={s}] (score={sc:.2f}) {entry}")
                    gap_msg += (
                        "\n\nA semantic search for the missing information surfaced these "
                        "candidate turns (NOT yet verified — check with READ/GREP before "
                        "including):\n" + "\n".join(lines)
                    )
            messages.append({"role": "user", "content": gap_msg})
            extra_raw = _run_loop(verify_budget)
            if not extra_raw:
                break
            extra = corpus.normalize_sids(extra_raw)
            if mode == "filter":
                extra = [s for s in extra if s in set(corpus.normalize_sids(seed))]
            # Monotonic: a verify round may only add, never touch what is chosen
            added_now = [s for s in extra if s not in set(final)]
            trace["sufficiency"][-1]["added"] = added_now
            if not added_now:
                break
            final = (final + added_now)[:max_sids]

        seed_norm = corpus.normalize_sids(seed)

        # ── Answer-blind per-item adjudication: the agent's FINAL is an "answer
        # citation" (the minimal-citation instinct: solve first, then keep only the
        # smallest turn set containing the answer span), so supporting evidence
        # without the answer span is discarded systematically -- the root of the
        # preference and multi-hop failures.
        # The remedy: an independent adjudication call (blind to the agent's
        # conversation, so it does not know the "answer") rules KEEP/DROP on each
        # discarded seed, judging topical relevance to the question. KEEPs are added
        # back to final (additive only; the agent's own 0.84-precision picks are left
        # alone). When adjudication succeeds it replaces the evidence_floor blind pad
        # -- the floor refills in rerank order, which cannot recover preference cues.
        adj_on = int(p.get("grep_agent_adjudicate", 0))
        adj_cats = p.get("grep_agent_adjudicate_categories")
        if adj_cats is not None and category not in adj_cats:
            adj_on = 0
        # KEEP-all categories: the gold for KU/temporal holds many supporting turns
        # that carry no answer but are required for the reasoning (time anchors,
        # dated mentions), and 20B adjudication judging by "contains the answer"
        # DROPs them systematically -- the cause of bucket B. These categories switch
        # to recall-recovery-only: every discarded seed is added back without passing
        # through an LLM DROP. This differs from min-keep in being a category-level
        # "do not cut" rather than a question-shape trigger.
        keep_all_cats = p.get("grep_agent_adjudicate_keep_all_categories")
        adjudicated = False
        if adj_on and len(final) < max_sids:
            pending = [s for s in seed_norm if s not in set(final)]
            if pending and keep_all_cats and category in keep_all_cats:
                trace["adjudication"] = {"keep_all": True, "kept": pending, "dropped": []}
                final = (final + [s for s in pending if s not in set(final)])[:max_sids]
                adjudicated = True
            elif pending:
                _t0 = time.perf_counter()
                try:
                    kept_adj, verdicts = _adjudicate_candidates(
                        llm, question=question, question_date=question_date,
                        corpus=corpus, pending=pending,
                    )
                    verdicts["ms"] = round((time.perf_counter() - _t0) * 1000)
                    trace["adjudication"] = verdicts
                    final = (final + [s for s in kept_adj
                                      if s not in set(final)])[:max_sids]
                    adjudicated = True
                except Exception as exc:  # an adjudication crash must not disturb the main flow; the floor carries on
                    trace["adjudication"] = {"error": str(exc)[:200]}

        # ── Min-keep (question-driven, not category-specific): aggregation and
        # latest-value questions (how many / how often / total / most recent /
        # current...) need every dated mention of the target fact present at once --
        # counting must be complete, and picking the latest requires something to
        # compare. When the agent cuts too thin, refill from the seeds in rerank
        # order. Triggered by question shape, so it generalizes to any dataset.
        min_keep = int(p.get("grep_agent_min_keep_aggregation", 0))
        if min_keep and len(final) < min_keep and _AGG_QUESTION_RE.search(question):
            pad = [s for s in seed_norm if s not in set(final)]
            trace["min_keep_padded"] = pad[: min_keep - len(final)]
            final = final + pad[: min_keep - len(final)]

        # ── The evidence_floor blind pad was retired on 2026-07-20 ─────────────
        # It padded `final` up to a floor in rerank order, which overrode the
        # agent's per-item decision with a ranking signal the agent had already
        # seen and rejected. It moved no accuracy, and it made "kept" ambiguous:
        # a padded sid looks identical to an adjudicated one in the trace.
        # grep_agent_evidence_floor now defaults to 0; see its note in
        # experiment_config. Recover the implementation from git if it is ever
        # revisited -- it should not come back without a metric to justify it.

        trace["final_sids"] = final
        trace["kept"] = [s for s in final if s in set(seed_norm)]
        trace["added"] = [s for s in final if s not in set(seed_norm)]
        trace["dropped"] = [s for s in seed_norm if s not in set(final)]

        # Safety net: if the selector dropped every upstream summary, keep the
        # original 16-summary context instead of answering from agent-fetched
        # evidence alone.  This is distinct from a no_final/exception
        # fallback: the agent completed, but produced zero retained seeds.
        if not trace["kept"]:
            trace["fallback"] = "zero_keep"
            trace["context_sids"] = seed_norm
            if abstain:
                return context + ABSTENTION_HINT, trace
            return context, trace

        if mode == "fetch_only":
            # Pure append: the original context is left word for word (baseline
            # evidence may be plain text with no sid, which rebuild would wrongly
            # delete), with the newly fetched units hung on the end. Guarantees the
            # information is never less than baseline.
            if not trace["added"]:
                trace["fallback"] = "no_addition"
                return context, trace
            lines = ["", "### Additional Evidence (agent-retrieved)"]
            ctx_sids = list(seed_norm)
            for s in trace["added"]:
                t = corpus.resolve(s)[0]
                entry = corpus.display_entry(s)
                dt = f"[{t.date}]" if t.date else ""
                lines.append(f"  • {dt}[sid={s}][score=--] {entry} ")
                ctx_sids.append(s)
            trace["context_sids"] = ctx_sids
            return context.rstrip("\n") + "\n".join(lines), trace

        refined, context_sids = _rebuild_context(
            context, corpus, final,
            include_pair=bool(p.get("grep_agent_include_pair", True)),
            include_prefix=include_answer_graph,
        )
        trace["context_sids"] = context_sids
        if abstain:
            refined += ABSTENTION_HINT
        return refined, trace

    except Exception:
        trace["fallback"] = "exception"
        trace["error"] = traceback.format_exc()[-2000:]
        return context, trace


def finalize_from_raw(
    *,
    final_raw,
    context: str,
    corpus: Corpus,
    seed: list[str],
    verified_sids: set,
    vector_candidate_sids: set,
    question: str,
    question_date: str | None,
    category: str | None,
    llm,
    p: dict,
    artifact_dir,
    trace: dict,
) -> tuple[str, dict]:
    """The v1 back half of the pipeline (provenance gate -> filter_fetch ->
    adjudicate -> floor -> rebuild) pulled out into its own function, so the
    planner-worker harness can reuse the same v1 mainline logic.

    Behaviour matches lines 811-1011 of refine_context, with one difference: the
    sufficiency loop needs v1's _run_loop, which planner-worker does not have, so
    this path skips sufficiency (v1 defaults to verify_rounds=0, so the mainline
    is unaffected). A None or empty final_raw falls back to the original context."""
    mode = p.get("grep_agent_mode", "filter_fetch")
    max_sids = int(p.get("grep_agent_max_sids", 16))
    include_answer_graph = bool(p.get("grep_agent_answer_include_graph_context", True))

    if not (final_raw and corpus.normalize_sids(final_raw)):
        trace["fallback"] = "no_final"
        trace["verified_sids"] = corpus.normalize_sids(list(verified_sids))
        return context, trace

    final = corpus.normalize_sids(final_raw)
    if mode == "filter":
        seed_set = set(corpus.normalize_sids(seed))
        final = [s for s in final if s in seed_set]
    elif mode == "fetch_only":
        seed_norm_ = corpus.normalize_sids(seed)
        final = seed_norm_ + [s for s in final if s not in set(seed_norm_)]

    # The provenance gate was removed on 2026-07-22: a VECTOR hit counts as
    # verified, and anything fetched is trusted.
    trace["verified_sids"] = corpus.normalize_sids(list(verified_sids))
    trace["vector_candidate_sids"] = corpus.normalize_sids(list(vector_candidate_sids))
    seed_set = set(corpus.normalize_sids(seed))
    trace["evidence_provenance"] = {
        s: ("seed+verified" if s in seed_set and s in verified_sids
            else "seed" if s in seed_set
            else "verified" if s in verified_sids
            else "unverified")
        for s in final
    }
    trace["final_before_cap"] = list(final)
    final = final[:max_sids]
    if not final:
        trace["fallback"] = "empty_final"
        return context, trace

    seed_norm = corpus.normalize_sids(seed)

    # Answer-blind per-item adjudication (the v1 mainline; adjudicate adds back the
    # discarded but topically relevant seeds)
    adj_on = int(p.get("grep_agent_adjudicate", 0))
    adj_cats = p.get("grep_agent_adjudicate_categories")
    if adj_cats is not None and category not in adj_cats:
        adj_on = 0
    keep_all_cats = p.get("grep_agent_adjudicate_keep_all_categories")
    adjudicated = False
    if adj_on and len(final) < max_sids:
        pending = [s for s in seed_norm if s not in set(final)]
        if pending and keep_all_cats and category in keep_all_cats:
            trace["adjudication"] = {"keep_all": True, "kept": pending, "dropped": []}
            final = (final + [s for s in pending if s not in set(final)])[:max_sids]
            adjudicated = True
        elif pending:
            _t0 = time.perf_counter()
            try:
                kept_adj, verdicts = _adjudicate_candidates(
                    llm, question=question, question_date=question_date,
                    corpus=corpus, pending=pending,
                )
                verdicts["ms"] = round((time.perf_counter() - _t0) * 1000)
                trace["adjudication"] = verdicts
                final = (final + [s for s in kept_adj if s not in set(final)])[:max_sids]
                adjudicated = True
            except Exception as exc:
                trace["adjudication"] = {"error": str(exc)[:200]}

    min_keep = int(p.get("grep_agent_min_keep_aggregation", 0))
    if min_keep and len(final) < min_keep and _AGG_QUESTION_RE.search(question):
        pad = [s for s in seed_norm if s not in set(final)]
        trace["min_keep_padded"] = pad[: min_keep - len(final)]
        final = final + pad[: min_keep - len(final)]

    # The evidence_floor blind pad was retired here too, for the same reason as
    # in the batch path above. min_keep padding stays: it is a floor on the
    # agent's own selection, not a rerank-ordered override of it.

    trace["final_sids"] = final
    trace["kept"] = [s for s in final if s in set(seed_norm)]
    trace["added"] = [s for s in final if s not in set(seed_norm)]
    trace["dropped"] = [s for s in seed_norm if s not in set(final)]

    if not trace["kept"]:
        trace["fallback"] = "zero_keep"
        trace["context_sids"] = seed_norm
        return context, trace

    if mode == "fetch_only":
        if not trace["added"]:
            trace["fallback"] = "no_addition"
            return context, trace
        lines = ["", "### Additional Evidence (agent-retrieved)"]
        ctx_sids = list(seed_norm)
        for s in trace["added"]:
            t = corpus.resolve(s)[0]
            entry = corpus.display_entry(s)
            dt = f"[{t.date}]" if t.date else ""
            lines.append(f"  • {dt}[sid={s}][score=--] {entry} ")
            ctx_sids.append(s)
        trace["context_sids"] = ctx_sids
        return context.rstrip("\n") + "\n".join(lines), trace

    refined, context_sids = _rebuild_context(
        context, corpus, final,
        include_pair=bool(p.get("grep_agent_include_pair", True)),
        include_prefix=include_answer_graph,
    )
    trace["context_sids"] = context_sids
    return refined, trace


_agent_llm_cache = None
_verify_llm_cache = None


def _agent_llm(default_llm):
    """GREP_AGENT_LLM_API / GREP_AGENT_MODEL_NAME can point at a different endpoint
    (following the JUDGE_* convention); when unset, the answering LLM passed in is
    shared."""
    global _agent_llm_cache
    base = os.getenv("GREP_AGENT_LLM_API")
    name = os.getenv("GREP_AGENT_MODEL_NAME")
    if not (base or name):
        return default_llm
    if _agent_llm_cache is None:
        from grace_mem.llm import LLMClient
        _agent_llm_cache = LLMClient(base_url=base or None, model_name=name or None)
    return _agent_llm_cache


def _verify_llm(default_llm):
    """GREP_AGENT_VERIFY_LLM_API / GREP_AGENT_VERIFY_MODEL_NAME can point the
    sufficiency verifier alone at a different endpoint (120B, say) without
    affecting the agent loop or answering. One of the main reasons v3-v6 were
    closed out was the oss-20b verifier's 43% misjudgement rate; this hook exists
    to retest with a stronger verifier."""
    global _verify_llm_cache
    base = os.getenv("GREP_AGENT_VERIFY_LLM_API")
    name = os.getenv("GREP_AGENT_VERIFY_MODEL_NAME")
    if not (base or name):
        return default_llm
    if _verify_llm_cache is None:
        from grace_mem.llm import LLMClient
        _verify_llm_cache = LLMClient(base_url=base or None, model_name=name or None, timeout=300.0)
    return _verify_llm_cache


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
        llm=_agent_llm(llm),
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
            from grace_mem.utils.error_analysis import append_analysis_record
            append_analysis_record(log_dir, "grep_agent", {"question": question, **trace})
        except Exception as exc:  # a logging failure must not affect answering
            print(f"[QA] Grep agent trace logging failed: {exc}")
    return refined
