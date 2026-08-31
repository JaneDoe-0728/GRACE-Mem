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

import re
import time
import traceback
from pathlib import Path

from experiment.agent_filter import vector_search
from experiment.agent_filter.context import (
    candidates_block,
    graph_context_from_context,
    rebuild_context,
    seed_scores_from_context,
    seed_sids_from_context,
)
from experiment.agent_filter.corpus import Corpus, load_corpus
from experiment.agent_filter.llm_factory import agent_llm, verify_llm
from experiment.agent_filter.models import SID_RE, Command
from experiment.agent_filter.prompting.agent import (
    ABSTENTION_HINT,
    CATEGORY_HINTS,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
)
from experiment.agent_filter.prompting.verification import (
    GAP_HINT_TEMPLATE,
    SUFFICIENCY_SYSTEM,
    SUFFICIENCY_USER,
)
from experiment.agent_filter.protocol import extract_final_sids, parse_response

# Detects aggregation/latest-value questions (a question-driven trigger for the
# retention strategy, independent of any dataset category label)
_AGG_QUESTION_RE = re.compile(
    r"\b(how many|how much|how often|how long|how frequently|total|count|sum|"
    r"number of|most recent(ly)?|latest|currently|current|in total|altogether)\b",
    re.IGNORECASE,
)


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
    hits = vector_search.search_summaries(
        artifact_dir, query, exclude=exclude, topn=topn, min_score=min_score,
    )
    return vector_search.render_hits(corpus, query, hits)


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
    from experiment.agent_filter.prompting.adjudication import (
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
            from experiment.agent_filter.prompting.skills import select_skills
            matched = select_skills(question)
            trace["skills"] = [n for n, _ in matched]
            hint = "\n\n".join(s for _, s in matched)
        if not hint:
            hint = CATEGORY_HINTS.get(category, "")

        # VECTOR tool: enabled only when this question's summaries VDB is present
        # (the agent decides for itself when to search semantically -- unlike the
        # disproven gap-repair approach where the verifier pushed candidates, here
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
        from experiment.agent_filter.prompting.agent import (
            VECTOR_TOOL_BLOCK,
            active_hypothesis_block,
        )
        emit_hyp = bool(int(p.get("grep_agent_emit_hypothesis", 0)))
        system = SYSTEM_PROMPT.format(
            max_calls=max_calls,
            vector_tool=VECTOR_TOOL_BLOCK if vector_ok else "",
            hypothesis_line=active_hypothesis_block() if emit_hyp else "",
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
            candidates=candidates_block(corpus, seed),
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        def _run_loop(budget: int) -> list[str] | None:
            """Run one GREP/READ->FINAL tool loop (shared by the main search and the
            verify top-up search)."""
            parse_failures = 0
            repeat_count = 0
            prev_cmd: Command | None = None
            for _ in range(budget):
                _t0 = time.perf_counter()
                resp = llm.chat(messages=messages, temperature=0.0, max_tokens=1024)
                parsed = parse_response(resp)
                cmd, reply, raw_reply, diag = (
                    parsed.command, parsed.reply, parsed.raw_reply, parsed.diagnostics,
                )
                # Only retain the compact command in conversation history.
                # Raw reasoning is diagnostic data, not useful context, and
                # replaying it would inflate the next request's input tokens.
                messages.append({
                    "role": "assistant",
                    "content": f"{cmd.kind} {cmd.arg}" if cmd else "",
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

                kind, arg = cmd.kind, cmd.arg
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
                    return extract_final_sids(arg, reply)

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
                    verified_sids.update(corpus.normalize_sids(SID_RE.findall(result)))
                elif kind == "VECTOR":
                    if vector_ok:
                        result = _vector_search(
                            artifact_dir, corpus, arg,
                            exclude=set(corpus.normalize_sids(seed)),
                            topn=int(p.get("grep_agent_vector_topn", 8)),
                            min_score=float(p.get("grep_agent_vector_min_score", 0.30)),
                        )
                        _vhits = corpus.normalize_sids(SID_RE.findall(result))
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
                    verified_sids.update(corpus.normalize_sids(SID_RE.findall(result)))
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

        _salvage_msgs = [
            ("STOP searching. Reply NOW with only one line listing the selected evidence "
            "sids, copied EXACTLY from this list (or ones you found via GREP):\n{seeds}\n"
            "FINAL <sid> <sid> ..."),
            ("Output ONLY the single line below, filled in with sids from this list — "
            "no other text, no tool calls:\n{seeds}\nFINAL <sid> <sid> ..."),
        ]

        def _ask_final(attempt: int) -> list[str]:
            """Force closure: attach the candidate sid list for the model to copy
            from, then extract the sids."""
            messages.append({"role": "user", "content":
                _salvage_msgs[min(attempt, 1)].format(seeds=" ".join(seed))})
            _t0 = time.perf_counter()
            resp = llm.chat(messages=messages, temperature=0.0, max_tokens=512)
            parsed = parse_response(resp)
            cmd, reply, diag = parsed.command, parsed.reply, parsed.diagnostics
            messages.append({
                "role": "assistant",
                "content": f"{cmd.kind} {cmd.arg}" if cmd else "",
            })
            arg = cmd.arg if cmd and cmd.kind == "FINAL" else ""
            out = extract_final_sids(arg, reply)
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
                    llm=llm, p=p, trace=trace,
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
                    verify_llm(llm), question=question, question_date=question_date,
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
                cands = vector_search.search_summaries(
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
        if adj_on and len(final) < max_sids:
            pending = [s for s in seed_norm if s not in set(final)]
            if pending and keep_all_cats and category in keep_all_cats:
                trace["adjudication"] = {"keep_all": True, "kept": pending, "dropped": []}
                final = (final + [s for s in pending if s not in set(final)])[:max_sids]
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
            evidence_sids = list(seed_norm)
            for s in trace["added"]:
                t = corpus.resolve(s)[0]
                entry = corpus.display_entry(s)
                dt = f"[{t.date}]" if t.date else ""
                lines.append(f"  • {dt}[sid={s}][score=--] {entry} ")
                evidence_sids.append(s)
            trace["context_sids"] = evidence_sids
            return context.rstrip("\n") + "\n".join(lines), trace

        refined, context_sids = rebuild_context(
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
    if adj_on and len(final) < max_sids:
        pending = [s for s in seed_norm if s not in set(final)]
        if pending and keep_all_cats and category in keep_all_cats:
            trace["adjudication"] = {"keep_all": True, "kept": pending, "dropped": []}
            final = (final + [s for s in pending if s not in set(final)])[:max_sids]
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
        evidence_sids = list(seed_norm)
        for s in trace["added"]:
            t = corpus.resolve(s)[0]
            entry = corpus.display_entry(s)
            dt = f"[{t.date}]" if t.date else ""
            lines.append(f"  • {dt}[sid={s}][score=--] {entry} ")
            evidence_sids.append(s)
        trace["context_sids"] = evidence_sids
        return context.rstrip("\n") + "\n".join(lines), trace

    refined, context_sids = rebuild_context(
        context, corpus, final,
        include_pair=bool(p.get("grep_agent_include_pair", True)),
        include_prefix=include_answer_graph,
    )
    trace["context_sids"] = context_sids
    return refined, trace


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
            from grace_mem.runtime.analysis_log import append_analysis_record
            append_analysis_record(log_dir, "grep_agent", {"question": question, **trace})
        except Exception as exc:  # a logging failure must not affect answering
            print(f"[QA] Grep agent trace logging failed: {exc}")
    return refined
