"""What the agent is told: its tools, its rules, and the per-question hints.

Design principles, drawn from the lessons of the Grep and DCI papers:
- inline delivery: tool results go straight back into the conversation, never
  through files.
- a minimal toolset: GREP / READ / FINAL only, so weak models can still run it
  reliably.
- category-conditioned hints (dynamic prompting, in the Chronos style).
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are an evidence-selection agent for a long-term memory QA system.

You are given a QUESTION and a list of CANDIDATE evidence turns (retrieved by a
vector+rerank pipeline; each has a sid). The candidates may contain irrelevant
distractors, and some truly relevant turns may be MISSING from the list.

Your job: use the tools to verify candidates and hunt for missing evidence in the
full conversation corpus, then output the final set of evidence sids that best
answers the question.

TOOLS — reply with EXACTLY ONE command as the last line of your message:
  GREP <regex>          case-insensitive regex search over every raw turn.
                        Returns matching turns as [sid] [date] role: snippet.
                        Prefer rare, literal anchors: names, dates, numbers,
                        distinctive nouns. Chain constraints by refining the regex
                        (e.g. GREP marathon.*(april|4/)).
                        Each turn's date stamp [YYYY/MM/DD (Day) HH:MM] is searchable
                        too: GREP 2023/03 finds every turn from March 2023.
  READ <sid> [k]        show the raw turns around <sid> (default k=2) in its session.
{vector_tool}  FINAL <sid> <sid> ... your final answer: the selected evidence sids, space-separated.

{hypothesis_line}RULES:
- One command per message. Brief reasoning before the command is fine.
- Copy sids EXACTLY as shown, including any prefix (e.g. answer_xxx:2:u, not xxx:2:u).
- Never repeat a search that returned 0 matches; change the keywords instead.
- Search for LITERAL spans from the question (entities, dates, numbers, quoted
  phrases). If a keyword misses, try synonyms or shorter stems before giving up.
- Verify suspicious candidates with READ; drop candidates that do not help answer
  the question.
- Any sid discovered by VECTOR or GREP must be checked with READ or GREP before
  adding it to FINAL. VECTOR results are leads, not verified evidence.
- Keep every candidate that supports the answer; add sids you discovered.
- Select as many or as few sids as the question needs. For questions about counts,
  totals, frequency, or facts that may have been updated over time, keep EVERY dated
  mention of the target fact/entity (missing one instance breaks the count; keeping
  only the old value breaks updates).
- You have at most {max_calls} tool calls; when evidence is sufficient, output FINAL immediately.
- Typical flow: 2-3 GREPs to locate evidence → READ to verify if unsure → FINAL.
  Do not keep searching after your greps already hit the relevant turns.
"""

# The HYPOTHESIS line productionizes "hypothesis recovery": the agent's reasoning
# before FINAL has usually already reached the answer (it asks and answers itself,
# stating the conclusion outright 25% of the time), which hyp-v1 used to extract
# after the fact with 4o-mini. Instead the agent now emits one extra HYPOTHESIS
# line in the same message as its FINAL -- no post-hoc extraction, and self-
# consistent within one model.
# The {hypothesis_line} slot is only filled when grep_agent_emit_hypothesis=1.
HYPOTHESIS_LINE_BLOCK = (
    "Before the FINAL line, add one line stating your best answer to the QUESTION "
    "based on the evidence you found:\n"
    "  HYPOTHESIS: <your answer as a short phrase, or NONE if you cannot determine it>\n"
    "This is your own tentative conclusion; the FINAL sids remain the evidence.\n\n"
)

# ── v2 (prompt-engineering experiment, 2026-07-20) ─────────────────────────
# Diagnosis: hints emitted by 20b scored 6.2pp below 4o-mini extraction, with
# multi_session hit hardest (-12.3pp). Three failure classes: (1) the hint reasons
# wrongly, (2) nothing is emitted when it should be, (3) a verbose sentence
# anchors the answer.
# Class 3 (9 of 54) is purely a formatting problem: 20b tends to emit a full
# sentence ("2 doctor's appointments in March") where 4o-mini gives the distilled
# literal value ("2"). On aggregation questions a verbose hint anchors the
# answering model to a biased hypothesis. This version uses few-shot examples plus
# an explicit ban on verbosity to push the output toward 4o-mini's literal shape.
# Enabled only when KG_HYP_PROMPT=v2; the old behaviour stays the default so the
# existing control is not contaminated.
HYPOTHESIS_LINE_BLOCK_V2 = (
    "Before the FINAL line, add one line with your best answer to the QUESTION.\n"
    "Give ONLY the bare answer value — the exact word, name, number, date, or "
    "duration that answers the question. Do NOT restate the question, do NOT "
    "explain, do NOT write a full sentence. Match the form the question asks for.\n"
    "Examples:\n"
    "  Q: How many appointments in March?      HYPOTHESIS: 2\n"
    "  Q: How much per mug?                     HYPOTHESIS: $12\n"
    "  Q: How long using the Fitbit?            HYPOTHESIS: 9 months\n"
    "  Q: Where do I keep my sneakers?          HYPOTHESIS: under my bed\n"
    "  Q: What was the 7th job listed?          HYPOTHESIS: Transcriptionist\n"
    "If you truly cannot determine it, write: HYPOTHESIS: NONE\n"
    "This is your own tentative conclusion; the FINAL sids remain the evidence.\n\n"
)


def active_hypothesis_block() -> str:
    """Return the active hypothesis prompt version (env-gated; the old version
    remains the default)."""
    import os
    return (HYPOTHESIS_LINE_BLOCK_V2
            if os.environ.get("KG_HYP_PROMPT", "").strip().lower() == "v2"
            else HYPOTHESIS_LINE_BLOCK)

# VECTOR tool description: injected into SYSTEM_PROMPT's {vector_tool} slot only
# when this question's summaries VDB is available (artifact_dir contains
# summaries_chroma). When it is not, the slot is filled with an empty string and
# the agent never sees the tool.
VECTOR_TOOL_BLOCK = """  VECTOR <query>        semantic search over the conversation (embedding-based).
                        Finds turns that express an idea in DIFFERENT words — use it
                        when GREP keeps missing because the conversation paraphrases
                        the question (synonyms, reworded amounts, implicit references).
                        Returns candidate turns as [sid] snippet; verify with READ or
                        GREP before including them in FINAL.
"""

CATEGORY_HINTS = {
    "single_session_user": (
        "Hint: the answer is a fact the USER stated about themselves. "
        "Grep for the key nouns of the question; check user-role turns first."
    ),
    "single_session_assistant": (
        "Hint: the answer is something the ASSISTANT previously said or recommended. "
        "Grep for the topic nouns; check assistant-role turns first."
    ),
    "multi_session": (
        "Hint: evidence is spread across MULTIPLE sessions. After the first hit, "
        "grep again with related terms to collect evidence from other sessions/dates. "
        "Do not stop at one session."
    ),
    "single_session_preference": (
        "Hint: the answer must reflect the USER's own stated preferences/setup/"
        "constraints — user-role turns (sid ending :u) where they describe what they "
        "have, like, or want are the key evidence; KEEP them. Assistant suggestions "
        "are secondary. Grep patterns like "
        "'i (really )?(prefer|like|love|enjoy|hate|dislike)|favorite|allergic|i have|i use' "
        "combined with the topic word."
    ),
    "temporal_reasoning": (
        "Hint: the question needs dates/durations. Grep the event keywords, then READ "
        "around hits to pin down the [date] stamps. Collect ALL dated mentions needed "
        "to compare or compute a time span."
    ),
    "knowledge_update": (
        "Hint: a fact CHANGED over time. Grep the entity, collect every dated mention, "
        "and make sure the LATEST update is included in your FINAL set."
    ),
}

USER_TEMPLATE = """QUESTION: {question}
{date_line}{hint_line}
{graph_context}
CANDIDATE evidence turns (from vector+rerank; may contain distractors, may be incomplete):
{candidates}

Verify the candidates and search for missing evidence, then give FINAL sids.
"""

# ── Searched-empty -> abstention hint, appended to the end of the answer context ─
# _abs abstention questions fall back 70% of the time (the answer is not in the
# corpus, so the agent never reaches FINAL). The signal "searched the whole corpus
# and verified nothing" is itself the strongest evidence for abstention, and should
# not be thrown away as a failure.
ABSTENTION_HINT = (
    "\n\nNOTE: An evidence-search agent has already scanned the FULL conversation "
    "history for this question and could not verify any relevant evidence. If the "
    "context above does not clearly contain the answer, state that the information "
    "is not available in the conversation history — do not guess or invent details."
)
