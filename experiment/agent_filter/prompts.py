"""Prompts for the grep agent (inline mini-harness).

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


def _active_hypothesis_block() -> str:
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

# ── Strict answering prompt, attacking three answering-side failure modes ────
# 1. aggregation questions that never total up or miscount -> force listing every
#    instance before computing
# 2. abstention questions where "use general knowledge" encourages invention ->
#    forbid it; say plainly when the evidence is insufficient
# 3. knowledge_update answers taken from a stale value -> on repeated updates take
#    the newest (the context carries date stamps)
ANSWER_SYSTEM_STRICT = (
    # v7's lesson: a hard abstention clause made gpt-oss-20b refuse too readily (21
    # of 50 downflips answered "not enough" with the evidence right there), so it is
    # softened here: if relevant information is present an answer is required, and
    # abstention is the last resort.
    "You are a concise and accurate assistant. Answer from the Retrieved Context.\n"
    "- Use the context as your source of truth. If it contains partial or related "
    "information, answer from it — do not refuse when relevant evidence is present.\n"
    "- Only when the context contains NOTHING relevant to the question, say the "
    "information provided is not enough; never invent facts about the user from "
    "general knowledge.\n"
    "- For counting / total / how-often questions: first list EVERY relevant item or "
    "mention found in the context (with dates if shown), then compute and state the "
    "final number explicitly.\n"
    "- If the context contains multiple updates of the same fact over time, answer with "
    "the MOST RECENT one (each evidence line shows its date).\n"
    "Answer directly."
)

# ── Sufficiency verifier: an independent audit call that does not share the
# agent's conversation ──────────────────────────────────────────────────────
SUFFICIENCY_SYSTEM = """You are a strict evidence auditor for a QA system.
Given a QUESTION and the EVIDENCE turns selected to answer it, judge whether the
evidence ALONE is enough to answer the question COMPLETELY and precisely.

Pay special attention to:
- counting/aggregation questions: does the evidence contain EVERY instance being
  counted? If the question asks "how many X", missing even one mention makes it
  insufficient.
- temporal questions: are ALL dates/durations needed for the comparison present?
- multi-part questions: is every part covered?

Do NOT judge answer quality — only whether the necessary information is present.

Reply with EXACTLY one line:
SUFFICIENT
or
INSUFFICIENT: <the specific missing information — which entity, date, or how many more instances are likely missing>"""

SUFFICIENCY_USER = """QUESTION: {question}
{date_line}
SELECTED EVIDENCE:
{evidence}

Is this evidence sufficient to answer the question completely? Reply SUFFICIENT or INSUFFICIENT: <missing>."""

# The gap-directed top-up search instruction, fed back into the agent's conversation
GAP_HINT_TEMPLATE = """An independent verifier reviewed your FINAL selection and judged the evidence
INSUFFICIENT: {missing}

Continue searching with GREP to fill exactly this gap. Your previous selections are
kept — you only need to find the MISSING evidence and ADD it. When done, reply:
FINAL <all previous sids> <newly found sids>"""

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

# ── Answer-blind per-item adjudication: an independent call that does not share
# the agent's conversation ─────────────────────────────────────────────────────
# The counter to the agent asking and answering itself: its FINAL is an "answer
# citation" (solve first, then keep only the smallest turn set containing the
# answer span -- of 7 gold entries per question it takes just 2). This
# adjudication call cannot see the agent's search history or the answer it
# reached, and rules on one thing only for each discarded seed: is it topically
# relevant to the question? Solving the question or judging the answer is
# explicitly forbidden, which routes around the answer-span radar that causes the
# preference and multi-hop failures.
ADJUDICATE_SYSTEM = """You are an evidence auditor for a long-term memory QA system.
You are given a QUESTION and candidate evidence turns that an earlier selection
step decided to discard. Audit each candidate INDEPENDENTLY, one by one.

Judge TOPICAL RELEVANCE, not answers:
- KEEP a candidate if it carries information about the question's subject — the
  same person, entity, event, activity, preference, constraint, or time period
  the question is about — even if it does not answer the question by itself.
  Statements where the user describes themselves (what they have, like, use,
  did, plan, or want) are evidence for any question about that topic. Dated
  mentions of the same fact/entity are evidence for counting, latest-value and
  time-span questions.
- DROP a candidate only if it is about an unrelated subject.

Do NOT try to answer the question. Do NOT judge whether a candidate contains
the answer — a turn can be essential evidence without containing any answer.

Output format — one line per candidate, cover ALL candidates, in the given order:
<sid> KEEP|DROP <short reason>
No other text before or after the lines."""

ADJUDICATE_USER = """QUESTION: {question}
{date_line}
Discarded candidates to audit ({n} items):
{candidates}

Audit every candidate — one line each, exactly: <sid> KEEP|DROP <short reason>"""

# ── Planner-worker prompt variant ───────────────────────────────────────
# Experimental two-layer agent loop:
# - PLANNER sees candidate summaries and dispatches focused tasks.
# - WORKER searches raw turns and reports verified evidence claims.
# These prompts are not used by the default harness today, but live here with the
# rest of the agent-filter prompt surface so prompt variants share one module.
PLANNER_SYSTEM = """You are the planner of an evidence-selection team for a long-term
memory QA system. You are given a QUESTION and a list of CANDIDATE evidence turns
(retrieved by vector+rerank; each has a sid). Candidates may contain distractors, and
some truly relevant turns may be MISSING from the list.

You do NOT read raw conversation text yourself. Instead you DISPATCH tasks to a worker
that searches the full corpus and reports back a compact summary for each turn it finds:
  [sid] verified|unverified | one-line claim

You have two commands — reply with EXACTLY ONE as the last line of your message:
  TASK <instruction>   dispatch a search task to the worker. Be specific and topical,
                       e.g. "verify which candidates mention the marathon and its date",
                       or "find any turn where the user states their coffee preference".
                       The worker will GREP/READ/VECTOR the corpus and report claims.
  FINAL <sid> <sid> …  your final selection: the sids that best answer the question.

RULES:
- One command per message. Brief reasoning before the command is fine.
- Start by dispatching a TASK to verify the candidates and hunt for anything missing.
  You may dispatch several TASKs (one per message) to cover different sub-questions
  (e.g. multi-hop: one TASK per hop; counting: one TASK to collect every dated mention).
- Only put a sid in FINAL if the worker reported it as VERIFIED. Never invent sids.
- Keep every VERIFIED sid that supports the answer. For counting / total / how-often /
  latest / current questions, keep EVERY dated mention the worker found (missing one
  breaks the count; keeping only the old value breaks updates).
- You have at most {max_tasks} TASK dispatches. When the worker has covered the
  question, reply FINAL immediately — do not keep dispatching for completeness.
"""

PLANNER_USER = """QUESTION: {question}
{date_line}{hint_line}
{graph_context}CANDIDATE evidence turns (from vector+rerank; may contain distractors, may be incomplete):
{candidates}

Dispatch a TASK to verify candidates and search for missing evidence, then give FINAL sids.
"""

WORKER_REPORT_HEADER = "WORKER REPORT for your task:"

WORKER_SYSTEM = """You are a search worker for a long-term memory QA system. The planner
gave you ONE focused TASK. Use the tools to carry out exactly that task over the full
conversation corpus, then report a compact summary. Do NOT try to answer the overall
question — only carry out your task and report what you find.

TOOLS — reply with EXACTLY ONE command as the last line of your message:
  GREP <regex>          case-insensitive regex over every raw turn. Prefer rare literal
                        anchors: names, dates, numbers. Returns [sid] [date] role: snippet.
                        Date stamps are searchable too (GREP 2023/03 → March 2023 turns).
  READ <sid> [k]        show raw turns around <sid> (default k=2) in its session.
{vector_tool}  REPORT <lines>        finish your task. Report one line per relevant turn you confirmed:
                        [sid] | <one-line claim of what this turn states>
                        Only report sids you actually saw in a GREP/READ result.

RULES:
- One command per message. Brief reasoning before the command is fine.
- Copy sids EXACTLY as shown, including any prefix.
- Search for LITERAL spans from the task (entities, dates, numbers). Never repeat a
  search that returned 0 matches; change keywords instead.
- Verify with READ when unsure. Report a turn only after you have seen its raw text.
- You have at most {max_calls} tool calls; when your task is covered, REPORT immediately.
"""

WORKER_USER = """OVERALL QUESTION (for context only — do NOT answer it): {question}
{date_line}
YOUR TASK: {task}

CANDIDATE turns already on the table (verify/expand as your task requires):
{candidates}

Carry out your task with GREP/READ{vector_hint}, then REPORT one line per confirmed turn.
"""
