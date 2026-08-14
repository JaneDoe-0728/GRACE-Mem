"""Judge prompt for open-domain questions, where gold is a reference not an oracle.

Separate from `judge.py` because the grading rule genuinely differs. These
questions admit several correct answers, so the gold string is one acceptable
answer rather than the only one, and a judge applying the standard prompt marks
correct answers wrong for not matching it.
"""

SYSTEM_PROMPT = "You are an expert grader for evidence-grounded conversational QA."

ACCURACY_PROMPT = """\
Label the generated answer as CORRECT or WRONG.

You will be given:
(1) Question
(2) Gold answer (a reference; not always the only valid answer)
(3) Generated answer
(4) Evidence turns (may be incomplete excerpts)

Important grading approach:
- Focus on the CORE CLAIM of the generated answer (the minimal direct answer).
- Ignore extra elaboration unless it directly changes the core claim.

Step 1: Identify the core claim of the generated answer (mentally; do not output it).
Step 2: Decide CORRECT/WRONG using these rules:

CORRECT if:
A) The core claim answers the question, AND
B) The core claim is supported by the evidence OR is a reasonable inference from the evidence
   (even if evidence is incomplete, allow cautious/general answers), AND
C) The core claim does NOT clearly contradict the gold answer.
   - Only treat it as a contradiction if it is mutually exclusive (e.g., opposite yes/no, different specific entity/value).

WRONG if:
- The core claim does not answer the question, OR
- The core claim contradicts the evidence, OR
- The core claim clearly contradicts the gold answer (mutually exclusive), OR
- The core claim is too vague to satisfy the question when specificity is required.

Do NOT mark WRONG just because:
- The generated answer adds unsupported minor details, as long as the core claim remains supported and consistent.

Now grade:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {response}

Evidence turns:
{evidence_turns}

Return ONLY JSON:
{{"label":"CORRECT"}} or {{"label":"WRONG"}}.
"""
