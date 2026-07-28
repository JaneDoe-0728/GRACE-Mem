SYSTEM_PROMPT = (
    "You are an expert grader that determines if answers to questions match a gold standard answer."
)

ACCURACY_PROMPT = """\
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {response}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

SYSTEM_PROMPT_PLUS = "You are an expert evaluation judge. Follow the requested rubric exactly."

PROMPT_TEMPLATES = {
    "multi-hop": """
You are a Fact-Checking Judge.
Your task: Compare the model's prediction with the reference answer (multi-hop fact QA).

Labels:
- "correct": The answer matches the reference entities (names, places, times) exactly.
- "partial": The answer misses some details or contains minor inaccuracies but gets the main entity right.
- "wrong": The answer is factually incorrect or hallucinates details not in the reference.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

Return your judgment strictly in JSON format:
{{"label": "correct"|"partial"|"wrong", "reason": "<short explanation>"}}
""",
    "single-hop": """
You are a Fact-Checking Judge.
Your task: Compare the model's prediction with the reference answer (single-hop fact QA).

Labels:
- "correct": The answer matches the reference entities exactly.
- "partial": The answer misses some details but gets the main entity right.
- "wrong": The answer is factually incorrect or hallucinates details not in the reference.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

Return your judgment strictly in JSON format:
{{"label": "correct"|"partial"|"wrong", "reason": "<short explanation>"}}
""",
    "temporal": """
You are a Temporal Logic Judge.
Your task: Check the calculation, duration, or sequence of events.

Labels:
- "correct": The calculated time, duration, or date matches the reference exactly (semantic equivalents are allowed).
- "wrong": The calculation is incorrect, the sequence is reversed, or the specific time is wrong.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

Return your judgment strictly in JSON format:
{{"label": "correct"|"wrong", "reason": "<short explanation>"}}
""",
    "common-sense": """
You are a Knowledge Logic Judge.
Your task: Assess if the prediction applies correct commonsense/world knowledge consistent with the reference.

Labels:
- "correct": The logic and inference are sound and match the reference conclusion.
- "partial": The reasoning is mostly correct but the final conclusion is vague or slightly off.
- "wrong": The reasoning contradicts commonsense or the reference.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

Return your judgment strictly in JSON format:
{{"label": "correct"|"partial"|"wrong", "reason": "<short explanation>"}}
""",
    "adversarial": """
You are a Skeptical Judge evaluating robustness.
The question is inherently misleading (e.g., asks about something not in the conversation).
Your task: Judge whether the model's answer conveys that "this was not mentioned in the conversation" (or equivalent refusal).

Labels:
- "correct": The prediction clearly conveys that the information was not mentioned / cannot be answered from the conversation. Score it.
- "wrong": The prediction does NOT convey that meaning—e.g., it gives a concrete answer or does not refuse. Do not score.

Model Prediction:
{pred}

Return your judgment strictly in JSON format:
{{"label": "correct"|"wrong", "reason": "<short explanation>"}}
""",
    "Cognitive": """
You are a Memory Awareness Judge.
Your task: Judge whether the Model Prediction considers or is linked to the Evidence. If there is a clear connection, the answer is correct (score 1); if not, it is wrong (no score).

Labels:
- "correct": The prediction explicitly or implicitly reflects/uses the evidence (memory or constraint). Give 1 point.
- "wrong": The prediction does not show such a link to the evidence. No point.

Memory/Evidence:
{evidence}

Model Prediction:
{pred}

Return your judgment strictly in JSON format:
{{"label": "correct"|"wrong", "reason": "<Does the prediction relate to the evidence?>"}}
""",
    "default": """
You are an expert evaluator.
Your task: Compare the prediction with the reference.

Labels:
- "correct": Factually consistent with the reference.
- "partial": Contains correct info but is incomplete.
- "wrong": Factually incorrect.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

Return your judgment strictly in JSON format:
{{"label": "correct"|"partial"|"wrong", "reason": "<short explanation>"}}
""",
}
