"""LongMem judge prompts -- aligned word for word with vectorize-io/hindsight's
judge_answer().

Each category uses a different scoring rubric, named with hyphens exactly as in
hindsight; longmem's underscored category names (single_session_user) are
converted to hyphens inside build_messages.
The judge LLM replies as JSON {reasoning, correct}, which stages/judge.py enforces
via response_format and parses tolerantly.
"""
from __future__ import annotations


def _to_hyphen(category: str | None) -> str:
    """longmem's underscored category -> hindsight's hyphenated one
    (single_session_user -> single-session-user)."""
    if not category:
        return ""
    return str(category).strip().lower().replace("_", "-")


# ── Detecting abstention (_abs) questions, and their dedicated rubric ────────
# LongMemEval's abstention questions plant distractors, and the gold says the
# information was never mentioned. The general rubric asks only "does the response
# contain the correct answer", which misjudges a clean abstention ("no record /
# insufficient information") as false because it has no exemption clause. These
# questions therefore need their own scoring logic.
# The decision boundary, calibrated with three 4o-mini votes plus a human tiebreak:
#   - saying only "no record / insufficient information / never mentioned"
#     -> correct=true (a successful abstention)
#   - "count is 0 / none" where that is the correct inference from the event never
#     having happened -> correct=true
#   - giving any concrete number, date, duration or ordering before abstaining
#     (led astray by a distractor) -> correct=false
_ABS_GOLD_MARKERS = (
    "the information provided is not enough",
    "you did not mention this information",
    "did not mention",
    "not mention this",
)


def is_abstention_gold(gold: str | None) -> bool:
    """Whether the gold is an abstention (the information asked for was never
    mentioned)."""
    if not gold:
        return False
    g = str(gold).strip().lower()
    if g in ("0", "0.", "none"):
        return True
    return any(m in g for m in _ABS_GOLD_MARKERS)


ABS_JUDGE_PROMPT = """You are grading an ABSTENTION question. The question asks about something the user NEVER actually told the assistant — the gold answer confirms the information was never provided (e.g. "The information provided is not enough", "You did not mention this information", or the count is 0 because the event never happened).

For these questions the ONLY correct behaviour is to ABSTAIN. Grade the model response:

Set correct=true if the response abstains — it states it has no record / no information / not enough information, says it cannot answer, or points out that the asked-about item (person, place, object, event) was never mentioned. Answering "0" or "none" is also correct when the gold answer is that the thing never happened.

A response ALSO stays correct if, while clearly abstaining on the asked-about item, it truthfully mentions a DIFFERENT but similar item the user actually did discuss — as long as it explicitly labels that item as the different one and does NOT present its value as the answer. Examples (all correct):
- "I have no record of you practicing violin — you only mentioned guitar, about 30 min/day." ("30 min/day" is honest contrast about GUITAR, not the violin answer.)
- "You haven't started a vintage-film collection — you've only been collecting vintage cameras, for about three months." ("three months" attaches to CAMERAS, the named distractor; films are abstained on.)
In every such case the specific value belongs to the distractor, which the response openly names as NOT the thing asked about. The presence of a number/date/duration is NOT by itself disqualifying — only a value offered AS the answer to the asked-about item is.

A response is ALSO correct when it cleanly abstains and merely OFFERS to compute or look something up if the user provides the missing input. Saying "I can't give a figure; if you tell me your current page I'll calculate the pages left" is a correct abstention — offering conditional help is NOT the same as claiming to know the answer. Do not mark such responses false for "implying it could answer."

Set correct=false if the response fabricates a definite answer anyway, OR if it lets a similar-but-different item (a distractor) supply THE answer to the question. Concretely, set correct=false when: (a) it gives a specific number, date, duration, name, or ordering for the ASKED-ABOUT item as though it were known; or (b) it answers the question using the distractor's value without flagging the substitution — e.g. asked about the Porsche but replying "the Ferrari started first on May 2" as the answer; asked "how long in Shinjuku" and replying "seven months" (a value silently taken from Harajuku); or asked how many footballs and replying that the ~15 autographed BASEBALLs "would be the number accumulated" (borrowing the baseball count as the football answer). The distinguishing test: is the concrete value presented AS the answer to what was asked, or as a hypothetical it endorses (→ false), or is it clearly attributed to a different, named item while the asked-about item is abstained on (→ true)?

Do NOT require the response to restate the gold wording. Judge only: did it cleanly abstain on the asked-about item (contrasting distractors by name is fine), or did it produce / borrow a concrete answer for the thing asked (→ false)?

Question: {question}
Gold answer: {gold}
Model response: {generated}

Provide your evaluation as JSON with:
- reasoning: one short sentence
- correct: true or false"""


def build_messages(
    *, question: str, gold: str, generated: str, category: str | None = None,
    is_abstention: bool | None = None,
) -> list[dict[str, str]]:
    """A word-for-word copy of the judge prompt construction in hindsight's
    benchmarks/common/benchmark_runner.py.

    One exception: abstention (_abs) questions go through ABS_JUDGE_PROMPT instead,
    because the general rubric's missing abstention exemption kills clean
    abstentions systematically. Whether a question is an abstention is taken
    **first** from the `is_abstention` the caller passes explicitly (sourced from
    the dataset's `_abs` filename tag, which is authoritative); only when that is
    absent does it fall back to detecting it from the gold text. This branch fires
    on abstention questions alone -- the prompt for every other question is
    untouched.
    """
    abstention = is_abstention if is_abstention is not None else is_abstention_gold(gold)
    if abstention:
        return [{"role": "user", "content": ABS_JUDGE_PROMPT.format(
            question=question, gold=gold, generated=generated)}]

    correct_answer = gold
    predicted_answer = generated
    cat = _to_hyphen(category)

    # LongMemEval-specific evaluation prompts
    if cat in ["single-session-user", "single-session-assistant", "multi-session"]:
        prompt_content = f"""Evaluate if the model response contains the correct answer to the question.

I will give you a question, a correct answer, and a response from a model.
Please set correct=true if the response contains the correct answer. Otherwise, set correct=no.
If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also set correct=true.
If the response only contains a subset of the information required by the answer, set correct=false

Question: {question}

Correct Answer: {correct_answer}

Model Response: {predicted_answer}

Evaluation criteria:
- Set correct=true if the response contains the correct answer
- Set correct=true if the response is equivalent to the correct answer or contains intermediate steps
- Set correct=false if the response is incorrect or missing key information

Provide your evaluation as JSON with:
- reasoning: One sentence explanation
- correct: true or false"""

    elif cat == "temporal-reasoning":
        prompt_content = """
I will give you a question, a correct answer, and a response from a model.
Please set correct=true if the response contains the correct answer. Otherwise, set correct=false.
If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also set correct=true.
If the response only contains a subset of the information required by the answer, answer correct=false.
In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.
"""

    elif cat == "knowledge-update":
        prompt_content = """
I will give you a question, a correct answer, and a response from a model.
Please set correct=true if the response contains the correct answer. Otherwise, set correct=false.
If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.
"""

    elif cat == "single-session-preference":
        prompt_content = """
I will give you a question, a answer for desired personalized response, and a response from a model.
Please set correct=true if the response satisfies the desired response. Otherwise, set correct=false.
The model does not need to reflect all the points in the desired response. The response is correct as long as it recalls and utilizes the user's personal information correctly.
"""

    else:
        # Default LoComo-style evaluation
        prompt_content = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
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
    There's an edge case where the actual answer can't be found in the data and in that case the gold answer will say so (e.g. 'You did not mention this information.'); if the generated answer says that it cannot be answered or it doesn't know all the details, it should be counted as CORRECT.
"""

    user_content = f"""{prompt_content}


Question: {question}
Gold answer: {correct_answer}
Generated answer: {predicted_answer}
First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred.
If it's correct, set correct=true.
"""
    return [{"role": "user", "content": user_content}]


# Backwards compatibility: the SYSTEM_PROMPT name is kept, pointing at the default
# rubric's content
SYSTEM_PROMPT = "LongMemEval / hindsight category-aware judge"
