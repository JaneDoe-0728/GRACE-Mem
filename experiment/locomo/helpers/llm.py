"""LLM access for the LoCoMo evaluation and judging stages.

Separate from `grace_mem.llm.client` on purpose. That client serves the system
under test; this one serves the evaluator, and mixing them would put judge
tokens into the pipeline's own cost accounting and make the two share retry and
seeding behaviour that should be tunable independently.

The same seed negotiation appears here as in the pipeline client -- send
seeded, retry unseeded if the backend rejects it, log the transition once --
because a judge that silently stopped being deterministic would move scores
between runs for reasons unrelated to the change under test.

The `build_*_messages` builders keep the standard and open-domain grading
rubrics explicit at their call sites.
"""

import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

import experiment.locomo.prompts.judge as judge_prompts
import experiment.locomo.prompts.open_domain as open_domain_prompts
from experiment.common.reproducibility import get_runtime_reproducibility

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

_SEEDED_BACKENDS: set[tuple[str, str, str]] = set()


def _log_seed_backend_state(base_url: str, model: str, state: str, detail: str) -> None:
    """Log a seed-support transition once per (endpoint, model, state).

    Deduplicated because the alternative is one warning per judged question:
    against a backend that ignores seeds, an unfiltered log buries everything
    else in the run.
    """
    key = (base_url, model, state)
    if key in _SEEDED_BACKENDS:
        return
    _SEEDED_BACKENDS.add(key)
    print(f"[locomo.llm] seed backend={state} model={model} base_url={base_url} detail={detail}")


def _chat_completion(
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    response_format: dict | None = None,
    timeout: int = 120,
) -> dict:
    # Judge calls use a dedicated endpoint when JUDGE_LLM_API is set.
    """Post one chat completion to the judge endpoint, retrying unseeded if rejected.

    JUDGE_LLM_API and JUDGE_MODEL_NAME take precedence over the pipeline's own
    LLM_API and MODEL_NAME. That separation is the point: judging the system
    with the same model that generated the answers lets a model's preference for
    its own phrasing show up as accuracy.

    The seed dance mirrors `grace_mem.llm.client` -- send seeded, retry once
    unseeded on a 400/422 mentioning "seed", and log the transition once so a run
    that quietly lost determinism is visible in the log rather than only in
    diverging scores.

    Raises:
        requests.HTTPError: On any non-seed failure.
        RuntimeError: If the backend returns a non-object payload.
    """
    base_url = (os.getenv("JUDGE_LLM_API") or os.getenv("LLM_API") or "").rstrip("/")
    model_name = os.getenv("JUDGE_MODEL_NAME") or os.getenv("MODEL_NAME", "")
    seed = get_runtime_reproducibility().seed
    payload: dict = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        **({"response_format": response_format} if response_format else {}),
    }
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    if resp.status_code in {400, 422} and "seed" in (resp.text or "").lower():
        _log_seed_backend_state(base_url, model_name, "unsupported", "retrying without seed")
        payload.pop("seed", None)
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
    elif resp.ok:
        _log_seed_backend_state(base_url, model_name, "accepted", f"seed={seed}")
    resp.raise_for_status()
    result = resp.json()
    if not isinstance(result, dict):
        raise RuntimeError("LLM API returned a non-object payload")
    return result


def _extract_completion_text(payload: dict) -> tuple[str, dict]:
    """Pull the reply text and usage block out of a completion payload.

    Returns:
        (text, usage). Usage is {} when the backend omitted it, so callers can
        index it without guarding.
    """
    choices = payload.get("choices") or []
    choice0 = choices[0] if choices else {}
    message = choice0.get("message") or {}
    content = message.get("content") or ""
    meta = {
        "finish_reason": choice0.get("finish_reason"),
        "content_len": len(content),
        "has_content": bool(content.strip()),
    }
    return content, meta


def llm_post(
    messages: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    retries: int = 3,
    retry_sleep_sec: float = 1.0,
    return_meta: bool = False,
) -> str | tuple[str, dict]:
    """Send a chat completion with retries, returning just the reply text.

    The workhorse for judging and answer generation. Retries because judging a
    full run makes thousands of calls and a transient failure would otherwise
    leave a hole in the results that reads as a wrong answer.
    """
    last_content = ""
    last_meta: dict = {}
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            payload = _chat_completion(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120,
            )
            content, meta = _extract_completion_text(payload)
            last_content = content
            last_meta = meta
            if content.strip():
                if return_meta:
                    return content, meta
                return content
            print(
                f"[llm_post] empty completion attempt={attempt}/{retries} "
                f"finish_reason={meta.get('finish_reason')!r} content_len={meta.get('content_len')}",
                file=sys.stderr,
            )
        except Exception as exc:
            last_error = exc
            print(
                f"[llm_post] attempt={attempt}/{retries} failed: {exc!r}",
                file=sys.stderr,
            )
        if attempt < max(1, retries):
            time.sleep(retry_sleep_sec * attempt)
    if last_error is not None and not last_content:
        # Preserve the old behavior of returning a string; callers decide whether to fall back.
        if return_meta:
            last_meta = {**last_meta, "error": repr(last_error)}
            return last_content, last_meta
        return last_content
    if return_meta:
        return last_content, last_meta
    return last_content


def build_messages(*, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def build_judge_standard_messages(*, question: str, gold: str, gen: str) -> list[dict[str, str]]:
    """Build the judge prompt for standard LoCoMo questions."""
    return build_messages(
        system_prompt=judge_prompts.SYSTEM_PROMPT,
        user_prompt=judge_prompts.ACCURACY_PROMPT.format(
            question=question,
            gold_answer=gold,
            response=gen,
        ),
    )

def build_open_domain_standard_messages(
    *,
    question: str,
    gold: str,
    gen: str,
    evidence_turns: str,
) -> list[dict[str, str]]:
    """Build the open-domain judge prompt, where gold is one acceptable answer.

    Uses the open-domain rubric rather than the standard one: these questions
    admit several correct answers, and the standard prompt marks correct answers
    wrong for not matching the reference.
    """
    return build_messages(
        system_prompt=open_domain_prompts.SYSTEM_PROMPT,
        user_prompt=open_domain_prompts.ACCURACY_PROMPT.format(
            question=question,
            gold_answer=gold,
            response=gen,
            evidence_turns=evidence_turns,
        ),
    )
