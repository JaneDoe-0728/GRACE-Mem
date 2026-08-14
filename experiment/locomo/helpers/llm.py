"""LLM access for the LoCoMo evaluation and judging stages.

Separate from `grace_mem.llm.client` on purpose. That client serves the system
under test; this one serves the evaluator, and mixing them would put judge
tokens into the pipeline's own cost accounting and make the two share retry and
seeding behaviour that should be tunable independently.

The same seed negotiation appears here as in the pipeline client -- send
seeded, retry unseeded if the backend rejects it, log the transition once --
because a judge that silently stopped being deterministic would move scores
between runs for reasons unrelated to the change under test.

The `build_*_messages` builders exist per (task, dataset variant) pair because
locomo-plus supplies extra context that changes the prompt's shape, not just
its content.
"""

import json
import os
import sys
import time
from pathlib import Path

import experiment.locomo.prompts.judge as judge_prompts
import experiment.locomo.prompts.open_domain as open_domain_prompts
import requests
from dotenv import load_dotenv

from experiment.common.reproducibility import get_runtime_reproducibility

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

_SEEDED_BACKENDS: set[tuple[str, str, str]] = set()


def _log_seed_backend_state(base_url: str, model: str, state: str, detail: str) -> None:
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


def llm_post_json(messages: list[dict], *, temperature: float = 0.1, max_tokens: int = 2048, retries: int = 3) -> dict:
    """Like llm_post but enforces JSON object output via response_format."""
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            payload = _chat_completion(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                timeout=180,
            )
            content, meta = _extract_completion_text(payload)
            if not content.strip():
                raise RuntimeError(
                    f"LLM returned empty JSON content (finish_reason={meta.get('finish_reason')!r})"
                )
            return json.loads(content)
        except Exception as exc:
            last_error = exc
            print(f"[llm_post_json] attempt={attempt}/{retries} failed: {exc!r}", file=sys.stderr)
            if attempt < max(1, retries):
                time.sleep(1.0 * attempt)
    raise RuntimeError("llm_post_json failed after retries") from last_error


def normalize_prompt_category(label: str, category: str | None) -> str:
    """Map a question's category onto the prompt template that should grade it.

    Categories have been labelled differently across dataset revisions, and an
    unrecognised label must not silently select the wrong rubric -- a temporal
    question graded by the open-domain prompt is scored on the wrong criterion.
    """
    normalized = str(category or "").strip()
    if normalized.lower() == "common-sense":
        return "common-sense"
    label_map = {
        "Multi-hop": "multi-hop",
        "Single-hop": "single-hop",
        "Temporal": "temporal",
        "Adversarial": "adversarial",
        "Cognitive": "Cognitive",
        "Open-domain": "default",
        "Unknown": "default",
    }
    return label_map.get(
        label,
        normalized if normalized in judge_prompts.PROMPT_TEMPLATES else "default",
    )


def build_messages(*, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_plus_messages(
    *,
    label: str,
    category: str | None,
    gold: str,
    pred: str,
    evidence: str,
) -> list[dict[str, str]]:
    template_key = normalize_prompt_category(label, category)
    template = judge_prompts.PROMPT_TEMPLATES.get(template_key, judge_prompts.PROMPT_TEMPLATES["default"])
    return build_messages(
        system_prompt=judge_prompts.SYSTEM_PROMPT_PLUS,
        user_prompt=template.format(gold=gold, pred=pred, evidence=evidence),
    )


def build_judge_standard_messages(*, question: str, gold: str, gen: str) -> list[dict[str, str]]:
    return build_messages(
        system_prompt=judge_prompts.SYSTEM_PROMPT,
        user_prompt=judge_prompts.ACCURACY_PROMPT.format(
            question=question,
            gold_answer=gold,
            response=gen,
        ),
    )


def build_judge_plus_messages(
    *,
    label: str,
    category: str | None,
    gold: str,
    pred: str,
    evidence: str,
) -> list[dict[str, str]]:
    return build_plus_messages(
        label=label,
        category=category,
        gold=gold,
        pred=pred,
        evidence=evidence,
    )


def build_open_domain_standard_messages(
    *,
    question: str,
    gold: str,
    gen: str,
    evidence_turns: str,
) -> list[dict[str, str]]:
    return build_messages(
        system_prompt=open_domain_prompts.SYSTEM_PROMPT,
        user_prompt=open_domain_prompts.ACCURACY_PROMPT.format(
            question=question,
            gold_answer=gold,
            response=gen,
            evidence_turns=evidence_turns,
        ),
    )


def build_open_domain_plus_messages(
    *,
    label: str,
    category: str | None,
    gold: str,
    pred: str,
    evidence: str,
) -> list[dict[str, str]]:
    return build_plus_messages(
        label=label,
        category=category,
        gold=gold,
        pred=pred,
        evidence=evidence,
    )
