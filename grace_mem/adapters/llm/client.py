"""The one HTTP client every LLM call in the pipeline goes through.

Two things are centralised here rather than left to callers.

Seeding: reproducibility requires the same seed on every completion, but the
backends this runs against are not consistent about supporting it -- some
reject an unknown `seed` field with 400/422, some raise inside the SDK. Rather
than making each caller decide, requests are sent seeded and retried unseeded
on exactly those failures, and the outcome is logged once per state so a run
that silently lost determinism is visible in the log instead of only in the
diverging results.

Accounting: every method records prompt/completion tokens against
`token_tracker` under its own label, which is what makes per-stage cost
attribution possible after the fact. A new call path that skips the tracker
disappears from those reports.
"""

import logging
import os
import time
from typing import Any

import httpx
import requests
from dotenv import load_dotenv
from openai import OpenAI

from grace_mem.adapters.llm.token_tracking import token_tracker
from grace_mem.domain.extraction import SCHEMA_keyword
from grace_mem.runtime.paths import resolve_project_root
from grace_mem.runtime.reproducibility import get_runtime_reproducibility

ENV_PATH = resolve_project_root() / ".env"
load_dotenv(dotenv_path=ENV_PATH)
LLM_API = os.getenv("LLM_API")
MODEL_NAME = os.getenv("MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
logger = logging.getLogger(__name__)

class _Namespace:
    """Recursively wraps a dict so attribute access mirrors the OpenAI SDK response shape."""
    def __init__(self, d: dict[str, Any]) -> None:
        """Convert nested dict and list values into attribute-accessible objects."""
        for k, v in d.items():
            setattr(self, k, _Namespace(v) if isinstance(v, dict) else
                    [_Namespace(i) if isinstance(i, dict) else i for i in v] if isinstance(v, list) else v)

def _dict_response(data: dict[str, Any]) -> "_Namespace":
    """Wrap a raw response dict in the lightweight namespace adapter."""
    return _Namespace(data)

class LLMClient:
    """Chat-completion client with seed negotiation and token accounting.

    Holds two transports on purpose: an OpenAI SDK client for streaming (it
    handles SSE framing and usage chunks) and plain `requests` for the
    non-streaming calls, where the SDK's response objects would have to be
    unwrapped back into dicts anyway. `_dict_response` bridges the difference so
    callers see the same `resp.choices[0].message.content` shape either way.

    Not thread-safe: the seed-state log set is mutated without a lock. Give
    each worker its own client -- which is also what keeps the token tracker's
    per-stage totals attributable.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout: float = 60.0,
        api_key: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Configure the HTTP clients and helper processors used for LLM calls."""
        base_url   = base_url   or LLM_API
        model_name = model_name or MODEL_NAME
        repro_cfg = get_runtime_reproducibility()
        self.model_name = model_name
        self.seed = int(seed if seed is not None else repro_cfg.seed)
        resolved_api_key = OPENAI_API_KEY if api_key in (None, "") else api_key
        self._base_url = (base_url or "").rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        }
        self._timeout = timeout
        self._seed_log_states: set[str] = set()
        self._closed = False
        self.client = OpenAI(
            base_url=base_url,
            api_key=resolved_api_key,
            http_client=httpx.Client(timeout=timeout),
        )

    def close(self) -> None:
        """Release the underlying HTTP transport once."""
        if self._closed:
            return
        self.client.close()
        self._closed = True

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _log_seed_state(self, state: str, detail: str) -> None:
        """Log a seed-support transition once per state, per client.

        Deduplicated because the alternative is one warning per completion:
        against a backend that ignores seeds, an unfiltered log buries every
        other message in the run.
        """
        if state in self._seed_log_states:
            return
        self._seed_log_states.add(state)
        logger.warning(
            "LLM backend seed %s for model=%s base_url=%s detail=%s",
            state,
            self.model_name,
            self._base_url,
            detail,
        )

    def _seeded_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of the payload carrying this client's seed.

        Copies rather than mutating so the caller keeps an unseeded payload to
        retry with -- `_post` depends on that original still being intact.
        """
        seeded = dict(payload)
        seeded["seed"] = self.seed
        return seeded

    @staticmethod
    def _seed_unsupported_response(resp: requests.Response) -> bool:
        """Report whether a rejection was caused by the `seed` field.

        There is no standard signal for "I do not support this parameter", so
        this matches on the two status codes backends use for schema rejection
        plus the phrasings seen in practice. The test is deliberately narrow:
        treating any 400 as a seed problem would retry unseeded on genuinely
        malformed requests and turn a hard error into a silent loss of
        determinism.
        """
        if resp.status_code not in {400, 422}:
            return False
        text = ""
        try:
            text = resp.text.lower()
        except Exception:
            return False
        return "seed" in text or "extra fields not permitted" in text or "unknown field" in text
    
    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one non-streaming request, retrying unseeded if the seed is rejected.

        The retry sends the caller's original `payload`, not the seeded copy,
        which is why `_seeded_payload` must not mutate in place.

        Raises:
            requests.HTTPError: For any failure that is not a seed rejection,
                and for a retry that fails on its own terms.
        """
        request_payload = self._seeded_payload(payload)
        resp = requests.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json=request_payload,
            timeout=self._timeout,
        )
        if self._seed_unsupported_response(resp):
            self._log_seed_state("unsupported", "retrying request without seed")
            resp = requests.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=self._timeout,
            )
        elif resp.ok:
            self._log_seed_state("accepted", f"seed={self.seed}")
        resp.raise_for_status()
        return resp.json()

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.0, max_tokens: int = 512) -> _Namespace:
        """Send one non-streaming chat completion.

        The general-purpose entry point, used by the judge among others.
        temperature defaults to 0 because its callers are scoring and
        classifying, where run-to-run variation is measurement noise.

        Returns:
            The response wrapped so attribute access matches the OpenAI SDK:
            `resp.choices[0].message.content`.
        """
        t0 = time.perf_counter()
        data = self._post({
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        usage = data.get("usage", {})
        if usage:
            token_tracker.record("chat", usage["prompt_tokens"], usage["completion_tokens"], time.perf_counter() - t0)
        return _dict_response(data)

    def generate_llm_extract(self, prompt: str, max_tokens: int = 3000, temperature: float = 0) -> tuple[str, float]:
        """Run the extraction model and return the raw text plus latency."""
        t0 = time.perf_counter()
        data = self._post({
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        elapsed = time.perf_counter() - t0
        usage = data.get("usage", {})
        if usage:
            token_tracker.record("generate_llm_extract", usage["prompt_tokens"], usage["completion_tokens"], elapsed)
        return data["choices"][0]["message"]["content"], elapsed

    def generate_llm_keyword(self, prompt: str, max_tokens: int | None = None, temperature: float = 0) -> tuple[str, float]:
        """Run keyword extraction and return the raw JSON string plus latency.

        max_tokens defaults to KG_KEYWORD_MAX_TOKENS (fallback 512). Reasoning
        backends (e.g. qwen3.5) spend the whole budget on the thinking channel and
        return an empty `content` unless the budget is raised (~8192 needed).
        """
        if max_tokens is None:
            max_tokens = int(os.getenv("KG_KEYWORD_MAX_TOKENS", "512"))
        t0 = time.perf_counter()
        data = self._post({
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "KeywordExtraction", "schema": SCHEMA_keyword}},
        })
        elapsed = time.perf_counter() - t0
        usage = data.get("usage", {})
        if usage:
            token_tracker.record("generate_llm_keyword", usage["prompt_tokens"], usage["completion_tokens"], elapsed)
        return data["choices"][0]["message"]["content"], elapsed

