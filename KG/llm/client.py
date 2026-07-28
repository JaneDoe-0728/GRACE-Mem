# llm/client.py
import logging
import os, time, json, threading, requests
from pathlib import Path
from typing import Any, Iterator
from dotenv import load_dotenv
from openai import OpenAI
import httpx
from KG.utils.common import SCHEMA_keyword
from KG.services.entity_manager import EntityOpsProcessor, EntityOpsConfig
from experiment.reproducibility import get_runtime_reproducibility

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)
LLM_API = os.getenv("LLM_API")
MODEL_NAME = os.getenv("MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Token tracker                                                                #
# --------------------------------------------------------------------------- #
_TOKEN_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "token_usage.jsonl"

# Human-readable labels for each LLM method
_METHOD_LABELS = {
    "chat":                   "qa_answer",           # QA 回答 / 答案評分
    "generate_llm_extract":   "entity_extraction",   # Entity / relationship 抽取
    "generate_llm_keyword":   "keyword_extraction",  # 檢索關鍵字抽取
    "generate_llm_dynamic_plan": "dynamic_planning",  # 動態檢索規劃
    "generate_llm_hyde":          "hyde_generation",   # HyDE 假設摘要生成
}

class _TokenTracker:
    """Thread-safe token usage tracker with per-call context (dataset + stage).

    Usage:
        token_tracker.set_context(dataset="abc123", stage="ingest")
        # ... LLM calls happen ...
        token_tracker.set_context(dataset="abc123", stage="qa")

    Each log entry in token_usage.jsonl contains:
        ts, dataset, stage, label, prompt_tokens, completion_tokens, total_tokens
    """
    def __init__(self, log_path: Path = _TOKEN_LOG_PATH) -> None:
        """Initialize shared token logging state and per-thread context storage."""
        self._lock = threading.Lock()
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()   # per-thread: dataset, stage
        # totals keyed by (dataset, stage, label)
        self._totals: dict[tuple, dict] = {}

    def set_context(
        self,
        dataset: str,
        stage: str,
        log_dir: "Path | str | None" = None,
        log_path: "Path | str | None" = None,
    ) -> None:
        """Set the current dataset, stage, and optional per-instance log path for this thread.

        ``log_path`` takes precedence over ``log_dir``.  Use ``log_path`` to write
        a per-sample file (e.g. ``tokens_usage42.jsonl``); use ``log_dir`` for a
        shared directory with a fixed ``token_usage.jsonl`` name.
        """
        self._local.dataset = dataset
        self._local.stage = stage
        if log_path is not None:
            p = Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._local.log_path = p
        elif log_dir is not None:
            p = Path(log_dir)
            p.mkdir(parents=True, exist_ok=True)
            self._local.log_path = p / "token_usage.jsonl"
        else:
            self._local.log_path = None  # fall back to global

    def _get_context(self) -> tuple[str, str]:
        """Return the current dataset and stage labels for this thread."""
        return (
            getattr(self._local, "dataset", "unknown"),
            getattr(self._local, "stage", "unknown"),
        )

    def record(self, method: str, prompt_tokens: int, completion_tokens: int, elapsed: float = 0.0) -> None:
        """Append one token-usage record and update in-memory aggregates."""
        dataset, stage = self._get_context()
        label = _METHOD_LABELS.get(method, method)
        total = prompt_tokens + completion_tokens
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset": dataset,
            "stage": stage,
            "label": label,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
            "elapsed_s": round(elapsed, 2),
            "tok_per_s": round(total / elapsed, 1) if elapsed > 0 else None,
        }
        key = (dataset, stage, label)
        # Per-dataset log (if set) + global log
        per_dataset_path: "Path | None" = getattr(self._local, "log_path", None)
        with self._lock:
            for path in filter(None, [per_dataset_path, self._log_path]):
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            if key not in self._totals:
                self._totals[key] = {"prompt": 0, "completion": 0, "calls": 0}
            self._totals[key]["prompt"] += prompt_tokens
            self._totals[key]["completion"] += completion_tokens
            self._totals[key]["calls"] += 1

    def summary(self) -> str:
        """Render the accumulated token-usage totals as a text table."""
        with self._lock:
            lines = ["=== Token Usage Summary ===",
                     f"  {'dataset':<20} {'stage':<10} {'label':<22} {'calls':>5}  {'prompt':>8}  {'completion':>10}  {'total':>8}",
                     "  " + "-" * 85]
            grand_p = grand_c = grand_calls = 0
            for (dataset, stage, label), d in sorted(self._totals.items()):
                lines.append(
                    f"  {dataset:<20} {stage:<10} {label:<22} {d['calls']:>5}  "
                    f"{d['prompt']:>8}  {d['completion']:>10}  {d['prompt']+d['completion']:>8}"
                )
                grand_p += d["prompt"]
                grand_c += d["completion"]
                grand_calls += d["calls"]
            lines.append("  " + "-" * 85)
            lines.append(
                f"  {'GRAND TOTAL':<20} {'':<10} {'':<22} {grand_calls:>5}  "
                f"{grand_p:>8}  {grand_c:>10}  {grand_p+grand_c:>8}"
            )
            return "\n".join(lines)

# Module-level singleton shared by all LLMClient instances
token_tracker = _TokenTracker()

class _Namespace:
    """Recursively wraps a dict so attribute access mirrors the OpenAI SDK response shape."""
    def __init__(self, d: dict[str, Any]) -> None:
        """Convert nested dict and list values into attribute-accessible objects."""
        for k, v in d.items():
            setattr(self, k, _Namespace(v) if isinstance(v, dict) else
                    [_Namespace(i) if isinstance(i, dict) else i for i in v] if isinstance(v, list) else v)

def _DictResponse(data: dict[str, Any]) -> "_Namespace":
    """Wrap a raw response dict in the lightweight namespace adapter."""
    return _Namespace(data)

class LLMClient:
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
        self.client = OpenAI(
            base_url=base_url,
            api_key=resolved_api_key,
            http_client=httpx.Client(timeout=timeout),
        )
        self.entity_ops = EntityOpsProcessor(
            generate_fn=self.generate_llm_extract,
            config=EntityOpsConfig()
        )

    def _log_seed_state(self, state: str, detail: str) -> None:
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
        seeded = dict(payload)
        seeded["seed"] = self.seed
        return seeded

    @staticmethod
    def _seed_unsupported_response(resp: requests.Response) -> bool:
        if resp.status_code not in {400, 422}:
            return False
        text = ""
        try:
            text = resp.text.lower()
        except Exception:
            return False
        return "seed" in text or "extra fields not permitted" in text or "unknown field" in text
    
    def stream_chat(self, messages: list[dict[str, Any]], temperature: float = 0.6, max_tokens: int = 2048) -> Iterator[Any]:
        """Streaming chat — usage is captured from the final chunk."""
        t0 = time.perf_counter()
        payload = dict(
            model=self.model_name,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=max_tokens,
            temperature=temperature,
            seed=self.seed,
        )
        try:
            stream = self.client.chat.completions.create(**payload)
            self._log_seed_state("accepted", f"seed={self.seed}")
        except Exception as exc:
            if "seed" not in str(exc).lower():
                raise
            self._log_seed_state("unsupported", "retrying stream request without seed")
            payload.pop("seed", None)
            stream = self.client.chat.completions.create(**payload)
        for chunk in stream:
            if chunk.usage:
                token_tracker.record(
                    "stream_chat",
                    chunk.usage.prompt_tokens,
                    chunk.usage.completion_tokens,
                    time.perf_counter() - t0,
                )
            yield chunk

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a single non-streaming request and return the parsed JSON body."""
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

    # <<< 非串流 (for llm_judge.py) >>>
    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.0, max_tokens: int = 512) -> _Namespace:
        """Send one non-streaming chat completion and return a namespace-shaped response."""
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
        # return a lightweight namespace so callers can do resp.choices[0].message.content
        return _DictResponse(data)

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

    def generate_llm_dynamic_plan(self, system_prompt: str, user_prompt: str,
                                  max_tokens: int = 512, temperature: float = 0) -> tuple[str, float]:
        """Run dynamic retrieval planning and return guidance text plus latency."""
        t0 = time.perf_counter()
        data = self._post({
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        elapsed = time.perf_counter() - t0
        usage = data.get("usage", {})
        if usage:
            token_tracker.record("generate_llm_dynamic_plan", usage["prompt_tokens"], usage["completion_tokens"], elapsed)
        return data["choices"][0]["message"]["content"], elapsed

    def generate_llm_hyde(self, system_prompt: str, user_prompt: str,
                          max_tokens: int = 512, temperature: float = 0) -> tuple[str, float]:
        """Run HyDE hypothetical-summary generation and return raw text plus latency."""
        t0 = time.perf_counter()
        data = self._post({
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        elapsed = time.perf_counter() - t0
        usage = data.get("usage", {})
        if usage:
            token_tracker.record("generate_llm_hyde", usage["prompt_tokens"], usage["completion_tokens"], elapsed)
        return data["choices"][0]["message"]["content"], elapsed

    def generate_entity_ops(self, new_entities: list[dict], similar_map: dict) -> dict:
        """Entity操作生成（委派給 EntityOpsProcessor）"""
        return self.entity_ops.process_batch(new_entities, similar_map)
    
    # def generate_entity_ops(self, new_entities: list[dict], similar_map: dict) -> dict:
    #     blocks = []
    #     for e in new_entities:
    #         name = e["entity_name"]
    #         type_val = e["entity_type"].value if hasattr(e["entity_type"], "value") else e["entity_type"]
    #         desc = e.get("entity_description", "")
    #         key = (name, type_val)
    #         cands = similar_map.get(key) or []

    #         lines = [f"[INPUT] name={name} | type={type_val} | desc={desc}"]
    #         if cands:
    #             lines.append("Candidates:")
    #             ids = []
    #             for meta, score in cands:
    #                 cid = meta.get("id", "?")
    #                 ids.append(cid)
    #                 source = meta.get("_source", "?")   # NEW
    #                 lines.append(
    #                     f"- id={cid} | name={meta.get('name')} | type={meta.get('type')} | "
    #                     f"score={score:.3f} | from={source} | desc={meta.get('description','')}"   # NEW
    #                 )
    #                 # lines.append(
    #                 #     f"- id={cid} | name={meta.get('name')} | type={meta.get('type')} | "
    #                 #     f"score={score:.3f} | desc={meta.get('description','')}"
    #                 # )
    #             quoted = ",".join([f'"{x}"' for x in ids])
    #             lines.append(f"Valid target_existing_id choices: [{quoted}]")
    #         else:
    #             lines.append("Candidates: (none)")
    #             lines.append("Valid target_existing_id choices: []  # No candidates -> only valid action is ADD")
    #         blocks.append("\n".join(lines))

    #     full_prompt = f"{ENTITY_OPS_RULES_V2}\n\n{ENTITY_OPS_FEW_SHOT}\n=== BATCH ===\n" + "\n\n---\n\n".join(blocks)
    #     print("Entity Ops real data:\n", blocks)

    #     raw_output, _ = self.generate_llm_extract(full_prompt, max_tokens=1600)
    #     parsed = _parse_entity_ops_block(raw_output or "")
    #     return self._validate_entity_ops_output(parsed, new_entities, similar_map)

    # def _validate_entity_ops_output(self, parsed: dict, new_entities: list[dict], similar_map: dict) -> dict:
    #     results = []
    #     for e, r in zip(new_entities, parsed.get("results", [])):
    #         name = e["entity_name"]
    #         type_val = e["entity_type"].value if hasattr(e["entity_type"], "value") else e["entity_type"]
    #         desc = (e.get("entity_description") or "").strip()
    #         action = r.get("action", "").strip().upper()
    #         teid = r.get("target_existing_id")
    #         cname = r.get("canonical_name") or name
    #         ctype = r.get("canonical_type") or type_val
    #         mdesc = (r.get("merged_description") or "").strip()

    #         valid_ids = {meta.get("id") for meta, _ in (similar_map.get((name, type_val)) or [])}
    #         if action not in {"ADD", "UPDATE"}:
    #             action = "ADD"; teid = None
    #         elif action == "UPDATE" and (not teid or teid not in valid_ids):
    #             if len(valid_ids) == 1:
    #                 teid = next(iter(valid_ids))
    #             else:
    #                 action = "ADD"; teid = None
    #         if action == "ADD":
    #             teid = None
    #         if not mdesc:
    #             mdesc = desc or f"{name} ({type_val})"

    #         results.append({
    #             "input_name": name,
    #             "input_type": type_val,
    #             "action": action,
    #             "target_existing_id": teid,
    #             "canonical_name": cname,
    #             "canonical_type": ctype,
    #             "merged_description": mdesc,
    #         })
    #     print("Validated Entity Ops Results:", results)
    #     return {"results": results}
