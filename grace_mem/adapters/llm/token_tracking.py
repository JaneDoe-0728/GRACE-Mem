"""Thread-safe token usage tracking shared by LLM-related services."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from grace_mem.runtime.paths import resolve_project_root

# Anchored on the checkout root, not on this file's depth: `parents[2]` was the
# repo root while this lived at KG/llm/client.py and became the package dir in
# the move, which silently relocated every run's token log into grace_mem/logs/.
_TOKEN_LOG_PATH = resolve_project_root() / "logs" / "token_usage.jsonl"

_METHOD_LABELS = {
    "chat": "qa_answer",
    "generate_llm_extract": "entity_extraction",
    "generate_llm_keyword": "keyword_extraction",
}


class TokenTracker:
    """Record per-thread LLM usage and aggregate it by dataset and stage."""

    def __init__(self, log_path: Path = _TOKEN_LOG_PATH) -> None:
        self._lock = threading.Lock()
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._totals: dict[tuple, dict] = {}

    def set_context(
        self,
        dataset: str,
        stage: str,
        log_dir: Path | str | None = None,
        log_path: Path | str | None = None,
    ) -> None:
        """Set labels and an optional per-dataset output path for this thread."""
        self._local.dataset = dataset
        self._local.stage = stage
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._local.log_path = path
        elif log_dir is not None:
            directory = Path(log_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self._local.log_path = directory / "token_usage.jsonl"
        else:
            self._local.log_path = None

    def _get_context(self) -> tuple[str, str]:
        return (
            getattr(self._local, "dataset", "unknown"),
            getattr(self._local, "stage", "unknown"),
        )

    def record(
        self,
        method: str,
        prompt_tokens: int,
        completion_tokens: int,
        elapsed: float = 0.0,
    ) -> None:
        """Append one usage record and update in-memory totals."""
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
        per_dataset_path: Path | None = getattr(self._local, "log_path", None)
        with self._lock:
            for path in filter(None, [per_dataset_path, self._log_path]):
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")
            if key not in self._totals:
                self._totals[key] = {"prompt": 0, "completion": 0, "calls": 0}
            self._totals[key]["prompt"] += prompt_tokens
            self._totals[key]["completion"] += completion_tokens
            self._totals[key]["calls"] += 1

    def summary(self) -> str:
        """Render accumulated token totals as a text table."""
        with self._lock:
            lines = [
                "=== Token Usage Summary ===",
                (f"  {'dataset':<20} {'stage':<10} {'label':<22} {'calls':>5}  "
                f"{'prompt':>8}  {'completion':>10}  {'total':>8}"),
                "  " + "-" * 85,
            ]
            grand_prompt = grand_completion = grand_calls = 0
            for (dataset, stage, label), totals in sorted(self._totals.items()):
                lines.append(
                    f"  {dataset:<20} {stage:<10} {label:<22} {totals['calls']:>5}  "
                    f"{totals['prompt']:>8}  {totals['completion']:>10}  "
                    f"{totals['prompt'] + totals['completion']:>8}"
                )
                grand_prompt += totals["prompt"]
                grand_completion += totals["completion"]
                grand_calls += totals["calls"]
            lines.append("  " + "-" * 85)
            lines.append(
                f"  {'GRAND TOTAL':<20} {'':<10} {'':<22} {grand_calls:>5}  "
                f"{grand_prompt:>8}  {grand_completion:>10}  "
                f"{grand_prompt + grand_completion:>8}"
            )
            return "\n".join(lines)


token_tracker = TokenTracker()
