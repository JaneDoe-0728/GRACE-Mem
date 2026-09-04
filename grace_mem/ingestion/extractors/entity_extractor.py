"""Asking the LLM for the entities in one turn.

A thin retry wrapper around the extraction call. The lock is passed in rather
than owned: the caller runs turns concurrently and decides how much of the LLM
it will use at once, which is not a decision an extractor can make for itself.

Context-length failures are not retried -- a prompt that was too long will be
too long again -- and are surfaced rather than swallowed.
"""

import threading
from typing import Any

from grace_mem.ingestion.parsing import (
    is_context_length_exceeded_error,
    parse_entities_only,
)
from grace_mem.utils.logger_config import make_module_jlog

_jlog = make_module_jlog(name="grace_mem.Ingestor", filename="kg_ingestor.jsonl")


class EntityExtractor:
    """Wraps extract_entities_only with retry logic. Receives _lock from caller."""

    def __init__(self, *, llm: Any, lock: threading.Lock, cfg: Any) -> None:
        """Store the shared LLM, lock, and extraction configuration."""
        self._llm = llm
        self._lock = lock
        self._cfg = cfg

    def extract(
        self,
        prompt_vars: dict,
        prompt_template: str,
        request_id: str,
        *,
        tuple_delim: str | None = None,
        record_delim: str | None = None,
        completion_delim: str | None = None,
        max_retries: int = 2,
    ) -> tuple[bool, Any]:
        """Extract entities from summary text. Returns (success, entities_list or error_msg)."""
        tuple_delimiter_val = tuple_delim or prompt_vars.get("tuple_delimiter", self._cfg.llm_tuple_delim)
        record_delimiter_val = record_delim or prompt_vars.get("record_delimiter", self._cfg.llm_record_delim)
        completion_delimiter_val = completion_delim or prompt_vars.get("completion_delimiter", self._cfg.llm_completion_delim)

        with self._lock:
            prompt = prompt_template.format(**prompt_vars)
            _jlog("entity_prompt_format_done", request_id)

            for attempt in range(max_retries + 1):
                try:
                    llm_output, latency_seconds = self._llm.generate_llm_extract(prompt)
                    print(f"=== RAW Entity Extraction (attempt {attempt + 1}) ===\n{llm_output}")
                    _jlog("llm_entity_extract_done", request_id, latency_sec=latency_seconds, attempt=attempt + 1)

                    entities = parse_entities_only(
                        llm_output, tuple_delimiter_val, record_delimiter_val, completion_delimiter_val
                    )
                    entity_count = len(entities)
                    print(f"Parsed Entities count: {entity_count}")
                    _jlog("parse_entity_extraction_done", request_id, entity_count=entity_count, attempt=attempt + 1)

                    if entity_count == 0 and attempt < max_retries:
                        print(f"0 entities extracted. Retrying... ({attempt + 1}/{max_retries})")
                        continue

                    print(f"✓ Entity Extraction: {entity_count} entities")
                    return (True, entities)

                except Exception as parse_error:
                    if is_context_length_exceeded_error(parse_error):
                        _jlog(
                            "context_length_limit_exceeded",
                            request_id,
                            stage="entity_extraction",
                            attempt=attempt + 1,
                            prompt_length=len(prompt),
                            error=str(parse_error),
                        )
                    _jlog("parse_entity_extraction_failed", request_id, error=str(parse_error), attempt=attempt + 1)
                    if attempt < max_retries:
                        print(f"Parse error: {parse_error}. Retrying...")
                        continue
                    else:
                        print(f"Entity parse failed: {parse_error}")
                        return (False, f"validation_error: {parse_error}")

        raise RuntimeError("Entity extraction retry loop ended without a result")
