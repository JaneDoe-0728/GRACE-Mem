"""Asking the LLM for the relationships in one turn.

The same shape as the entity extractor and deliberately separate: the two run
as distinct calls with distinct prompts, and a change to how relationships are
asked for should not touch the file that asks for entities.

Endpoints come back as entity *names*, not ids -- ids do not exist until the
sync step resolves them.
"""

import threading
import time
from typing import Any

from grace_mem.ingestion.parsing import (
    is_context_length_exceeded_error,
    parse_relationships_only,
)
from grace_mem.utils.logger_config import make_module_jlog

_jlog = make_module_jlog(name="grace_mem.Ingestor", filename="kg_ingestor.jsonl")

# Seconds to wait before retrying a *raised* extraction failure, doubling each
# attempt. Zero for the empty-result retry, which is the model being unhelpful
# rather than the transport being unavailable.
#
# The wait exists because the cost of giving up rose: a relationship failure used
# to log and let the turn write its entities with relationships=[], and since
# 04cba26 it fails the whole turn -- which under _require_successful_ingest ends a
# LongMem dataset or a LoCoMo sample. Three back-to-back calls inside the same
# few milliseconds all hit the same rate limit or the same dropped connection, so
# the retries were not buying what they looked like they were buying.
_RETRY_BACKOFF_SEC = 2.0


class RelationshipExtractor:
    """Wraps extract_relationships_only with retry logic. Receives _lock from caller."""

    def __init__(self, *, llm: Any, lock: threading.Lock, cfg: Any) -> None:
        """Store the shared LLM, lock, and relationship-extraction configuration."""
        self._llm = llm
        self._lock = lock
        self._cfg = cfg

    def extract(
        self,
        prompt_vars: dict,
        prompt_template: str,
        extracted_entities: list[Any],
        request_id: str,
        *,
        tuple_delim: str | None = None,
        record_delim: str | None = None,
        completion_delim: str | None = None,
        max_retries: int = 2,
        max_error_retries: int = 4,
    ) -> tuple[bool, Any]:
        """Extract relationships using already-extracted entities. Returns (success, rels or error).

        Two retry budgets, because the two failures are not the same failure.
        ``max_retries`` covers a parse that succeeded and found nothing -- the model
        being unhelpful, worth one or two more asks and no more. ``max_error_retries``
        covers a call that raised, which is usually the transport (rate limit, dropped
        connection, a truncated body); those get more attempts and a growing wait,
        since returning False here now costs the caller its whole turn.
        """
        tuple_delimiter_val = tuple_delim or prompt_vars.get("tuple_delimiter", self._cfg.llm_tuple_delim)
        record_delimiter_val = record_delim or prompt_vars.get("record_delimiter", self._cfg.llm_record_delim)
        completion_delimiter_val = completion_delim or prompt_vars.get("completion_delimiter", self._cfg.llm_completion_delim)

        entities_text_lines = [
            f"- {e.entity_name} ({e.entity_type}): {e.entity_description}"
            for e in extracted_entities
        ]
        entities_text = "\n".join(entities_text_lines) if entities_text_lines else "No entities extracted."
        prompt_vars_with_entities = {**prompt_vars, "entities_text": entities_text}
        valid_entity_names = {e.entity_name for e in extracted_entities}

        with self._lock:
            prompt = prompt_template.format(**prompt_vars_with_entities)
            _jlog("relationship_prompt_format_done", request_id, entity_count=len(extracted_entities))

            for attempt in range(max(max_retries, max_error_retries) + 1):
                try:
                    llm_output, latency_seconds = self._llm.generate_llm_extract(prompt)
                    print(f"=== RAW Relationship Extraction (attempt {attempt + 1}) ===\n{llm_output}")
                    _jlog("llm_relationship_extract_done", request_id, latency_sec=latency_seconds, attempt=attempt + 1)

                    relationships = parse_relationships_only(
                        llm_output, tuple_delimiter_val, record_delimiter_val,
                        completion_delimiter_val, valid_entity_names
                    )
                    relationship_count = len(relationships)
                    print(f"Parsed Relationships count: {relationship_count}")
                    _jlog("parse_relationship_extraction_done", request_id, relationship_count=relationship_count, attempt=attempt + 1)

                    if relationship_count == 0 and attempt < max_retries:
                        print(f"0 relationships extracted. Retrying... ({attempt + 1}/{max_retries})")
                        continue

                    print(f"✓ Relationship Extraction: {relationship_count} relationships")
                    return (True, relationships)

                except Exception as parse_error:
                    over_context = is_context_length_exceeded_error(parse_error)
                    if over_context:
                        _jlog(
                            "context_length_limit_exceeded",
                            request_id,
                            stage="relationship_extraction",
                            attempt=attempt + 1,
                            prompt_length=len(prompt),
                            extracted_entity_count=len(extracted_entities),
                            error=str(parse_error),
                        )
                    _jlog("parse_relationship_extraction_failed", request_id, error=str(parse_error), attempt=attempt + 1)
                    # An over-long prompt fails identically every time: the same
                    # prompt is resent unchanged. Retrying it only spends the backoff.
                    if not over_context and attempt < max_error_retries:
                        backoff = _RETRY_BACKOFF_SEC * (2 ** attempt)
                        print(f"Parse error: {parse_error}. Retrying in {backoff:.0f}s...")
                        _jlog(
                            "relationship_extraction_retry_backoff",
                            request_id,
                            attempt=attempt + 1,
                            backoff_sec=backoff,
                        )
                        time.sleep(backoff)
                        continue
                    else:
                        print(f"Relationship parse failed: {parse_error}")
                        return (False, f"validation_error: {parse_error}")

        raise RuntimeError("Relationship extraction retry loop ended without a result")
