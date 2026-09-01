"""Ingestion orchestration: conversation turns in, knowledge graph out.

Per turn the pipeline runs four steps, in this order and for these reasons:

  1. Summarize the turn and write it to the summaries vector store.
  2. Extract entities.
  3. Extract relationships, given the entities from step 2.
  4. Reconcile against existing graph state and write.

Steps 2 and 3 are separate LLM calls rather than one. Asking for entities and
relationships together produced relationships referencing entities that were
never emitted, because the model had no fixed entity set to point at; feeding
step 2's output into step 3 removes the ambiguity.

Step 4 is where the difficulty lives. A conversation mentions the same person
across many turns in different words, so a new extraction is usually an update
to an existing node, not a new one. `ExtractionSyncer` resolves that with a
vector search for similar entities followed by an LLM adjudication -- see
`services/entity_manager.py`.

Temporal handling threads through all four steps. Conversations say "last
Tuesday", which is only meaningful relative to the turn's timestamp, so
temporal expressions are resolved to absolute dates at ingest time. Resolving
them at query time instead would mean re-deriving a context that is no longer
available. The `_repair_temporal_entities` and `_temporal_*` helpers below
exist because the extractor sometimes returns a bare surface form where an
anchored one is needed.

`Ingestor` runs the full LLM path, including LLM adjudication in step 4.
"""

import os
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from grace_mem.domain.extraction import ExtractionResult
from grace_mem.ingestion.extractors import (
    EntityExtractor,
    RelationshipExtractor,
)
from grace_mem.ingestion.parsing import is_context_length_exceeded_error
from grace_mem.ingestion.prompts.config import EXTRA_KWARGS
from grace_mem.ingestion.prompts.extraction import (
    entity_extraction_only,
    relationship_extraction_only,
)
from grace_mem.ingestion.steps.compress import Compressor
from grace_mem.ingestion.steps.sync import ExtractionSyncer
from grace_mem.ingestion.temporal_repair import (
    _expected_temporal_marker_entities,
    _repair_temporal_entities,
)
from grace_mem.runtime.analysis_log import append_analysis_record
from grace_mem.runtime.logger_config import _StepTimer, make_module_jlog, setup_logger
from grace_mem.temporal import (
    TimeContext,
    augment_temporal_text,
    build_time_context,
    format_temporal_hints_for_prompt,
)
from grace_mem.temporal.query_time_parser import parse_query_time

_jlog = make_module_jlog(name="grace_mem.Ingestor", filename="kg_ingestor.jsonl")
_TRACE_PRETTY_LOG_DIR = os.environ.get("KG_TRACE_PRETTY_LOG_DIR", "logs")
_trace_pretty_log = setup_logger(
    name="kg_ingest_trace_pretty",
    log_dir=_TRACE_PRETTY_LOG_DIR,
    to_console=False,
)


class IngestionFailedError(RuntimeError):
    """Raised when neither the primary nor fallback ingest produced durable graph state."""


def _require_successful_ingest(results: list[tuple[str, bool, Any]]) -> dict[str, Any]:
    """Return the sync payload or raise when an ingest stage reported failure."""
    if not results:
        raise IngestionFailedError("ingest returned no stage result")

    failed = [(stage, payload) for stage, success, payload in results if not success]
    if failed:
        stage, payload = failed[0]
        raise IngestionFailedError(f"{stage} failed: {payload}")

    payload = results[0][2]
    if not isinstance(payload, dict):
        raise IngestionFailedError("ingest did not return a sync payload")
    if payload.get("graph_sync_ok", True) is False:
        error_count = payload.get("graph_sync_error_count", 0)
        raise IngestionFailedError(
            f"graph synchronization failed ({error_count} reported errors)"
        )
    return payload


@dataclass(frozen=True)
class IngestorConfig:
    """Tuning knobs for one ingestion run.

    Frozen because these values are logged once at startup as the run's
    provenance. A config that could drift mid-run would make that record a
    lie, and reproducing the run from it impossible.

    Attributes:
        summary_embed_dim: Must match the embedding model's output width, or
            the summaries store rejects every write.
        similar_entity_top_k: Candidates offered to the LLM when deciding
            whether an extracted entity is one already in the graph.
        entity_sim_threshold: Cosine floor for those candidates. The pair
            trades errors in opposite directions: too low and unrelated
            entities reach adjudication where the LLM may merge them; too high
            and genuine restatements never become candidates and the graph
            grows duplicate nodes for one person.
        summary_context_prev_k_default: Preceding turns included when
            summarizing. Non-zero because a turn read alone loses its
            referents -- "he said yes" needs the turns that named him.
        ingest_mode: Recorded for observability only; changing it alters no
            behaviour here.
        llm_tuple_delim: Field separator inside one extracted record.
        llm_record_delim: Separator between records.
        llm_completion_delim: End-of-output marker. All three are deliberately
            unlikely token sequences: the extractor parses free text, so a
            delimiter that could occur naturally in conversation would split a
            record mid-value.
    """

    summary_embed_dim: int = 1024

    similar_entity_top_k: int = 3
    entity_sim_threshold: float = 0.7

    summary_context_prev_k_default: int = 2

    ingest_mode: str = "turn_pairs"

    llm_tuple_delim: str = "<|>"
    llm_record_delim: str = "<|RECORD|>"
    llm_completion_delim: str = "<|COMPLETE|>"




class Ingestor:
    """Drives summarize -> extract -> reconcile for each conversation turn.

    Collaborators (llm, graph, vector manager, entity and relationship
    services) are injected rather than constructed here, which keeps resource
    construction outside the ingestion workflow.

    Thread-safety is partial and deliberate: `_lock` serializes the extractor
    calls that share LLM state, but the storage backends are assumed to handle
    their own concurrency. Give each worker its own Ingestor.
    """

    # Overridable per instance via __init__(config=...).
    DEFAULTS = IngestorConfig()

    def __init__(self, *, llm: Any, graph: Any, mgr: Any, ent_svc: Any, rel_svc: Any, config: dict | IngestorConfig | None = None) -> None:
        """Wire up the ingestion steps and resolve the effective config.

        Args:
            llm: Chat client for summarization and extraction.
            graph: FalkorDB graph backend.
            mgr: Vector store manager supplying the summaries collection.
            ent_svc: Entity manager owning identity resolution and merging.
            rel_svc: Relationship manager owning edge merging.
            config: A dict of overrides layered onto DEFAULTS, or a complete
                IngestorConfig used as-is. The dict form is accepted because
                the experiment configs are plain dicts and would otherwise
                each have to construct the dataclass.
        """
        self.llm = llm
        self.graph = graph
        self.vector_db_manager = mgr
        self.entity_service = ent_svc
        self.relationship_service = rel_svc

        if isinstance(config, dict) or config is None:
            base = self.DEFAULTS.__dict__.copy()
            base.update(config or {})
            self.cfg = IngestorConfig(**base)
        else:
            self.cfg = config

        self.summaries_vdb = self.vector_db_manager.get_summaries_vdb(dim=self.cfg.summary_embed_dim)
        self._lock = threading.Lock()

        # One shared lock across both extractors: they issue LLM calls that
        # must not interleave for a single turn, since relationship extraction
        # consumes the entity set the previous call produced.
        self._compressor = Compressor(summaries_vdb=self.summaries_vdb)
        self._entity_extractor = EntityExtractor(llm=self.llm, lock=self._lock, cfg=self.cfg)
        self._rel_extractor = RelationshipExtractor(llm=self.llm, lock=self._lock, cfg=self.cfg)
        self._syncer = ExtractionSyncer(
            llm=self.llm,
            graph=self.graph,
            entity_service=self.entity_service,
            relationship_service=self.relationship_service,
            cfg=self.cfg,
        )

        # The effective config is logged once here so a run's artifacts record
        # what actually ran, not what the defaults happen to be today.
        _jlog(
            "ingestor_initialized",
            request_id="INIT",
            summary_vdb_dim=self.cfg.summary_embed_dim,
            has_llm=bool(llm),
            has_graph=bool(graph),
            has_vector_db_manager=bool(mgr),
            has_entity_service=bool(ent_svc),
            has_relationship_service=bool(rel_svc),
            effective_config=self.cfg.__dict__,
        )

    @staticmethod
    def _format_ingest_trace_text(*, request_id: str, session_id: int | str, message_id: int, summary_id: str | None, entity_names: list[str], relationship_names: list[str], delta: dict[str, Any], failure_type: str | None = None) -> str:
        """Render one turn's ingest outcome as the human-readable trace block.

        The companion to the JSON record in `_log_ingest_delta`: same facts,
        written for someone scrolling a log rather than for a parser. Absent
        fields print as "-" so every block keeps the same shape and successive
        turns can be diffed against each other.
        """
        lines = [
            "=" * 80,
            f"request_id: {request_id}",
            f"session_id: {session_id}",
            f"message_id: {message_id}",
            f"summary_id: {summary_id or '-'}",
            f"entities_added: {delta.get('entities_added', 0)}",
            f"entities_updated: {delta.get('entities_updated', 0)}",
            f"relationships_added: {delta.get('relationships_added', 0)}",
            f"relationships_skipped: {delta.get('relationships_skipped', 0)}",
            f"graph_sync_ok: {delta.get('graph_sync_ok')}",
            f"failure_type: {failure_type or '-'}",
            f"entity_names: {'; '.join(entity_names) if entity_names else '-'}",
            f"relationship_names: {'; '.join(relationship_names) if relationship_names else '-'}",
        ]
        return "\n".join(lines)

    def _log_ingest_delta(
        self,
        *,
        request_id: str,
        session_id: int | str,
        message_id: int,
        summary_id: str | None,
        result: dict[str, Any] | None,
        entities: list[Any],
        relationships: list[Any],
        total_elapsed_sec: float,
    ) -> None:
        """Record what one turn actually changed in the graph, and flag no-ops.

        Ingestion rarely fails loudly. The common failure is a turn that runs
        clean and writes nothing -- the extractor returned nothing usable, or
        everything it returned was rejected during sync. End-to-end accuracy
        drops and there is no exception to point at. Classifying those two
        cases here is what makes them findable after the run:

        - ingest_zero_entities: extraction produced no entities at all.
        - ingest_zero_delta: entities were extracted but none survived to
          become a graph write.

        Both are written to the analysis log as a `failure_verdict` in addition
        to the normal delta record, so a sweep for silent losses does not have
        to re-derive the condition.
        """
        relationship_metas = (result or {}).get("relationship_metas") or []
        entity_summary = ((result or {}).get("entity_summary") or {})
        entities_added = int(entity_summary.get("added", 0) or 0)
        entities_updated = int(entity_summary.get("updated", 0) or 0)
        relationships_added = len(relationship_metas)
        relationships_skipped = max(len(relationships) - relationships_added, 0)
        failure_type = None
        if entities_added == 0 and entities_updated == 0 and not entities:
            failure_type = "ingest_zero_entities"
        elif entities_added == 0 and entities_updated == 0 and relationships_added == 0:
            failure_type = "ingest_zero_delta"
        delta: dict[str, Any] = {
            "request_id": request_id,
            "session_id": str(session_id),
            "message_id": message_id,
            "summary_id": summary_id,
            "entities_extracted": len(entities),
            "relationships_extracted": len(relationships),
            "entities_added": entities_added,
            "entities_updated": entities_updated,
            "relationships_added": relationships_added,
            "relationships_skipped": relationships_skipped,
            "graph_sync_ok": bool((result or {}).get("graph_sync_ok")),
            # Capped at 10: this record is written once per turn, and a
            # chatty turn would otherwise dominate the log with names that
            # add nothing to the counts already recorded above.
            "created_entity_names": [meta.get("name") for meta in ((result or {}).get("entity_idx") or {}).values()][:10],
            "created_relationship_names": [
                f"{meta.get('source_entity')} -> {meta.get('target_entity')}"
                for meta in relationship_metas[:10]
            ],
            "total_elapsed_sec": round(total_elapsed_sec, 6),
            "failure_type": failure_type,
        }
        append_analysis_record(_TRACE_PRETTY_LOG_DIR, "ingest_delta", delta)
        if failure_type:
            append_analysis_record(
                _TRACE_PRETTY_LOG_DIR,
                "failure_verdict",
                {
                    "scope": "ingest",
                    "request_id": request_id,
                    "session_id": str(session_id),
                    "message_id": message_id,
                    "failure_type": failure_type,
                    "summary_id": summary_id,
                },
            )
        _trace_pretty_log.info(
            self._format_ingest_trace_text(
                request_id=request_id,
                session_id=session_id,
                message_id=message_id,
                summary_id=summary_id,
                entity_names=delta["created_entity_names"],
                relationship_names=delta["created_relationship_names"],
                delta=delta,
                failure_type=failure_type,
            )
        )

    # ---------- Utility: Unified step logging ----------
    @contextmanager
    def log_step(self, event: str, request_id: str, **payload: Any) -> Iterator[None]:
        """Log a start/done/failed wrapper around one ingestion step."""
        t = _StepTimer()
        _jlog(f"{event}_start", request_id, **payload)
        try:
            yield
            _jlog(f"{event}_done", request_id, elapsed_sec=t.sec(), **payload)
        except Exception as e:
            _jlog(f"{event}_failed", request_id, error=str(e), error_type=type(e).__name__, elapsed_sec=t.sec(), **payload)
            raise

    # ---------- Thin delegation methods (preserve existing call sites) ----------
    def summarize_turn(self, session_id: int | str, message_id: int, user_text: str, assistant_text: str, request_id: str, dialogue_datetime: str | None = None, temporal_hints: list | None = None, tctx: TimeContext | None = None) -> tuple[str, str]:
        """Delegate turn summarization to the compressor with step logging."""
        with self.log_step("summarize_turn", request_id, session_id=session_id, message_id=message_id):
            return self._compressor.summarize_turn(
                session_id, message_id, user_text, assistant_text, request_id,
                dialogue_datetime=dialogue_datetime, temporal_hints=temporal_hints,
                tctx=tctx,
            )

    def extract_entities_only(self, prompt_vars: dict[str, Any], prompt_template: str, request_id: str, *, tuple_delim: str | None = None, record_delim: str | None = None, completion_delim: str | None = None, max_retries: int = 2) -> tuple[bool, Any]:
        """Run only the entity-extraction phase under unified step logging."""
        with self.log_step("entity_extraction", request_id):
            return self._entity_extractor.extract(
                prompt_vars, prompt_template, request_id,
                tuple_delim=tuple_delim, record_delim=record_delim,
                completion_delim=completion_delim, max_retries=max_retries,
            )

    def extract_relationships_only(self, prompt_vars: dict[str, Any], prompt_template: str, extracted_entities: list[Any], request_id: str, *, tuple_delim: str | None = None, record_delim: str | None = None, completion_delim: str | None = None, max_retries: int = 2) -> tuple[bool, Any]:
        """Run only the relationship-extraction phase under unified step logging."""
        with self.log_step("relationship_extraction", request_id):
            return self._rel_extractor.extract(
                prompt_vars, prompt_template, extracted_entities, request_id,
                tuple_delim=tuple_delim, record_delim=record_delim,
                completion_delim=completion_delim, max_retries=max_retries,
            )

    def apply_extraction_and_sync(self, result: ExtractionResult, provenance: dict | None = None, request_id: str = "UNKNOWN", *, entity_sim_topk: int | None = None, entity_sim_threshold: float | None = None) -> dict[str, Any]:
        """Apply extracted entities and relationships to the vector stores and graph."""
        return self._syncer.sync(
            result, provenance, request_id,
            entity_sim_topk=entity_sim_topk,
            entity_sim_threshold=entity_sim_threshold,
        )

    # ---------- Main Orchestration Methods ----------
    def ingest_turn(
        self,
        prompt_vars: dict,
        prompt_templates: dict | None = None,
        provenance: dict | None = None,
        request_id: str = "UNKNOWN",
        *,
        entity_sim_topk: int | None = None,
        entity_sim_threshold: float | None = None,
        temporal_hints: list | None = None,
        tctx: TimeContext | None = None,
    ) -> list[tuple[str, bool, Any]]:
        """
        TWO-STEP extraction: entities first, then relationships
        1) Extract entities using entity_extraction prompt
        2) Extract relationships using relationship_extraction prompt + extracted entities
        3) Apply & sync to VDB and FalkorDB
        """
        if prompt_templates is None:
            prompt_templates = {
                "entity_extraction": entity_extraction_only["entity_extraction"],
                "relationship_extraction": relationship_extraction_only["relationship_extraction"],
            }

        timer_total = _StepTimer()

        _jlog("ingest_turn_start", request_id, prompt_count=len(prompt_templates),
              prompt_names=list(prompt_templates.keys()), has_provenance=bool(provenance))

        # STEP 1: Extract entities
        entity_template = prompt_templates.get("entity_extraction")
        if not entity_template:
            _jlog("ingest_turn_failed", request_id, reason="Missing entity_extraction template")
            return [("entity_extraction", False, "Missing entity_extraction template")]

        entity_extraction_success, entities_or_error = self.extract_entities_only(
            prompt_vars=prompt_vars, prompt_template=entity_template, request_id=request_id)

        if not entity_extraction_success:
            _jlog("entity_extraction_failed", request_id, error=entities_or_error)
            return [("entity_extraction", False, entities_or_error)]

        entities = entities_or_error
        _jlog("entity_extraction_success", request_id, entity_count=len(entities))

        # STEP 2: Extract relationships (if template exists and entities found)
        relationships = []
        relationship_template = prompt_templates.get("relationship_extraction")

        if relationship_template and entities:
            relationship_extraction_success, relationships_or_error = self.extract_relationships_only(
                prompt_vars=prompt_vars, prompt_template=relationship_template,
                extracted_entities=entities, request_id=request_id)

            if relationship_extraction_success:
                relationships = relationships_or_error
                _jlog("relationship_extraction_success", request_id, relationship_count=len(relationships))
            else:
                _jlog("relationship_extraction_failed", request_id, error=relationships_or_error)
                return [("relationship_extraction", False, relationships_or_error)]
        else:
            _jlog("relationship_extraction_skipped", request_id,
                  reason="No template" if not relationship_template else "No entities")

        # Pre-write repair: fix relative-phrase temporal entity names before KG write.
        if temporal_hints:
            entities, relationships = _repair_temporal_entities(entities, relationships, temporal_hints, tctx)
            _jlog("temporal_entity_repair_done", request_id, entity_count=len(entities))

        # STEP 3: Build ExtractionResult and apply
        extraction_result = ExtractionResult(entities=entities, relationships=relationships)

        try:
            result = self.apply_extraction_and_sync(
                extraction_result, provenance=provenance, request_id=request_id,
                entity_sim_topk=entity_sim_topk, entity_sim_threshold=entity_sim_threshold)
            _jlog("apply_entity_ops_complete", request_id)
            _sync_ok = result.get("graph_sync_ok", True) if isinstance(result, dict) else True
            _jlog(
                "ingest_turn_complete",
                request_id,
                success=_sync_ok,
                entity_count=len(entities),
                relationship_count=len(relationships),
                total_elapsed_sec=timer_total.sec(),
                graph_sync_ok=_sync_ok,
                graph_sync_missing_entity_count=result.get("graph_sync_missing_entity_count", 0) if isinstance(result, dict) else 0,
                graph_sync_missing_relationship_count=result.get("graph_sync_missing_relationship_count", 0) if isinstance(result, dict) else 0,
                graph_sync_error_count=result.get("graph_sync_error_count", 0) if isinstance(result, dict) else 0,
            )
            if isinstance(result, dict):
                result["extracted_entities"] = entities
                result["extracted_relationships"] = relationships
            return [("two_step_extraction", _sync_ok, result)]

        except Exception as processing_error:
            _jlog("apply_entity_ops_failed", request_id, error=str(processing_error), error_type=type(processing_error).__name__)
            return [("two_step_extraction", False, f"apply_entity_ops_error: {processing_error}")]

    # ---------- Main Workflow ----------
    def summarize_and_ingest_turn(
        self,
        session_id: int | str,
        message_id: int,
        user_text: str,
        assistant_text: str,
        prev_k: int | None = None,
        *,
        entity_sim_topk: int | None = None,
        entity_sim_threshold: float | None = None,
        dialogue_datetime: str | None = None,
    ) -> Any:
        """
        Generate summary from query & response → TWO-STEP extraction (entities → relationships) → Write to VDB & KG
        """
        prev_k = prev_k if prev_k is not None else self.cfg.summary_context_prev_k_default

        request_id = str(uuid.uuid4())
        timer_total = _StepTimer()

        _jlog("summarize_and_ingest_turn_start", request_id,
              session_id=session_id, message_id=message_id,
              user_text_length=len(user_text), assistant_text_length=len(assistant_text), prev_k=prev_k)

        summary_id: str | None = None
        try:
            # Step 0: Deterministic temporal parsing.
            # Rewrite relative temporal phrases directly to resolved absolute values.
            # Hints are kept as secondary metadata for summary annotation and pre-write repair.
            temporal_hints: list[dict] = []
            _tctx: TimeContext | None = None
            aug_user_text = user_text
            aug_asst_text = assistant_text

            if dialogue_datetime:
                reference_dt = parse_query_time(dialogue_datetime)
                if reference_dt is not None:
                    _tctx = build_time_context(
                        reference_dt=reference_dt,
                        reference_time_str=dialogue_datetime,
                        source="ingestor",
                    )
                    seen_originals: set[str] = set()
                    for _raw, _label in [(user_text, "user"), (assistant_text, "assistant")]:
                        if not _raw or not _raw.strip():
                            continue
                        _augmented, _hints, _unresolved = augment_temporal_text(_raw, _tctx)
                        print(f"Augmented {_label.capitalize()} Text:\n{_augmented}\n")
                        if _label == "user":
                            aug_user_text = _augmented
                        else:
                            aug_asst_text = _augmented
                        for h in _hints:
                            if h["original"].lower() not in seen_originals:
                                seen_originals.add(h["original"].lower())
                                temporal_hints.append(h)
                        if _unresolved:
                            _jlog(
                                "temporal_unresolved_phrases",
                                request_id,
                                source=_label,
                                reference_time=dialogue_datetime,
                                phrases=[c.original_text for c in _unresolved],
                                statuses=[
                                    {"text": c.original_text,
                                     "status": c.resolution.status.value}
                                    for c in _unresolved
                                ],
                            )
                    if temporal_hints:
                        _jlog(
                            "temporal_hints_extracted",
                            request_id,
                            reference_time=dialogue_datetime,
                            hints=[{"original": h["original"], "resolved_to": h["resolved_to"]} for h in temporal_hints],
                        )

            # 1) Generate summary and write to VDB.
            # Pass temporal_hints so the stored summary includes resolved absolute dates.
            summary_id, _summary_text = self.summarize_turn(
                session_id=session_id, message_id=message_id,
                user_text=user_text, assistant_text=assistant_text,
                request_id=request_id, dialogue_datetime=dialogue_datetime,
                temporal_hints=temporal_hints if temporal_hints else None,
                tctx=_tctx,
            )

            # 2) Build variables for two-step extraction using rewritten text.
            # The rewritten conversation is the primary temporal signal for the LLM.
            # {temporal_hints} is kept as a secondary reference only.
            if aug_asst_text and aug_asst_text.strip():
                curr_text = f"User: {aug_user_text.strip()}\nAssistant: {aug_asst_text.strip()}"
            else:
                curr_text = aug_user_text.strip()
            variables = {**EXTRA_KWARGS, "raw_conversation": curr_text}
            variables["dialogue_datetime"] = dialogue_datetime or "not available"
            variables["temporal_hints"] = format_temporal_hints_for_prompt(temporal_hints)

            # Use two-step prompts
            prompt_template = {
                "entity_extraction": entity_extraction_only["entity_extraction"],
                "relationship_extraction": relationship_extraction_only["relationship_extraction"]
            }

            _jlog("using_prompt_template", request_id, template_name="two_step (entity + relationship)")

            # Print assembled entity extraction prompt (for debugging)
            print("=" * 80)
            print("=== ENTITY EXTRACTION PROMPT ===")
            print("=" * 80)
            print(prompt_template["entity_extraction"].format(**variables))
            print("=" * 80)

            # Build provenance
            provenance = {"summary_ids": [summary_id], "session_id": session_id, "message_id": message_id}
            if dialogue_datetime is not None:
                provenance["dialogue_datetime"] = dialogue_datetime

            results = self.ingest_turn(
                prompt_vars=variables, prompt_templates=prompt_template,
                provenance=provenance, request_id=request_id,
                entity_sim_topk=entity_sim_topk, entity_sim_threshold=entity_sim_threshold,
                temporal_hints=temporal_hints if temporal_hints else None,
                tctx=_tctx)

            sync_payload = _require_successful_ingest(results)
            _jlog("llm_extraction_and_syncKG_done", request_id)
            _jlog("summarize_and_ingest_turn_complete", request_id, success=True, total_elapsed_sec=timer_total.sec())

            # Post-extraction guardrail: warn if any entity name is a temporal literal that
            # the parser already resolved — indicates LLM ignored the pre-resolved hints.
            if temporal_hints:
                hint_originals = {h["original"].lower() for h in temporal_hints}
                extracted_temporal_keys = {
                    (
                        getattr(getattr(_ent, "entity_type", None), "value", ""),
                        getattr(_ent, "entity_name", "").strip(),
                    )
                    for _ent in sync_payload.get("extracted_entities", [])
                }
                for _ent in sync_payload.get("extracted_entities", []):
                    _name = getattr(_ent, "entity_name", "").lower().strip()
                    if _name in hint_originals:
                        _jlog(
                            "temporal_literal_entity_name_warning",
                            request_id,
                            entity_name=getattr(_ent, "entity_name", ""),
                            entity_type=getattr(getattr(_ent, "entity_type", None), "value", ""),
                            expected_resolved=[
                                h["resolved_to"] for h in temporal_hints
                                if h["original"].lower() == _name
                            ],
                        )
                missing_temporal_markers = [
                    marker
                    for marker in _expected_temporal_marker_entities(temporal_hints)
                    if (marker["entity_type"], marker["entity_name"]) not in extracted_temporal_keys
                ]
                if missing_temporal_markers:
                    _jlog(
                        "temporal_marker_entities_missing",
                        request_id,
                        missing_entities=missing_temporal_markers,
                        extracted_temporal_entities=sorted(extracted_temporal_keys),
                    )
            self._log_ingest_delta(
                request_id=request_id,
                session_id=session_id,
                message_id=message_id,
                summary_id=summary_id,
                result=sync_payload,
                entities=sync_payload.get("extracted_entities", []),
                relationships=sync_payload.get("extracted_relationships", []),
                total_elapsed_sec=timer_total.sec(),
            )
            return {
                "request_id": request_id,
                "summary_id": summary_id,
                "results": results,
            }

        except Exception as e:
            if is_context_length_exceeded_error(e):
                _jlog(
                    "context_length_limit_exceeded",
                    request_id,
                    stage="summarize_and_ingest_turn",
                    session_id=session_id,
                    message_id=message_id,
                    user_text_length=len(user_text),
                    assistant_text_length=len(assistant_text),
                    prev_k=prev_k,
                    error=str(e),
                )
            _jlog("summarize_and_ingest_turn_failed", request_id,
                  error=str(e), error_type=type(e).__name__, total_elapsed_sec=timer_total.sec())

            # Fallback: try with two-step extraction directly on original text.
            # A successful fallback is safe to checkpoint; a failed fallback
            # raises so benchmark runners cannot mark this session complete.
            try:
                _jlog("fallback_ingest_start", request_id, reason="primary_ingest_failed")
                if assistant_text and assistant_text.strip():
                    curr_text = f"User: {user_text.strip()}\nAssistant: {assistant_text.strip()}"
                else:
                    curr_text = user_text.strip()
                prompt_vars = {**EXTRA_KWARGS, "raw_conversation": curr_text}
                prompt_vars["dialogue_datetime"] = dialogue_datetime or "not available"
                prompt_vars["temporal_hints"] = format_temporal_hints_for_prompt(temporal_hints)

                fallback_template = {
                    "entity_extraction": entity_extraction_only["entity_extraction"],
                    "relationship_extraction": relationship_extraction_only["relationship_extraction"]
                }

                fallback_provenance = None
                if summary_id is not None:
                    fallback_provenance = {
                        "summary_ids": [summary_id],
                        "session_id": session_id,
                        "message_id": message_id,
                    }
                    if dialogue_datetime is not None:
                        fallback_provenance["dialogue_datetime"] = dialogue_datetime

                fallback_results = self.ingest_turn(
                    prompt_vars=prompt_vars, prompt_templates=fallback_template,
                    provenance=fallback_provenance, request_id=request_id,
                    entity_sim_topk=entity_sim_topk, entity_sim_threshold=entity_sim_threshold)

                _require_successful_ingest(fallback_results)
                _jlog("fallback_ingest_complete", request_id)
                append_analysis_record(
                    _TRACE_PRETTY_LOG_DIR,
                    "failure_verdict",
                    {
                        "scope": "ingest",
                        "request_id": request_id,
                        "session_id": str(session_id),
                        "message_id": message_id,
                        "failure_type": "ingest_exception",
                        "error": str(e),
                    },
                )
                return {
                    "request_id": request_id,
                    "recovered_error": str(e),
                    "ingest_results": fallback_results,
                }
            except Exception as fallback_error:
                if is_context_length_exceeded_error(fallback_error):
                    _jlog(
                        "context_length_limit_exceeded",
                        request_id,
                        stage="fallback_ingest",
                        session_id=session_id,
                        message_id=message_id,
                        user_text_length=len(user_text),
                        assistant_text_length=len(assistant_text),
                        prev_k=prev_k,
                        error=str(fallback_error),
                    )
                _jlog("fallback_ingest_failed", request_id,
                      fallback_error=str(fallback_error), fallback_error_type=type(fallback_error).__name__)
                append_analysis_record(
                    _TRACE_PRETTY_LOG_DIR,
                    "failure_verdict",
                    {
                        "scope": "ingest",
                        "request_id": request_id,
                        "session_id": str(session_id),
                        "message_id": message_id,
                        "failure_type": "ingest_exception",
                        "error": str(e),
                        "fallback_error": str(fallback_error),
                    },
                )
                raise IngestionFailedError(
                    f"primary ingest failed: {e}; fallback ingest failed: {fallback_error}"
                ) from fallback_error
