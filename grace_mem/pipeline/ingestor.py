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

`Ingestor` runs the full LLM path. `IngestorNoEntityOps` at the bottom is the
ablation that skips LLM adjudication in step 4 and merges on exact key match,
which is what isolates that call's contribution to end-to-end accuracy.
"""

import re
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional
from dataclasses import dataclass
import os

from grace_mem.llm.prompts import EXTRA_KWARGS, entity_extraction_only, relationship_extraction_only
from grace_mem.utils.common import Entity, EntityType, ExtractionResult, Relationship, _entity_key, is_context_length_exceeded_error
from grace_mem.utils.logger_config import _StepTimer, make_module_jlog, setup_logger
from grace_mem.utils.query_time_parser import parse_query_time
from grace_mem.utils.temporal import (
    TimeContext,
    augment_temporal_text,
    build_time_context,
    extract_temporal_hints,
    format_temporal_hints_for_prompt,
    rewrite_temporal_text,
)
from grace_mem.utils.error_analysis import append_analysis_record

from grace_mem.pipeline.ingest_steps.compress import Compressor
from grace_mem.pipeline.ingest_steps.extract import EntityExtractor, RelationshipExtractor
from grace_mem.pipeline.ingest_steps.sync import ExtractionSyncer


_jlog = make_module_jlog(name="grace_mem.Ingestor", filename="kg_ingestor.jsonl")
_TRACE_PRETTY_LOG_DIR = os.environ.get("KG_TRACE_PRETTY_LOG_DIR", "logs")
_trace_pretty_log = setup_logger(
    name="kg_ingest_trace_pretty",
    log_dir=_TRACE_PRETTY_LOG_DIR,
    to_console=False,
)


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


def _repair_temporal_entities(
    entities: list[Entity],
    relationships: list[Relationship],
    temporal_hints: list[dict],
    tctx: Optional[TimeContext] = None,
) -> tuple[list[Entity], list[Relationship]]:
    """Normalize temporal entity names to resolved values before the KG write.

    A time entity is only useful in the graph if its name is absolute. Two
    mentions of "last Tuesday" from different weeks must not collapse into one
    node, and a query for a date range cannot match a relative phrase. So every
    Date/Time/Timespan name has to become a resolved value before it is written
    -- after that point the turn's reference time is gone and the phrase can
    never be resolved again.

    The repairs, all driven by `temporal_hints` extracted upstream:

    - Date/Timespan named by a relative phrase -> resolved ISO value.
    - Date/Timespan named with bracket marker syntax ("[TIMESPAN: 2022]") ->
      stripped. The extraction prompt uses those markers; they are scaffolding,
      not part of the name.
    - Date/Timespan whose name is an unambiguous suffix of an expected marker
      -> renamed to the full marker.
    - Event names carrying a relative phrase or embedded marker -> cleaned.
    - Descriptions rewritten the same way, so retrieval text agrees with the
      node name.
    - Relationship endpoints follow any rename, or the edge would point at a
      node that no longer exists under that name.
    - Expected marker entities the extractor missed -> injected, so a hint that
      was resolved does not vanish for want of the model mentioning it.

    Returns:
        The repaired (entities, relationships). Inputs are not mutated.
    """
    _TEMPORAL_TYPES = (EntityType.Date, EntityType.Time, EntityType.Timespan)
    # A whole name that is a bracket marker: "[TIMESPAN: the weekend before 2023-07-15]"
    _BRACKET_RE = re.compile(r'^\[(?:DATE|TIMESPAN|TIME):\s*(.*?)\]$', re.IGNORECASE)
    # The same markers embedded mid-string, which is how they show up in Event names.
    _EMBEDDED_BRACKET_RE = re.compile(r'\s*\[(?:DATE|TIMESPAN|TIME):[^\]]*\]', re.IGNORECASE)

    hint_lookup: dict[str, dict] = {h["original"].lower(): h for h in temporal_hints}
    # Reverse index, resolved value -> hint. Needed because the model sometimes
    # does the resolution itself and emits "2023-05-29 to 2023-06-04" directly.
    # Without this the entity would look unmatched and lose its provenance back
    # to the original phrase ("last week"), which the temporal metadata records.
    resolved_lookup: dict[str, dict] = {}
    for h in temporal_hints:
        for key in (h.get("display_value"), h.get("resolved_to")):
            if key and key.lower() not in resolved_lookup:
                resolved_lookup[key.lower()] = h
    rename_map: dict[str, str] = {}

    # (entity_type_str, entity_name) → hint, for every expected marker entity
    marker_hint_for: dict[tuple[str, str], dict] = {}
    for h in temporal_hints:
        for m in (h.get("markers") or []):
            key = (m["entity_type"], m["entity_name"])
            if key not in marker_hint_for:
                marker_hint_for[key] = h

    def _temporal_meta_for(name: str, etype: EntityType) -> Optional[dict]:
        """Build the temporal metadata block for one entity name, or None.

        Two sources, in priority order. A precomputed hint is preferred because
        it was resolved against the turn's own reference time. Failing that,
        the name is re-parsed against `tctx` -- which catches phrases the hint
        extractor missed but the entity extractor surfaced.

        Returns None when neither path yields a resolution, meaning the name is
        not temporal after all and should be left alone.
        """
        key = name.lower().strip()
        hint = hint_lookup.get(key) or resolved_lookup.get(key)
        if hint:
            return {
                "temporal": {
                    "display_value": hint.get("display_value") or hint.get("resolved_to"),
                    "normalized_time": hint.get("normalized_time"),
                    "normalized_start": hint.get("normalized_start"),
                    "normalized_end": hint.get("normalized_end"),
                    "granularity": hint.get("granularity"),
                    "original_phrase": hint.get("original"),
                    "reference_time": hint.get("reference_time"),
                    "status": hint.get("status"),
                    "confidence": hint.get("confidence"),
                }
            }
        if tctx is None:
            return None
        rewritten, meta = rewrite_temporal_text(name, tctx)
        constraints = meta.get("constraints") or []
        if not constraints:
            return None
        resolution = (constraints[0] or {}).get("resolution") or {}
        display = resolution.get("display_value") or rewritten
        return {
            "temporal": {
                "display_value": display,
                "normalized_time": resolution.get("normalized_time"),
                "normalized_start": resolution.get("normalized_start"),
                "normalized_end": resolution.get("normalized_end"),
                "granularity": resolution.get("granularity"),
                "original_phrase": constraints[0].get("original_text"),
                "reference_time": resolution.get("reference_time"),
                "status": resolution.get("status"),
                "confidence": resolution.get("confidence"),
            }
        }

    def _temporal_anchor_description(name: str, etype: EntityType, meta: Optional[dict]) -> str:
        """Compose a description for a temporal node the extractor left bare.

        Injected marker entities arrive with no description, and a node with an
        empty description is effectively invisible to retrieval: the entity
        vector store embeds the description, so an empty one embeds to noise.
        Stating the granularity in words also gives dense search something to
        match a query like "that week" against.
        """
        temporal = (meta or {}).get("temporal") or {}
        granularity = temporal.get("granularity")
        if etype == EntityType.Date:
            return f"The calendar date {name}."
        if etype == EntityType.Time:
            return f"The clock time {name}."
        if granularity == "week":
            return f"The week-long timespan {name}."
        if granularity == "weekend":
            return f"The weekend timespan {name}."
        if granularity == "month":
            return f"The month-long timespan {name}."
        if granularity == "season":
            return f"The seasonal timespan {name}."
        if granularity == "year":
            return f"The year-long timespan {name}."
        return f"The timespan {name}."

    def _prefer_existing_temporal_description(
        description: Optional[str],
        *,
        name: str,
        etype: EntityType,
        meta: Optional[dict],
    ) -> str:
        """Keep the extractor's description, falling back to a generated anchor.

        Only fills a genuine gap. The extracted description says what the date
        meant in the conversation; the generated one only says what kind of
        date it is, so overwriting would trade specific text for generic.
        """
        if description and description.strip():
            return description.strip()
        return _temporal_anchor_description(name, etype, meta)

    repaired: list[Entity] = []
    covered_markers: set[tuple[str, str]] = set()

    for ent in entities:
        name = ent.entity_name
        etype = ent.entity_type
        desc = ent.entity_description
        meta = getattr(ent, "entity_metadata", None)
        if tctx is not None and desc and etype == EntityType.Event:
            desc, _ = rewrite_temporal_text(desc, tctx)
        name_lower = name.lower().strip()

        if etype in _TEMPORAL_TYPES:
            original_name = name

            # Fix 1: strip bracket marker syntax like [TIMESPAN: 2022] → 2022
            _bracket = _BRACKET_RE.match(name.strip())
            if _bracket:
                name = _bracket.group(1).strip()
                name_lower = name.lower()

            hint = hint_lookup.get(name_lower)
            resolved = hint.get("resolved_to") if hint else None
            # Fallback: re-resolve relative phrases (e.g. "Yesterday") via the parser.
            # Skip names that already contain an ISO date (YYYY-MM-DD) — re-parsing
            # those causes two bugs: date ranges get truncated to the first date, and
            # "night of YYYY-MM-DD" shifts one day due to daypart boundary semantics.
            _has_iso_date = bool(re.search(r'\d{4}-\d{2}-\d{2}', name))
            if resolved is None and tctx is not None and not _has_iso_date:
                sub_hints = extract_temporal_hints([name], tctx)
                if sub_hints:
                    hint = sub_hints[0]
                    resolved = hint.get("resolved_to")
            final_name = resolved or name

            # Fix 2: partial extract — if still unresolved, rename when name is an
            # unambiguous suffix of exactly one expected marker entity of the same type.
            # e.g. "before 2023-07-15" → "the weekend before 2023-07-15"
            if final_name == name:
                _candidates = [
                    mname
                    for (mtype, mname) in marker_hint_for
                    if mtype == etype.value and mname.endswith(name) and mname != name
                ]
                if len(_candidates) == 1:
                    final_name = _candidates[0]

            final_meta = meta or _temporal_meta_for(name, etype) or _temporal_meta_for(final_name, etype)
            final_desc = _prefer_existing_temporal_description(
                desc,
                name=final_name,
                etype=etype,
                meta=final_meta,
            )
            repaired.append(Entity(entity_name=final_name, entity_type=etype, entity_description=final_desc, entity_metadata=final_meta))
            if final_name != original_name:
                rename_map[original_name] = final_name
                _jlog("temporal_entity_name_repaired", "repair",
                      original_name=original_name, repaired_name=final_name, entity_type=etype.value)
            covered_markers.add((etype.value, final_name))
            continue

        elif etype == EntityType.Event:
            new_name = name
            # strip embedded bracket markers (e.g. "pottery workshop[TIMESPAN: 2022]")
            new_name = _EMBEDDED_BRACKET_RE.sub("", new_name).strip(" ,.-")
            for original_phrase, hint in hint_lookup.items():
                pattern = re.compile(re.escape(original_phrase), re.IGNORECASE)
                if pattern.search(new_name):
                    stripped = pattern.sub("", new_name).strip(" ,.-")
                    if stripped:
                        new_name = stripped
            if new_name != name:
                repaired.append(Entity(entity_name=new_name, entity_type=etype, entity_description=desc, entity_metadata=meta))
                rename_map[name] = new_name
                _jlog("temporal_entity_name_repaired", "repair",
                      original_name=name, repaired_name=new_name, entity_type=etype.value)
                continue

        repaired.append(Entity(entity_name=name, entity_type=etype, entity_description=desc, entity_metadata=meta))

    # Fix 3: inject expected marker entities that the LLM omitted entirely
    for (mtype, mname), hint in marker_hint_for.items():
        if (mtype, mname) in covered_markers:
            continue
        try:
            etype = EntityType(mtype)
        except ValueError:
            continue
        inj_meta: dict = {
            "temporal": {
                "display_value": hint.get("display_value") or hint.get("resolved_to"),
                "normalized_time": hint.get("normalized_time"),
                "normalized_start": hint.get("normalized_start"),
                "normalized_end": hint.get("normalized_end"),
                "granularity": hint.get("granularity"),
                "original_phrase": hint.get("original"),
                "reference_time": hint.get("reference_time"),
                "status": hint.get("status"),
                "confidence": hint.get("confidence"),
            }
        }
        inj_desc = _temporal_anchor_description(mname, etype, inj_meta)
        repaired.append(Entity(entity_name=mname, entity_type=etype, entity_description=inj_desc, entity_metadata=inj_meta))
        _jlog("temporal_entity_injected", "repair",
              entity_name=mname, entity_type=mtype, original_phrase=hint.get("original"))

    # Fix 4: fallback hard-inject for hints whose markers list was empty (or whose
    # resolved name wasn't captured by Fix 3).  Uses granularity to pick entity type.
    _GRANULARITY_TO_ETYPE: dict[str, EntityType] = {
        "day": EntityType.Date,
        "time": EntityType.Time,
    }
    _covered_names_lower: set[str] = {e.entity_name.lower() for e in repaired}
    for hint in temporal_hints:
        candidate = hint.get("display_value") or hint.get("resolved_to")
        if not candidate:
            continue
        if candidate.lower() in _covered_names_lower:
            continue
        granularity = (hint.get("granularity") or "").lower()
        etype = _GRANULARITY_TO_ETYPE.get(granularity, EntityType.Timespan)
        # skip if normalized_time present but granularity says day — time wins
        if hint.get("normalized_time") and etype == EntityType.Date:
            etype = EntityType.Time
        inj_meta: dict = {
            "temporal": {
                "display_value": hint.get("display_value") or hint.get("resolved_to"),
                "normalized_time": hint.get("normalized_time"),
                "normalized_start": hint.get("normalized_start"),
                "normalized_end": hint.get("normalized_end"),
                "granularity": hint.get("granularity"),
                "original_phrase": hint.get("original"),
                "reference_time": hint.get("reference_time"),
                "status": hint.get("status"),
                "confidence": hint.get("confidence"),
            }
        }
        inj_desc = _temporal_anchor_description(candidate, etype, inj_meta)
        repaired.append(Entity(entity_name=candidate, entity_type=etype, entity_description=inj_desc, entity_metadata=inj_meta))
        _covered_names_lower.add(candidate.lower())
        _jlog("temporal_entity_hint_fallback_injected", "repair",
              entity_name=candidate, entity_type=etype.value, original_phrase=hint.get("original"))

    repaired_relationships: list[Relationship] = []
    for rel in relationships:
        repaired_relationships.append(
            Relationship(
                source_entity=rename_map.get(rel.source_entity, rel.source_entity),
                target_entity=rename_map.get(rel.target_entity, rel.target_entity),
                relationship_description=rel.relationship_description,
                relationship_keywords=rel.relationship_keywords,
            )
        )

    return repaired, repaired_relationships


def _expected_temporal_marker_entities(temporal_hints: list[dict]) -> list[dict[str, str]]:
    """List the temporal entities the hints say the extractor should have found.

    Every resolved hint declares the marker entities it implies; comparing that
    list against what extraction actually returned is how `_repair_temporal_entities`
    knows what to inject. Deduplicated on (type, name) because several phrases
    in one turn commonly resolve to the same date -- "yesterday" and "the 14th"
    should yield one node, not two.

    Markers missing either field are skipped rather than defaulted: a nameless
    entity cannot be matched against extraction output or written to the graph.
    """
    expected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for hint in temporal_hints or []:
        for marker in hint.get("markers") or []:
            entity_type = (marker.get("entity_type") or "").strip()
            entity_name = (marker.get("entity_name") or "").strip()
            if not entity_type or not entity_name:
                continue
            key = (entity_type, entity_name)
            if key in seen:
                continue
            seen.add(key)
            expected.append(
                {
                    "entity_type": entity_type,
                    "entity_name": entity_name,
                    "original": hint.get("original", ""),
                    "marker_type": marker.get("marker_type", ""),
                }
            )
    return expected


class Ingestor:
    """Drives summarize -> extract -> reconcile for each conversation turn.

    Collaborators (llm, graph, vector manager, entity and relationship
    services) are injected rather than constructed here, which is what lets the
    experiment harness swap in the FalkorDB or Neo4j backend, and lets the
    ablations subclass one step without rebuilding the pipeline.

    Thread-safety is partial and deliberate: `_lock` serializes the extractor
    calls that share LLM state, but the storage backends are assumed to handle
    their own concurrency. Give each worker its own Ingestor.
    """

    # Overridable per instance via __init__(config=...).
    DEFAULTS = IngestorConfig()

    def __init__(self, *, llm: Any, graph: Any, mgr: Any, ent_svc: Any, rel_svc: Any, config: Optional[dict | IngestorConfig] = None) -> None:
        """Wire up the ingestion steps and resolve the effective config.

        Args:
            llm: Chat client for summarization and extraction.
            graph: Graph backend, either graph/falkordb.py or graph/neo4j.py.
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
        delta = {
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
    def summarize_turn(self, session_id: int | str, message_id: int, user_text: str, assistant_text: str, prev_k: int | None, request_id: str, dialogue_datetime: Optional[str] = None, temporal_hints: Optional[list] = None, tctx: Optional[TimeContext] = None) -> tuple[str, str]:
        """Delegate turn summarization to the compressor with step logging."""
        with self.log_step("summarize_turn", request_id, session_id=session_id, message_id=message_id):
            return self._compressor.summarize_turn(
                session_id, message_id, user_text, assistant_text, request_id,
                dialogue_datetime=dialogue_datetime, temporal_hints=temporal_hints,
                tctx=tctx,
            )

    def extract_entities_only(self, prompt_vars: dict[str, Any], prompt_template: str, request_id: str, *, tuple_delim: Optional[str] = None, record_delim: Optional[str] = None, completion_delim: Optional[str] = None, max_retries: int = 2) -> tuple[bool, Any]:
        """Run only the entity-extraction phase under unified step logging."""
        with self.log_step("entity_extraction", request_id):
            return self._entity_extractor.extract(
                prompt_vars, prompt_template, request_id,
                tuple_delim=tuple_delim, record_delim=record_delim,
                completion_delim=completion_delim, max_retries=max_retries,
            )

    def extract_relationships_only(self, prompt_vars: dict[str, Any], prompt_template: str, extracted_entities: List[Any], request_id: str, *, tuple_delim: Optional[str] = None, record_delim: Optional[str] = None, completion_delim: Optional[str] = None, max_retries: int = 2) -> tuple[bool, Any]:
        """Run only the relationship-extraction phase under unified step logging."""
        with self.log_step("relationship_extraction", request_id):
            return self._rel_extractor.extract(
                prompt_vars, prompt_template, extracted_entities, request_id,
                tuple_delim=tuple_delim, record_delim=record_delim,
                completion_delim=completion_delim, max_retries=max_retries,
            )

    def apply_extraction_and_sync(self, result: ExtractionResult, provenance: Optional[dict] = None, request_id: str = "UNKNOWN", *, entity_sim_topk: Optional[int] = None, entity_sim_threshold: Optional[float] = None) -> dict[str, Any]:
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
        prompt_templates: dict = None,
        provenance: Optional[dict] = None,
        request_id: str = "UNKNOWN",
        *,
        entity_sim_topk: Optional[int] = None,
        entity_sim_threshold: Optional[float] = None,
        temporal_hints: Optional[list] = None,
        tctx: Optional[TimeContext] = None,
    ) -> List[tuple[str, bool, Any]]:
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
                success=True,
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
            return [("two_step_extraction", True, result)]

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
        prev_k: Optional[int] = None,
        *,
        entity_sim_topk: Optional[int] = None,
        entity_sim_threshold: Optional[float] = None,
        dialogue_datetime: Optional[str] = None,
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

        try:
            # Step 0: Deterministic temporal parsing.
            # Rewrite relative temporal phrases directly to resolved absolute values.
            # Hints are kept as secondary metadata for summary annotation and pre-write repair.
            temporal_hints: list[dict] = []
            _tctx: Optional[TimeContext] = None
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
                prev_k=prev_k, request_id=request_id, dialogue_datetime=dialogue_datetime,
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

            _jlog("llm_extraction_and_syncKG_done", request_id)
            _jlog("summarize_and_ingest_turn_complete", request_id, success=True, total_elapsed_sec=timer_total.sec())
            sync_payload = results[0][2] if results and results[0][1] and isinstance(results[0][2], dict) else {}

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

            # Fallback: try with two-step extraction directly on original text
            try:
                _jlog("fallback_ingest_start", request_id, reason="summary_generation_failed")
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

                fallback_results = self.ingest_turn(
                    prompt_vars=prompt_vars, prompt_templates=fallback_template,
                    provenance=None, request_id=request_id,
                    entity_sim_topk=entity_sim_topk, entity_sim_threshold=entity_sim_threshold)

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
                return {"request_id": request_id, "error": str(e), "ingest_results": fallback_results}
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
                return {"request_id": request_id, "error": str(e), "fallback_error": str(fallback_error)}


class IngestorNoEntityOps(Ingestor):
    """
    Ingestor variant that skips LLM generate_entity_ops.

    Exact (name, type) matches are UPDATED; otherwise ADD. This keeps the same
    ingestion/extraction flow as Ingestor, but replaces similarity-based,
    LLM-driven entity canonicalization with deterministic cache lookup.
    """

    @staticmethod
    def _merge_desc(existing_desc: str, new_desc: str) -> str:
        """Append a new description only when it adds non-duplicate content."""
        existing_desc = (existing_desc or "").strip()
        new_desc = (new_desc or "").strip()
        if not existing_desc:
            return new_desc
        if not new_desc or new_desc in existing_desc:
            return existing_desc
        return f"{existing_desc}; {new_desc}"

    def _build_ops_without_llm(self, entities: List[Any]) -> dict[str, Any]:
        """
        Build ops_results for EntityManager.apply_ops without LLM.

        Uses exact match in cache (name+type) to decide UPDATE vs ADD.
        """
        cache = getattr(self.entity_service, "_GLOBAL_CACHE", {}) or {}
        ent_cache = cache.get("entities", {}) or {}

        normalized = self.entity_service.normalize_entities(entities)
        results: list[dict[str, Any]] = []

        for entity in normalized:
            name = (entity.get("entity_name") or "").strip()
            type_val = (entity.get("entity_type") or "").strip()
            desc = (entity.get("entity_description") or "").strip()
            if not name:
                continue

            key_nt = _entity_key(name, type_val)
            existing = ent_cache.get(key_nt)

            if existing:
                merged_desc = self._merge_desc(existing.get("description", ""), desc)
                results.append(
                    {
                        "input_name": name,
                        "input_type": type_val,
                        "action": "UPDATE",
                        "target_existing_id": existing.get("id"),
                        "canonical_name": existing.get("name") or name,
                        "canonical_type": existing.get("type") or type_val,
                        "merged_description": merged_desc or desc,
                    }
                )
            else:
                results.append(
                    {
                        "input_name": name,
                        "input_type": type_val,
                        "action": "ADD",
                        "target_existing_id": None,
                        "canonical_name": name,
                        "canonical_type": type_val,
                        "merged_description": desc,
                    }
                )

        return {"results": results}

    def apply_extraction_and_sync(
        self,
        result: ExtractionResult,
        provenance: Optional[dict] = None,
        request_id: str = "UNKNOWN",
        *,
        entity_sim_topk: Optional[int] = None,
        entity_sim_threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Apply extraction results without LLM entity ops:
        1) Build ops via exact-match cache lookup
        2) Apply ops to entity service
        3) Upsert relationships
        4) Sync to graph
        """
        timer_total = _StepTimer()
        new_entities = result.entities or []
        new_relationships = result.relationships or []

        _jlog(
            "apply_extraction_and_sync_no_ops_start",
            request_id,
            entity_count=len(new_entities),
            relationship_count=len(new_relationships),
            has_provenance=bool(provenance),
        )

        timer_ops = _StepTimer()
        ops_data = self._build_ops_without_llm(new_entities)
        _jlog("entity_ops_skipped_llm", request_id, elapsed_sec=timer_ops.sec())

        timer_apply = _StepTimer()
        entity_idx, input2resolved, summary = self.entity_service.apply_ops(
            ops_data, provenance=provenance, request_id=request_id
        )
        _jlog(
            "apply_entity_ops_done",
            request_id,
            entity_idx_size=len(entity_idx),
            input2resolved_size=len(input2resolved),
            elapsed_sec=timer_apply.sec(),
        )

        timer_rel = _StepTimer()
        relationship_metas = self.relationship_service.upsert_from_extraction(
            result, provenance, input2resolved=input2resolved, request_id=request_id
        )
        _jlog(
            "upsert_relationships_done",
            request_id,
            relationship_count=len(relationship_metas),
            elapsed_sec=timer_rel.sec(),
        )

        timer_graph = _StepTimer()
        graph_sync_ok = True
        graph_sync_error_count = 0
        try:
            entity_count = self.graph.sync_entities(entity_idx)
            relationship_count = self.graph.sync_relationships(relationship_metas)
            _jlog(
                "neo4j_sync_done",
                request_id,
                entity_upsert_count=entity_count,
                relationship_upsert_count=relationship_count,
                elapsed_sec=timer_graph.sec(),
            )
        except Exception as e:
            graph_sync_ok = False
            graph_sync_error_count += 1
            _jlog(
                "neo4j_sync_failed",
                request_id,
                error=str(e),
                error_type=type(e).__name__,
                elapsed_sec=timer_graph.sec(),
            )

        _jlog(
            "apply_extraction_and_sync_no_ops_complete",
            request_id,
            graph_sync_ok=graph_sync_ok,
            graph_sync_error_count=graph_sync_error_count,
            total_elapsed_sec=timer_total.sec(),
        )

        return {
            "entity_idx": entity_idx,
            "entity_summary": summary,
            "relationship_metas": relationship_metas,
            "graph_sync_ok": graph_sync_ok,
            "graph_sync_error_count": graph_sync_error_count,
        }
