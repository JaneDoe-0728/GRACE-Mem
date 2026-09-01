"""ExtractionSyncer: VDB writes + FalkorDB sync."""
from typing import Any

from grace_mem.domain.extraction import ExtractionResult
from grace_mem.runtime.logger_config import _StepTimer, make_module_jlog

_jlog = make_module_jlog(name="grace_mem.Ingestor", filename="kg_ingestor.jsonl")


class ExtractionSyncer:
    """Apply entity ops, upsert relationships, and sync to FalkorDB."""

    def __init__(self, *, llm: Any, graph: Any, entity_service: Any, relationship_service: Any, cfg: Any) -> None:
        """Store the collaborators used to apply extraction results and sync storage."""
        self._llm = llm
        self._graph = graph
        self._entity_service = entity_service
        self._relationship_service = relationship_service
        self._cfg = cfg

    def sync(
        self,
        result: ExtractionResult,
        provenance: dict | None,
        request_id: str,
        *,
        entity_sim_topk: int | None = None,
        entity_sim_threshold: float | None = None,
    ) -> dict:
        """
        1) Find similar entities → 2) Generate entity ops → 3) Apply → 4) Upsert rels → 5) Sync graph.
        Returns dict with entity_idx, relationship_metas, graph_sync_ok.
        """
        entity_sim_topk = entity_sim_topk if entity_sim_topk is not None else self._cfg.similar_entity_top_k
        entity_sim_threshold = (
            entity_sim_threshold if entity_sim_threshold is not None else self._cfg.entity_sim_threshold
        )

        timer_total = _StepTimer()
        new_entities = result.entities or []
        new_relationships = result.relationships or []

        _jlog(
            "apply_extraction_and_sync_start",
            request_id,
            entity_count=len(new_entities),
            relationship_count=len(new_relationships),
            has_provenance=bool(provenance),
            entity_sim_topk=entity_sim_topk,
            entity_sim_threshold=entity_sim_threshold,
        )

        # 1) Find similar entities
        timer_similar = _StepTimer()
        similar_map = self._entity_service.find_similar_for_hybrid(
            new_entities, top_k=entity_sim_topk, threshold=entity_sim_threshold,
        )
        _jlog("find_similar_entities_done", request_id, similar_map_size=len(similar_map), elapsed_sec=timer_similar.sec())

        # 2) Generate entity operations
        timer_ops = _StepTimer()
        ops_data = self._llm.generate_entity_ops(
            new_entities=[
                {
                    "entity_name": e.entity_name,
                    "entity_type": e.entity_type,
                    "entity_description": e.entity_description,
                    "entity_metadata": getattr(e, "entity_metadata", None),
                }
                for e in new_entities
            ],
            similar_map=similar_map,
        )
        _jlog("generate_entity_ops_done", request_id, elapsed_sec=timer_ops.sec())

        # 3) Apply entity operations
        timer_apply = _StepTimer()
        entity_idx, input2resolved, summary = self._entity_service.apply_ops(
            ops_data, provenance=provenance, request_id=request_id
        )
        _jlog(
            "apply_entity_ops_done",
            request_id,
            entity_idx_size=len(entity_idx),
            input2resolved_size=len(input2resolved),
            elapsed_sec=timer_apply.sec(),
        )

        # 4) Upsert relationships
        timer_rel = _StepTimer()
        relationship_metas = self._relationship_service.upsert_from_extraction(
            result, provenance, input2resolved=input2resolved, request_id=request_id
        )
        _jlog("upsert_relationships_done", request_id, relationship_count=len(relationship_metas), elapsed_sec=timer_rel.sec())

        # 5) Sync to FalkorDB
        timer_neo4j = _StepTimer()
        graph_sync_ok = True
        graph_sync_error_count = 0
        try:
            entity_count = self._graph.sync_entities(entity_idx)
            relationship_count = self._graph.sync_relationships(relationship_metas)
            _jlog(
                "neo4j_sync_done",
                request_id,
                entity_upsert_count=entity_count,
                relationship_upsert_count=relationship_count,
                elapsed_sec=timer_neo4j.sec(),
            )
        except Exception as e:
            graph_sync_ok = False
            graph_sync_error_count += 1
            _jlog("neo4j_sync_failed", request_id, error=str(e), error_type=type(e).__name__, elapsed_sec=timer_neo4j.sec())

        # 5b) Verify FalkorDB contains the synced IDs
        graph_missing_entity_count = 0
        graph_missing_rel_count = 0
        if graph_sync_ok and hasattr(self._graph, "check_entity_ids"):
            timer_verify = _StepTimer()
            try:
                # ``entity_idx`` is keyed by the manager's normalized lookup
                # key (for example ``user::person``), while FalkorDB stores the
                # stable id in each metadata value (``person_user``).  Verify
                # the ids that were actually passed to ``sync_entities``.
                expected_ent_ids = [
                    str(meta["id"])
                    for meta in entity_idx.values()
                    if meta and meta.get("id")
                ]
                expected_rel_ids = [m["id"] for m in relationship_metas if m.get("id")]

                found_ent = set(self._graph.check_entity_ids(expected_ent_ids))
                found_rel = set(self._graph.check_relationship_ids(expected_rel_ids))

                missing_ents = [i for i in expected_ent_ids if i not in found_ent]
                missing_rels = [i for i in expected_rel_ids if i not in found_rel]
                graph_missing_entity_count = len(missing_ents)
                graph_missing_rel_count = len(missing_rels)

                _jlog(
                    "falkordb_sync_verify_done",
                    request_id,
                    expected_entity_count=len(expected_ent_ids),
                    graph_matched_entity_count=len(found_ent),
                    missing_entity_count=graph_missing_entity_count,
                    missing_entity_sample=missing_ents[:5],
                    expected_relationship_count=len(expected_rel_ids),
                    graph_matched_relationship_count=len(found_rel),
                    missing_relationship_count=graph_missing_rel_count,
                    missing_relationship_sample=missing_rels[:5],
                    elapsed_sec=timer_verify.sec(),
                )

                if missing_ents or missing_rels:
                    graph_sync_ok = False
            except Exception as ve:
                graph_sync_ok = False
                graph_sync_error_count += 1
                _jlog(
                    "falkordb_sync_verify_failed",
                    request_id,
                    error=str(ve),
                    error_type=type(ve).__name__,
                    elapsed_sec=timer_verify.sec(),
                )

        _jlog(
            "apply_extraction_and_sync_complete",
            request_id,
            graph_sync_ok=graph_sync_ok,
            graph_sync_missing_entity_count=graph_missing_entity_count,
            graph_sync_missing_relationship_count=graph_missing_rel_count,
            graph_sync_error_count=graph_sync_error_count,
            total_elapsed_sec=timer_total.sec(),
        )
        return {
            "entity_idx": entity_idx,
            "entity_summary": summary,
            "relationship_metas": relationship_metas,
            "graph_sync_ok": graph_sync_ok,
            "graph_sync_missing_entity_count": graph_missing_entity_count,
            "graph_sync_missing_relationship_count": graph_missing_rel_count,
            "graph_sync_error_count": graph_sync_error_count,
        }
