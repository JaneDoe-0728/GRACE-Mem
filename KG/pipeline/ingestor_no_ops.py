# pipeline/ingestor_no_ops.py
from typing import Any, Dict, List, Optional

from KG.pipeline.ingestor import Ingestor
from KG.utils.logger_config import _StepTimer, make_module_jlog
from KG.utils.common import ExtractionResult, _entity_key

_jlog = make_module_jlog(name="KG.IngestorNoEntityOps", filename="kg_ingestor.jsonl")


class IngestorNoEntityOps(Ingestor):
    """
    Ingestor variant that skips LLM generate_entity_ops.
    - Exact (name, type) matches are UPDATED; otherwise ADD.
    - No similarity-based merge or LLM-driven canonicalization.
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

    def _build_ops_without_llm(self, entities: List[Any]) -> Dict[str, Any]:
        """
        Build ops_results for EntityManager.apply_ops without LLM.
        Uses exact match in cache (name+type) to decide UPDATE vs ADD.
        """
        cache = getattr(self.entity_service, "_GLOBAL_CACHE", {}) or {}
        ent_cache = cache.get("entities", {}) or {}

        normalized = self.entity_service.normalize_entities(entities)
        results: List[Dict[str, Any]] = []

        for e in normalized:
            name = (e.get("entity_name") or "").strip()
            type_val = (e.get("entity_type") or "").strip()
            desc = (e.get("entity_description") or "").strip()
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
        entity_sim_topk: Optional[int] = None,  # kept for API compatibility; unused
        entity_sim_threshold: Optional[float] = None,  # kept for API compatibility; unused
    ) -> dict:
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

        # 1) Build entity ops without LLM
        timer_ops = _StepTimer()
        ops_data = self._build_ops_without_llm(new_entities)
        _jlog("entity_ops_skipped_llm", request_id, elapsed_sec=timer_ops.sec())

        # 2) Apply entity operations
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

        # 3) Upsert relationships
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

        # 4) Sync to graph
        timer_graph = _StepTimer()
        graph_sync_ok = True
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
            _jlog(
                "neo4j_sync_failed",
                request_id,
                error=str(e),
                error_type=type(e).__name__,
                elapsed_sec=timer_graph.sec(),
            )

        _jlog("apply_extraction_and_sync_no_ops_complete", request_id, graph_sync_ok=graph_sync_ok, total_elapsed_sec=timer_total.sec())

        return {"entity_idx": entity_idx, "relationship_metas": relationship_metas, "graph_sync_ok": graph_sync_ok}
