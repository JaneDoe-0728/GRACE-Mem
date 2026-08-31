"""
Context filtering, intersection, and reranking logic.
"""
import csv
import os
import time
from typing import Any

from grace_mem.adapters.cache.cache import build_id_to_meta_maps
from grace_mem.runtime.logger_config import _StepTimer, make_module_jlog

_jlog = make_module_jlog(name="grace_mem.Retrieval.Filtering", filename="kg_retrieval_filtering.jsonl")

_RERANKER_SCORE_CSV = os.path.join("logs", "reranker_scores.csv")
_RERANKER_CSV_HEADER = [
    "ts_ms", "request_id",
    "item_type", "item_id", "name",
    "score", "rank",
    "above_threshold", "selected",
    "threshold", "top_k",
    "drop_reason",  # "": selected | "topk_cutoff": passed threshold but rank > top_k | "below_threshold": score < threshold
]


def _append_reranker_scores(
    rows: list[dict],
) -> None:
    """Append scored candidate rows to logs/reranker_scores.csv."""
    os.makedirs("logs", exist_ok=True)
    write_header = not os.path.exists(_RERANKER_SCORE_CSV)
    with open(_RERANKER_SCORE_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RERANKER_CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


class EvidenceFilter:
    """
    Handles filtering and reranking of context entities and relationships.
    """

    def __init__(self, vector_db_manager: Any, cache: dict[str, Any]) -> None:
        """
        Args:
            vector_db_manager: Manager for accessing vector databases
            cache: Global cache with entity/relationship metadata
        """
        self.vector_db_manager = vector_db_manager
        self.cache = cache

    def _entity_names_from_ids(self, entity_ids: set[str] | list[str]) -> list[str]:
        """Resolve entity IDs into readable names."""
        ent_id2meta, _ = build_id_to_meta_maps(self.cache)
        names: list[str] = []
        seen: set[str] = set()
        for entity_id in entity_ids:
            meta = ent_id2meta.get(entity_id, {}) or {}
            name = meta.get("name") or entity_id
            if name not in seen:
                names.append(name)
                seen.add(name)
        return names

    def _relationship_names_from_ids(self, relationship_ids: set[str] | list[str]) -> list[str]:
        """Resolve relationship IDs into readable source->target labels."""
        ent_id2meta, rel_id2meta = build_id_to_meta_maps(self.cache)
        labels: list[str] = []
        seen: set[str] = set()
        for relationship_id in relationship_ids:
            meta = rel_id2meta.get(relationship_id, {}) or {}
            if not meta:
                label = relationship_id
            else:
                source_id = meta.get("source_id")
                target_id = meta.get("target_id")
                source_meta = ent_id2meta.get(source_id, {}) if isinstance(source_id, str) else {}
                target_meta = ent_id2meta.get(target_id, {}) if isinstance(target_id, str) else {}
                src_name = (
                    meta.get("source_entity")
                    or source_meta.get("name")
                    or source_id
                    or "?"
                )
                tgt_name = (
                    meta.get("target_entity")
                    or target_meta.get("name")
                    or target_id
                    or "?"
                )
                desc = (meta.get("description") or "").strip()
                label = f"{src_name} -> {tgt_name}" if not desc else f"{src_name} -> {tgt_name} | {desc}"
            if label not in seen:
                labels.append(label)
                seen.add(label)
        return labels

    def compute_subgraph_intersection(
        self,
        node_subgraph: dict,
        edge_subgraph: list,
        use_union: bool = True,
        request_id: str | None = None,
    ) -> tuple[set[str], set[str]]:
        """
        Compute intersection (or union) of entity/relationship IDs from local and global subgraphs.

        Args:
            node_subgraph: Node-based subgraph from entity search
            edge_subgraph: Edge-based subgraph from relationship search
            use_union: If True, use union; if False, use intersection
            request_id: Request ID for logging

        Returns:
            (entity_ids, relationship_ids) tuple
        """
        _jlog(
            "subgraph_merge_start",
            request_id,
            step="2.5",
            node_subgraph_nodes=len(node_subgraph or {}),
            edge_subgraph_edges=len(edge_subgraph or []),
            operation="union" if use_union or not edge_subgraph else "intersection",
        )

        # Collect local (entity-based) subgraph IDs
        local_entity_set = set(node_subgraph.keys()) | {
            nb["neighbor_id"]
            for b in node_subgraph.values()
            for nb in (b.get("neighbors") or [])
        }
        local_rel_set = {
            nb["rel_id"]
            for b in node_subgraph.values()
            for nb in (b.get("neighbors") or [])
            if "rel_id" in nb
        }

        # Collect global (relationship-based) subgraph IDs
        global_entity_set = (
            ({e["source_id"] for e in edge_subgraph} | {e["target_id"] for e in edge_subgraph})
            if edge_subgraph
            else set()
        )
        global_rel_set = ({e["rel_id"] for e in edge_subgraph}) if edge_subgraph else set()

        # Intersection or union
        if use_union or not edge_subgraph:
            intersect_entity_ids = local_entity_set | global_entity_set
            intersect_rel_ids = local_rel_set | global_rel_set
        else:
            intersect_entity_ids = local_entity_set & global_entity_set
            intersect_rel_ids = local_rel_set & global_rel_set

        _jlog(
            "intersection_done",
            request_id,
            step="2.5",
            local_entities=len(local_entity_set),
            local_rels=len(local_rel_set),
            global_entities=len(global_entity_set),
            global_rels=len(global_rel_set),
            intersect_entities=len(intersect_entity_ids),
            intersect_rels=len(intersect_rel_ids),
            use_union=use_union,
            operation="union" if use_union or not edge_subgraph else "intersection",
            sample_entity_ids=sorted(intersect_entity_ids)[:20],
            sample_relationship_ids=sorted(intersect_rel_ids)[:20],
            sample_entity_names=self._entity_names_from_ids(sorted(intersect_entity_ids)[:20]),
            sample_relationship_names=self._relationship_names_from_ids(sorted(intersect_rel_ids)[:20]),
        )

        return intersect_entity_ids, intersect_rel_ids





    def rerank_filter(
        self,
        question: str,
        entity_ids: set[str],
        relationship_ids: set[str],
        entity_top_k: int,
        relationship_top_k: int,
        threshold: float,
        request_id: str | None = None,
    ) -> tuple[list[str], list[str]]:
        """Primary reranker filter: score all candidates, keep top-K above threshold.

        Returns ordered lists (reranker rank order, deduplicated) rather than
        sets so the final Retrieved_Context ordering is deterministic across runs.
        """
        from grace_mem.retrieval.reranker import get_reranker

        timer_total = _StepTimer()
        _jlog(
            "rerank_filter_start",
            request_id,
            step="2.7",
            entity_count=len(entity_ids),
            relationship_count=len(relationship_ids),
            entity_top_k=entity_top_k,
            relationship_top_k=relationship_top_k,
            threshold=threshold,
        )

        reranker = get_reranker()
        entity_id2meta, relationship_id2meta = build_id_to_meta_maps(self.cache)
        ts_ms = int(time.time() * 1000)
        csv_rows: list[dict] = []

        # Score all entities
        entity_texts = []
        entity_ids_list = []
        for entity_id in entity_ids:
            meta = entity_id2meta.get(entity_id, {})
            if not meta:
                continue
            name = meta.get("name", "").strip()
            type_val = meta.get("type", "").strip()
            description = meta.get("description", "").strip()
            text = f"{name} [type={type_val}] {description}".strip() if name else f"{type_val} {description}".strip()
            entity_texts.append(text)
            entity_ids_list.append(entity_id)

        # Ordered list preserving reranker rank order; a set tracks membership
        # for dedup so the final context ordering is deterministic across runs.
        selected_entity_ids: list[str] = []
        _seen_entity_ids: set[str] = set()
        if entity_texts:
            # Fetch all scores without threshold so every candidate is logged.
            all_entity_results = reranker.rank_pairs(query=question, texts=entity_texts, threshold=None, doc_type="entity")
            # Apply threshold then top-K to determine selection.
            passing = [(i, s) for i, s in all_entity_results if s >= threshold]
            for idx, score in passing[:entity_top_k]:
                eid = entity_ids_list[idx]
                if eid not in _seen_entity_ids:
                    _seen_entity_ids.add(eid)
                    selected_entity_ids.append(eid)
            _jlog(
                "rerank_filter_entities_done",
                request_id,
                step="2.7",
                selected_count=len(selected_entity_ids),
                sample_selected=[
                    {"name": entity_id2meta.get(entity_ids_list[idx], {}).get("name", entity_ids_list[idx]), "score": score}
                    for idx, score in passing[:entity_top_k]
                ],
            )
            selected_top_k_ids = {entity_ids_list[i] for i, _ in passing[:entity_top_k]}
            for rank, (idx, score) in enumerate(all_entity_results, start=1):
                eid = entity_ids_list[idx]
                meta = entity_id2meta.get(eid, {})
                above = score >= threshold
                sel = eid in selected_top_k_ids
                if sel:
                    drop_reason = ""
                elif above:
                    drop_reason = "topk_cutoff"
                else:
                    drop_reason = "below_threshold"
                csv_rows.append({
                    "ts_ms": ts_ms,
                    "request_id": request_id,
                    "item_type": "entity",
                    "item_id": eid,
                    "name": meta.get("name", eid),
                    "score": round(score, 6),
                    "rank": rank,
                    "above_threshold": above,
                    "selected": sel,
                    "threshold": threshold,
                    "top_k": entity_top_k,
                    "drop_reason": drop_reason,
                })

        # Score all relationships
        relationship_texts = []
        relationship_ids_list = []
        for relationship_id in relationship_ids:
            meta = relationship_id2meta.get(relationship_id, {})
            if not meta:
                continue
            source_entity = meta.get("source_entity", "")
            target_entity = meta.get("target_entity", "")
            description = meta.get("description", "")
            keywords = meta.get("keywords", "")
            text = f"{source_entity} -> {target_entity} | {description} (keywords: {keywords})"
            relationship_texts.append(text)
            relationship_ids_list.append(relationship_id)

        selected_relationship_ids: list[str] = []
        _seen_relationship_ids: set[str] = set()
        if relationship_texts:
            all_rel_results = reranker.rank_pairs(query=question, texts=relationship_texts, threshold=None, doc_type="relationship")
            passing = [(i, s) for i, s in all_rel_results if s >= threshold]
            for idx, score in passing[:relationship_top_k]:
                rid = relationship_ids_list[idx]
                if rid not in _seen_relationship_ids:
                    _seen_relationship_ids.add(rid)
                    selected_relationship_ids.append(rid)
            _jlog(
                "rerank_filter_relationships_done",
                request_id,
                step="2.7",
                selected_count=len(selected_relationship_ids),
                sample_selected=[
                    {"name": self._relationship_names_from_ids([relationship_ids_list[idx]])[0], "score": score}
                    for idx, score in passing[:relationship_top_k]
                ],
            )
            selected_top_k_rel_ids = {relationship_ids_list[i] for i, _ in passing[:relationship_top_k]}
            for rank, (idx, score) in enumerate(all_rel_results, start=1):
                rid = relationship_ids_list[idx]
                meta = relationship_id2meta.get(rid, {})
                name = f"{meta.get('source_entity', '')} -> {meta.get('target_entity', '')}"
                above = score >= threshold
                sel = rid in selected_top_k_rel_ids
                if sel:
                    drop_reason = ""
                elif above:
                    drop_reason = "topk_cutoff"
                else:
                    drop_reason = "below_threshold"
                csv_rows.append({
                    "ts_ms": ts_ms,
                    "request_id": request_id,
                    "item_type": "relationship",
                    "item_id": rid,
                    "name": name,
                    "score": round(score, 6),
                    "rank": rank,
                    "above_threshold": above,
                    "selected": sel,
                    "threshold": threshold,
                    "top_k": relationship_top_k,
                    "drop_reason": drop_reason,
                })

        if csv_rows:
            _append_reranker_scores(csv_rows)

        _jlog(
            "rerank_filter_complete",
            request_id,
            step="2.7",
            final_entity_count=len(selected_entity_ids),
            final_relationship_count=len(selected_relationship_ids),
            total_elapsed_sec=timer_total.sec(),
        )

        return selected_entity_ids, selected_relationship_ids
