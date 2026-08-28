"""
Context filtering, intersection, and reranking logic.
"""
import csv
import os
import time
from typing import Any

from grace_mem.storage import build_id_to_meta_maps
from grace_mem.utils.logger_config import _StepTimer, make_module_jlog

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


class ContextFilter:
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

    def filter_by_similarity(
        self,
        entity_ids: set[str],
        relationship_ids: set[str],
        query_vec: Any,
        filter_entity_threshold: float,
        filter_relationship_threshold: float,
        filter_entity_top_k: int | None = None,
        filter_relationship_top_k: int | None = None,
        request_id: str | None = None,
    ) -> tuple[list[str], list[str]]:
        """
        Filter entities and relationships by similarity to query vector.

        Args:
            entity_ids: Set of entity IDs to filter
            relationship_ids: Set of relationship IDs to filter
            query_vec: Query embedding vector
            filter_entity_threshold: Minimum similarity for entities
            filter_relationship_threshold: Minimum similarity for relationships
            filter_entity_top_k: Maximum number of entities to keep
            filter_relationship_top_k: Maximum number of relationships to keep
            request_id: Request ID for logging

        Returns:
            (filtered_entity_ids, filtered_relationship_ids) tuple
        """
        timer_filter = _StepTimer()
        entity_id_list = list(entity_ids)
        relationship_id_list = list(relationship_ids)
        entity_name_map = {
            entity_id: name
            for entity_id, name in zip(entity_id_list, self._entity_names_from_ids(entity_id_list))
        }
        relationship_name_map = {
            relationship_id: name
            for relationship_id, name in zip(
                relationship_id_list, self._relationship_names_from_ids(relationship_id_list)
            )
        }
        _jlog(
            "similarity_filter_start",
            request_id,
            step="2.6",
            entity_candidate_count=len(entity_ids),
            relationship_candidate_count=len(relationship_ids),
            entity_candidate_names=list(entity_name_map.values())[:20],
            relationship_candidate_names=list(relationship_name_map.values())[:20],
            filter_entity_threshold=filter_entity_threshold,
            filter_relationship_threshold=filter_relationship_threshold,
            filter_entity_top_k=filter_entity_top_k,
            filter_relationship_top_k=filter_relationship_top_k,
        )

        # Filter entities
        entity_candidates: list[tuple[str, float]] = []
        for entity_id in entity_ids:
            res = self.vector_db_manager.get_entities_vdb(0).compare_by_id(
                entity_id,
                query_vec,
                threshold=filter_entity_threshold,
            )
            passed = bool(res)
            _jlog(
                "entity_compare_by_id",
                request_id,
                step="2.6",
                entity_id=entity_id,
                entity_name=entity_name_map.get(entity_id, entity_id),
                passed=passed,
                filter_entity_threshold=filter_entity_threshold,
                score=float(res[1]) if res is not None else None,
            )
            if res is not None:
                _, score = res
                entity_candidates.append((entity_id, score))

        # Sort by similarity descending, tie-break by ID for determinism
        entity_candidates.sort(key=lambda x: (-x[1], x[0]))
        entity_pass_count = len(entity_candidates)

        # Take top-k
        if filter_entity_top_k is not None and filter_entity_top_k > 0:
            entity_candidates = entity_candidates[:filter_entity_top_k]

        filtered_entities = [entity_id for entity_id, score in entity_candidates]
        _jlog(
            "entity_similarity_filter_done",
            request_id,
            step="2.6",
            passed_count=entity_pass_count,
            returned_count=len(filtered_entities),
            sample_top_entities=[
                {"name": entity_name_map.get(entity_id, entity_id), "score": score}
                for entity_id, score in entity_candidates[:10]
            ],
        )

        # Filter relationships
        relationship_candidates: list[tuple[str, float]] = []
        for relationship_id in relationship_ids:
            res = self.vector_db_manager.get_relationships_vdb(0).compare_by_id(
                relationship_id,
                query_vec,
                threshold=filter_relationship_threshold,
            )
            passed = bool(res)
            _jlog(
                "relationship_compare_by_id",
                request_id,
                step="2.6",
                relationship_id=relationship_id,
                relationship_name=relationship_name_map.get(relationship_id, relationship_id),
                passed=passed,
                filter_relationship_threshold=filter_relationship_threshold,
                score=float(res[1]) if res is not None else None,
            )
            if res is not None:
                _meta, score = res
                relationship_candidates.append((relationship_id, score))

        # Sort by similarity descending, tie-break by ID for determinism
        relationship_candidates.sort(key=lambda x: (-x[1], x[0]))
        relationship_pass_count = len(relationship_candidates)

        # Take top-k
        if filter_relationship_top_k is not None and filter_relationship_top_k > 0:
            relationship_candidates = relationship_candidates[:filter_relationship_top_k]

        filtered_relationships = [relationship_id for relationship_id, score in relationship_candidates]
        _jlog(
            "relationship_similarity_filter_done",
            request_id,
            step="2.6",
            passed_count=relationship_pass_count,
            returned_count=len(filtered_relationships),
            sample_top_relationships=[
                {"name": relationship_name_map.get(relationship_id, relationship_id), "score": score}
                for relationship_id, score in relationship_candidates[:10]
            ],
        )

        _jlog(
            "intersection_filtered",
            request_id,
            step="2.6",
            filtered_entity_count=len(filtered_entities),
            filtered_relationship_count=len(filtered_relationships),
            elapsed_sec=timer_filter.sec(),
        )

        return filtered_entities, filtered_relationships

    def compute_cosine_scores(
        self,
        entity_ids: set[str] | list[str],
        query_vec: Any,
    ) -> dict[str, float]:
        """Return raw cosine scores for all entity_ids (no threshold)."""
        entity_vdb = self.vector_db_manager.get_entities_vdb(0)
        scores: dict[str, float] = {}
        for eid in entity_ids:
            s = entity_vdb.compare_by_id_raw(eid, query_vec)
            if s is not None:
                scores[eid] = s
        return scores

    def filter_by_rrf(
        self,
        entity_ids: set[str],
        relationship_ids: set[str],
        query_vec: Any,
        rrf_k: float,
        filter_entity_top_k: int,
        filter_relationship_top_k: int,
        entity_emb_scores: dict[str, float],
        entity_bm25_scores: dict[str, float],
        rel_endpoint_scores: dict[str, float],
        rel_emb_scores: dict[str, float],
        node_subgraph_rel_ids: set[str],
        filter_method: str = "rrf",
        request_id: str | None = None,
    ) -> tuple[list[str], list[str], dict[str, float]]:
        """
        Filter entities and relationships using Reciprocal Rank Fusion over multiple signals.

        Signals for entities:
          L1 fresh cosine similarity (all union candidates)
          L2 entity-embedding search rank (Src1 seeds)
          L3 entity-BM25 search rank (Src2 seeds)
          L4 relation-endpoint rank (Src3 endpoint entities)

        Signals for relations:
          L1 fresh cosine similarity
          L2 relation-embedding search rank (Src3)
          L3 node-subgraph presence (binary, rank=1)

        Returns:
            (filtered_entity_ids, filtered_relationship_ids, entity_rrf_scores)
        """
        timer = _StepTimer()

        def _scores_to_ranks(scores: dict[str, float]) -> dict[str, int]:
            sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
            return {eid: i + 1 for i, eid in enumerate(sorted_ids)}

        entity_vdb = self.vector_db_manager.get_entities_vdb(0) if filter_method == "similarity" else None
        rel_vdb = self.vector_db_manager.get_relationships_vdb(0)

        entity_name_map = {
            eid: name
            for eid, name in zip(list(entity_ids), self._entity_names_from_ids(list(entity_ids)))
        }
        relationship_name_map = {
            rid: name
            for rid, name in zip(list(relationship_ids), self._relationship_names_from_ids(list(relationship_ids)))
        }

        _jlog(
            "rrf_filter_start",
            request_id,
            step="2.6",
            entity_candidate_count=len(entity_ids),
            relationship_candidate_count=len(relationship_ids),
            rrf_k=rrf_k,
        )

        # ── Entity RRF ─────────────────────────────────────────────────────────
        if filter_method == "similarity" and entity_vdb is not None:
            cosine_scores: dict[str, float] = {}
            for eid in entity_ids:
                s = entity_vdb.compare_by_id_raw(eid, query_vec)
                if s is not None:
                    cosine_scores[eid] = s
            cosine_ranks: dict[str, int] = _scores_to_ranks(cosine_scores)
        else:
            cosine_ranks = {}

        emb_ranks = _scores_to_ranks(entity_emb_scores)
        bm25_ranks = _scores_to_ranks(entity_bm25_scores)
        endpoint_ranks = _scores_to_ranks(rel_endpoint_scores)
        entity_rank_dicts = [d for d in [cosine_ranks, emb_ranks, bm25_ranks, endpoint_ranks] if d]

        entity_rrf_scores: dict[str, float] = {}
        for eid in entity_ids:
            active = [rd for rd in entity_rank_dicts if eid in rd]
            if active:
                entity_rrf_scores[eid] = sum(
                    1.0 / (rrf_k + rd[eid]) for rd in active
                ) / len(active)

        sorted_entities = sorted(entity_rrf_scores, key=lambda x: (-entity_rrf_scores[x], x))
        if filter_entity_top_k is not None and filter_entity_top_k > 0:
            sorted_entities = sorted_entities[:filter_entity_top_k]

        _jlog(
            "rrf_entity_filter_done",
            request_id,
            step="2.6",
            candidate_count=len(entity_ids),
            selected_count=len(sorted_entities),
            top_entities=[
                {
                    "entity_id": eid,
                    "entity_name": entity_name_map.get(eid, eid),
                    "rrf_score": round(entity_rrf_scores[eid], 6),
                    "cosine_rank": cosine_ranks.get(eid),
                    "emb_rank": emb_ranks.get(eid),
                    "bm25_rank": bm25_ranks.get(eid),
                    "endpoint_rank": endpoint_ranks.get(eid),
                }
                for eid in sorted_entities[:10]
            ],
        )

        # ── Relation RRF ───────────────────────────────────────────────────────
        cosine_scores_rel: dict[str, float] = {}
        for rid in relationship_ids:
            s = rel_vdb.compare_by_id_raw(rid, query_vec)
            if s is not None:
                cosine_scores_rel[rid] = s

        cosine_ranks_rel = _scores_to_ranks(cosine_scores_rel)
        emb_ranks_rel = _scores_to_ranks(rel_emb_scores)
        node_presence_ranks: dict[str, int] = {
            rid: 1 for rid in node_subgraph_rel_ids if rid in relationship_ids
        }
        rel_rank_dicts = [cosine_ranks_rel, emb_ranks_rel, node_presence_ranks]

        rel_rrf_scores: dict[str, float] = {}
        for rid in relationship_ids:
            rel_rrf_scores[rid] = sum(
                1.0 / (rrf_k + rd[rid]) for rd in rel_rank_dicts if rid in rd
            )

        sorted_rels = sorted(rel_rrf_scores, key=lambda x: (-rel_rrf_scores[x], x))
        if filter_relationship_top_k is not None and filter_relationship_top_k > 0:
            sorted_rels = sorted_rels[:filter_relationship_top_k]

        _jlog(
            "rrf_filter_done",
            request_id,
            step="2.6",
            entity_selected_count=len(sorted_entities),
            rel_selected_count=len(sorted_rels),
            top_relationships=[
                {
                    "relationship_id": rid,
                    "relationship_name": relationship_name_map.get(rid, rid),
                    "rrf_score": round(rel_rrf_scores[rid], 6),
                    "cosine_rank": cosine_ranks_rel.get(rid),
                    "emb_rank": emb_ranks_rel.get(rid),
                    "node_subgraph_present": rid in node_presence_ranks,
                }
                for rid in sorted_rels[:10]
            ],
            elapsed_sec=timer.sec(),
        )

        return sorted_entities, sorted_rels, entity_rrf_scores

    def rerank_and_recover(
        self,
        question: str,
        all_entity_ids: set[str],
        all_relationship_ids: set[str],
        filtered_entity_ids: set[str],
        filtered_relationship_ids: set[str],
        reranker_threshold: float,
        reranker_top_k: int,
        request_id: str | None = None,
    ) -> tuple[set[str], set[str]]:
        """
        Use reranker to recover filtered-out entities and relationships.

        Args:
            question: Query text
            all_entity_ids: All entity IDs before filtering
            all_relationship_ids: All relationship IDs before filtering
            filtered_entity_ids: Entity IDs after filtering
            filtered_relationship_ids: Relationship IDs after filtering
            reranker_threshold: Minimum reranker score to recover item
            reranker_top_k: Maximum items to recover per type
            request_id: Request ID for logging

        Returns:
            (updated_entity_ids, updated_relationship_ids) tuple
        """
        from grace_mem.utils.reranker import get_reranker

        timer_rerank_total = _StepTimer()
        _jlog(
            "reranker_start",
            request_id,
            step="2.7",
            all_entity_count=len(all_entity_ids),
            all_relationship_count=len(all_relationship_ids),
            filtered_entity_count=len(filtered_entity_ids),
            filtered_relationship_count=len(filtered_relationship_ids),
            reranker_threshold=reranker_threshold,
            reranker_top_k=reranker_top_k,
        )

        # Load reranker model
        timer_load = _StepTimer()
        reranker = get_reranker()
        _jlog(
            "reranker_model_loaded",
            request_id,
            step="2.7",
            elapsed_sec=timer_load.sec(),
        )

        # Collect filtered-out entities
        filtered_out_entity_ids = all_entity_ids - filtered_entity_ids
        _jlog(
            "reranker_start_entities",
            request_id,
            step="2.7",
            filtered_out_count=len(filtered_out_entity_ids),
            filtered_out_names=self._entity_names_from_ids(list(filtered_out_entity_ids))[:20],
            reranker_threshold=reranker_threshold,
            reranker_top_k=reranker_top_k,
        )

        timer_entity_rerank = _StepTimer()
        new_entity_ids = set(filtered_entity_ids)

        if filtered_out_entity_ids:
            # Build entity texts for reranking
            entity_id2meta, _ = build_id_to_meta_maps(self.cache)
            filtered_out_entity_texts = []
            filtered_out_entity_ids_list = []

            for entity_id in filtered_out_entity_ids:
                meta = entity_id2meta.get(entity_id, {})
                if not meta:
                    continue
                name = meta.get("name", "").strip()
                type_val = meta.get("type", "").strip()
                description = meta.get("description", "").strip()
                # Match EntityManager._build_entity_repr format
                if name:
                    text = f"{name} [type={type_val}] {description}".strip()
                else:
                    text = f"{type_val} {description}".strip()
                filtered_out_entity_texts.append(text)
                filtered_out_entity_ids_list.append(entity_id)

            # Rerank filtered-out entities
            if filtered_out_entity_texts:
                rerank_results = reranker.rank_pairs(
                    query=question,
                    texts=filtered_out_entity_texts,
                    threshold=reranker_threshold,
                    doc_type="entity",
                )

                # Recover top-k items
                recovered_entity_count = 0
                for idx, score in rerank_results[:reranker_top_k]:
                    recovered_entity_id = filtered_out_entity_ids_list[idx]
                    new_entity_ids.add(recovered_entity_id)
                    recovered_entity_count += 1

                _jlog(
                    "reranker_entities_done",
                    request_id,
                    step="2.7",
                    recovered_count=recovered_entity_count,
                    sample_recovered=[
                        {
                            "name": entity_id2meta.get(filtered_out_entity_ids_list[idx], {}).get("name", filtered_out_entity_ids_list[idx]),
                            "score": score,
                        }
                        for idx, score in rerank_results[:reranker_top_k]
                    ],
                    elapsed_sec=timer_entity_rerank.sec(),
                )
            else:
                _jlog(
                    "reranker_entities_skipped",
                    request_id,
                    step="2.7",
                    reason="no_entity_texts",
                )
        else:
            _jlog(
                "reranker_entities_skipped",
                request_id,
                step="2.7",
                reason="no_filtered_out_entities",
            )

        # Collect filtered-out relationships
        timer_relationship_rerank = _StepTimer()
        filtered_out_relationship_ids = all_relationship_ids - filtered_relationship_ids
        new_relationship_ids = set(filtered_relationship_ids)
        _jlog(
            "reranker_start_relationships",
            request_id,
            step="2.7",
            filtered_out_count=len(filtered_out_relationship_ids),
            filtered_out_names=self._relationship_names_from_ids(list(filtered_out_relationship_ids))[:20],
            reranker_threshold=reranker_threshold,
            reranker_top_k=reranker_top_k,
        )

        if filtered_out_relationship_ids:
            # Build relationship texts for reranking
            _, relationship_id2meta = build_id_to_meta_maps(self.cache)
            filtered_out_relationship_texts = []
            filtered_out_relationship_ids_list = []

            for relationship_id in filtered_out_relationship_ids:
                meta = relationship_id2meta.get(relationship_id, {})
                if not meta:
                    continue
                source_entity = meta.get("source_entity", "")
                target_entity = meta.get("target_entity", "")
                description = meta.get("description", "")
                keywords = meta.get("keywords", "")
                # Match RelationshipManager format
                text = f"{source_entity} -> {target_entity} | {description} (keywords: {keywords})"
                filtered_out_relationship_texts.append(text)
                filtered_out_relationship_ids_list.append(relationship_id)

            # Rerank filtered-out relationships
            if filtered_out_relationship_texts:
                rerank_results = reranker.rank_pairs(
                    query=question,
                    texts=filtered_out_relationship_texts,
                    threshold=reranker_threshold,
                    doc_type="relationship",
                )

                # Recover top-k items
                recovered_relationship_count = 0
                for idx, score in rerank_results[:reranker_top_k]:
                    recovered_relationship_id = filtered_out_relationship_ids_list[idx]
                    new_relationship_ids.add(recovered_relationship_id)
                    recovered_relationship_count += 1

                _jlog(
                    "reranker_relationships_done",
                    request_id,
                    step="2.7",
                    recovered_count=recovered_relationship_count,
                    sample_recovered=[
                        {
                            "name": self._relationship_names_from_ids([filtered_out_relationship_ids_list[idx]])[0],
                            "score": score,
                        }
                        for idx, score in rerank_results[:reranker_top_k]
                    ],
                    elapsed_sec=timer_relationship_rerank.sec(),
                )
            else:
                _jlog(
                    "reranker_relationships_skipped",
                    request_id,
                    step="2.7",
                    reason="no_relationship_texts",
                )
        else:
            _jlog(
                "reranker_relationships_skipped",
                request_id,
                step="2.7",
                reason="no_filtered_out_relationships",
            )

        _jlog(
            "reranker_complete",
            request_id,
            step="2.7",
            final_entity_count=len(new_entity_ids),
            final_relationship_count=len(new_relationship_ids),
            final_entity_names=self._entity_names_from_ids(list(new_entity_ids))[:20],
            final_relationship_names=self._relationship_names_from_ids(list(new_relationship_ids))[:20],
            total_elapsed_sec=timer_rerank_total.sec(),
        )

        return new_entity_ids, new_relationship_ids

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
        from grace_mem.utils.reranker import get_reranker

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
