# pipeline/retrieval_steps/search.py
"""
Entity and relationship search functionality using hybrid vector + BM25 approach.
"""
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import faiss

from KG.utils.common import tokenize_en
from KG.utils.logger_config import _StepTimer, make_module_jlog

_jlog = make_module_jlog(name="KG.Retrieval.Search", filename="kg_retrieval_search.jsonl")


def _relationship_label(meta: Dict[str, Any]) -> str:
    """Render relationship metadata into a human-readable source->target label."""
    src_name = meta.get("source_entity") or meta.get("source_name") or meta.get("source_id") or "?"
    tgt_name = meta.get("target_entity") or meta.get("target_name") or meta.get("target_id") or "?"
    desc = (meta.get("description") or meta.get("name") or "").strip()
    return f"{src_name} -> {tgt_name}" if not desc else f"{src_name} -> {tgt_name} | {desc}"


class EntityRelationshipSearcher:
    """
    Handles hybrid search for entities and relationships using vector similarity and BM25.
    """

    def __init__(self, vector_db_manager: Any, embed_function: Any) -> None:
        """
        Args:
            vector_db_manager: Manager for accessing vector databases
            embed_function: Function to embed text into vectors
        """
        """Store the vector manager and embedding function used by retrieval."""
        self.vector_db_manager = vector_db_manager
        self.embed = embed_function

    def search_entities_hybrid(
        self,
        *,
        query_vec: np.ndarray,
        low_level_keywords: Iterable[str],
        entity_vec_threshold: float,
        entity_top_k: int,
        request_id: Optional[str] = None,
    ) -> Dict[str, List[Tuple[Dict[str, Any], float]]]:
        """
        Search entities using both vector similarity and BM25 keyword matching.

        Returns dict mapping:
        - '__vector__' → [(meta, sim_score), ...] from vector search
        - '<keyword>' → [(meta, bm25_score), ...] from BM25 search

        Args:
            query_vec: Query embedding vector
            low_level_keywords: Keywords for BM25 search
            entity_vec_threshold: Minimum cosine similarity threshold
            entity_top_k: Maximum number of results from vector search
            request_id: Request ID for logging

        Returns:
            Dictionary of search results by source (vector or keyword)
        """
        timer_total = _StepTimer()
        keywords = [str(keyword) for keyword in (low_level_keywords or []) if keyword]
        entities_hit: Dict[str, List[Tuple[Dict[str, Any], float]]] = {}

        _jlog(
            "search_entities_hybrid_start",
            request_id,
            step="2.1",
            keyword_count=len(keywords),
            low_level_keywords=keywords,
            entity_vec_threshold=entity_vec_threshold,
            entity_top_k=entity_top_k,
        )

        # 1) Vector search
        timer_vec = _StepTimer()
        vdb = self.vector_db_manager.get_entities_vdb(query_vec.shape[0])
        _jlog(
            "vector_search_start",
            request_id,
            step="2.1",
            threshold=entity_vec_threshold,
            entity_top_k=entity_top_k,
        )
        vec_hits = vdb.search(query_vec, top_k=entity_top_k, threshold=entity_vec_threshold)
        entities_hit["__vector__"] = [(meta, float(sim)) for meta, sim in vec_hits if meta and sim is not None]

        _jlog(
            "vector_search_done",
            request_id,
            step="2.1",
            threshold=entity_vec_threshold,
            entity_top_k=entity_top_k,
            hit_count=len(entities_hit["__vector__"]),
            sample_hits=[
                {"id": m.get("id"), "name": m.get("name"), "sim": s}
                for m, s in entities_hit["__vector__"][:10]
            ],
            elapsed_sec=timer_vec.sec(),
        )

        # Ablation F:關閉 BM25 分支 — 實體 seeds 只留向量路徑。
        # 環境變數慣例比照 KG_ABLATION_NO_GRAPH,預設關。
        if os.getenv("KG_ABLATION_NO_BM25", "0").lower() not in ("0", "", "false"):
            _jlog("bm25_skipped_ablation", request_id, step="2.1")
            _jlog(
                "search_entities_hybrid_complete",
                request_id,
                step="2.1",
                vector_hit_count=len(entities_hit.get("__vector__", [])),
                bm25_keyword_count=0,
                bm25_total_hit_pairs=0,
                unique_entity_count=len(
                    {
                        meta.get("id")
                        for hits in entities_hit.values()
                        for meta, _ in hits
                        if meta
                    }
                ),
                elapsed_sec=timer_total.sec(),
            )
            return entities_hit

        # 2) BM25 search for each keyword
        bm25 = self.vector_db_manager.get_entities_bm25(load_if_empty=True)
        metas = bm25.metas
        if not metas:
            _jlog("bm25_empty_index", request_id, step="2.1")
            _jlog(
                "search_entities_hybrid_complete",
                request_id,
                step="2.1",
                vector_hit_count=len(entities_hit.get("__vector__", [])),
                bm25_keyword_count=0,
                bm25_total_hit_pairs=0,
                unique_entity_count=len(
                    {
                        meta.get("id")
                        for hits in entities_hit.values()
                        for meta, _ in hits
                        if meta
                    }
                ),
                elapsed_sec=timer_total.sec(),
            )
            return entities_hit

        _jlog(
            "bm25_search_start",
            request_id,
            step="2.1",
            keyword_count=len(keywords),
        )

        for keyword in keywords:

            timer_kw = _StepTimer()
            # Tokenize keyword
            q_tokens = tokenize_en(str(keyword))
            if not q_tokens:
                _jlog("bm25_keyword_empty_tokens", request_id, step="2.1", keyword=str(keyword))
                continue

            _jlog(
                "bm25_keyword_processing",
                request_id,
                step="2.1",
                keyword=str(keyword),
                token_count=len(q_tokens),
                tokens=q_tokens,
            )

            name_scores, desc_scores = bm25.get_scores(q_tokens)
            name_scores = np.asarray(name_scores, dtype=float)
            desc_scores = np.asarray(desc_scores, dtype=float)

            scores = np.maximum(name_scores, desc_scores)
            max_score = float(scores.max()) if scores.size > 0 else 0.0
            bm25_threshold = round(0.6 * max_score, 1)
            strong_mask = (scores >= bm25_threshold) if max_score > 0 else np.zeros_like(scores, dtype=bool)
            idxs = np.where(strong_mask)[0]

            idxs = list(idxs)[::-1]  # Reverse scan
            keyword_hits: Dict[str, Tuple[Dict[str, Any], float]] = {}
            hit_debug: List[dict] = []

            for idx in idxs:
                meta = metas[idx]
                if not meta:
                    continue
                entity_id = meta.get("id")
                if not entity_id:
                    continue

                # Take max BM25 score from name/desc
                bm25_score = float(max(name_scores[idx], desc_scores[idx]))

                # Debug: print which fields matched
                matched_fields = []
                if name_scores[idx] > 0:
                    matched_fields.append("name")
                if desc_scores[idx] > 0:
                    matched_fields.append("desc")

                if entity_id not in keyword_hits or bm25_score > keyword_hits[entity_id][1]:
                    keyword_hits[entity_id] = (meta, bm25_score)

                hit_debug.append({
                    "id": entity_id,
                    "name": meta.get("name"),
                    "fields": "/".join(matched_fields),
                    "bm25": bm25_score,
                })

            entities_hit[str(keyword)] = list(keyword_hits.values())

            _jlog(
                "bm25_keyword_done",
                request_id,
                step="2.1",
                keyword=str(keyword),
                bm25_threshold=round(bm25_threshold, 4),
                max_score=round(max_score, 4),
                candidate_count=len(idxs),
                hit_count=len(keyword_hits),
                sample_hits=hit_debug[:10],
                elapsed_sec=timer_kw.sec(),
            )

        # Log total BM25 results
        total_kw_hits = sum(len(v) for k, v in entities_hit.items() if k != "__vector__")
        _jlog(
            "bm25_all_done",
            request_id,
            step="2.1",
            keyword_count=len(keywords),
            total_hit_pairs=total_kw_hits,
        )
        _jlog(
            "search_entities_hybrid_complete",
            request_id,
            step="2.1",
            vector_hit_count=len(entities_hit.get("__vector__", [])),
            bm25_keyword_count=len(keywords),
            bm25_total_hit_pairs=total_kw_hits,
            unique_entity_count=len(
                {
                    meta.get("id")
                    for hits in entities_hit.values()
                    for meta, _ in hits
                    if meta
                }
            ),
            sample_entity_names=[
                meta.get("name") or meta.get("id")
                for hits in entities_hit.values()
                for meta, _ in hits[:1]
                if meta
            ][:20],
            elapsed_sec=timer_total.sec(),
        )
        return entities_hit

    def search_relationships_by_vec(
        self,
        keywords: List[str],
        relationship_top_k: int,
        relationship_vec_threshold: float,
        request_id: Optional[str] = None,
    ) -> Dict[str, List[Tuple[Dict[str, Any], float]]]:
        """
        Search relationships using vector similarity on keywords.

        Args:
            keywords: Keywords to search for
            relationship_top_k: Maximum number of results per keyword
            relationship_vec_threshold: Minimum similarity threshold
            request_id: Request ID for logging

        Returns:
            Dictionary mapping keyword -> [(meta, score), ...]
        """
        timer = _StepTimer()
        if not keywords:
            _jlog("search_relationships_empty", request_id, step="2.3", reason="no_keywords")
            return {}

        _jlog(
            "search_relationships_start",
            request_id,
            step="2.3",
            keyword_count=len(keywords),
            keywords=keywords,
            relationship_top_k=relationship_top_k,
            relationship_vec_threshold=relationship_vec_threshold,
        )

        # Embed keywords
        timer_embed = _StepTimer()
        vecs = self.embed(keywords)
        _jlog(
            "keywords_embedded",
            request_id,
            step="2.3",
            keyword_count=len(keywords),
            vector_dim=vecs.shape[1] if len(vecs.shape) > 1 else 0,
            elapsed_sec=timer_embed.sec(),
        )

        vdb = self.vector_db_manager.get_relationships_vdb(vecs.shape[1])
        out = {
            keyword: vdb.search(vecs[i], top_k=relationship_top_k, threshold=relationship_vec_threshold)
            for i, keyword in enumerate(keywords)
        }

        _jlog(
            "rel_vector_search_done",
            request_id,
            step="2.3",
            relationship_top_k=relationship_top_k,
            threshold=relationship_vec_threshold,
            keywords=keywords,
            per_keyword_hits={k: len(v) for k, v in out.items()},
            sample_any=[
                {"keyword": k, "label": _relationship_label(m), "sim": float(s)}
                for k, hits in out.items()
                for (m, s) in (hits[:1] if hits else [])
            ][:10],
            elapsed_sec=timer.sec(),
        )
        _jlog(
            "search_relationships_complete",
            request_id,
            step="2.3",
            keyword_count=len(keywords),
            total_hit_pairs=sum(len(v) for v in out.values()),
            unique_relationship_count=len(
                {
                    meta.get("id")
                    for hits in out.values()
                    for meta, _ in hits
                    if meta
                }
            ),
            sample_relationship_names=[
                _relationship_label(meta)
                for hits in out.values()
                for meta, _ in hits[:1]
                if meta
            ][:20],
            elapsed_sec=timer.sec(),
        )
        return out

    def embed_query(self, question: str, request_id: Optional[str] = None) -> np.ndarray:
        """
        Embed query text and normalize.

        Args:
            question: Query text
            request_id: Request ID for logging

        Returns:
            Normalized query vector
        """
        timer_embed = _StepTimer()
        _jlog(
            "query_embedding_start",
            request_id,
            step="2.0",
            question=question,
        )
        query_vec = self.embed([question])[0].astype(np.float32)
        faiss.normalize_L2(query_vec.reshape(1, -1))

        _jlog(
            "query_embedding_done",
            request_id,
            step="2.0",
            vector_dim=query_vec.shape[0],
            l2_norm=float(np.linalg.norm(query_vec)),
            elapsed_sec=timer_embed.sec(),
        )
        return query_vec
