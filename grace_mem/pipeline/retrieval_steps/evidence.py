# pipeline/retrieval_steps/evidence.py
"""
Evidence building from provenance and summaries.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
from grace_mem.storage import build_id_to_meta_maps
from grace_mem.services import Provenance
from grace_mem.utils.logger_config import _StepTimer, make_module_jlog
from grace_mem.pipeline.retrieval_steps.summary_scoring import (
    ScoringWeights,
    rank_summaries_by_graph_linked_score,
    select_summaries_rrf,
    select_summaries_rrf_mmr,
)

_jlog = make_module_jlog(name="grace_mem.Retrieval.Evidence", filename="kg_retrieval_evidence.jsonl")

_GRAPH_WEIGHTED_MODES = {
    "graph_count", "graph_semantic", "graph_semantic_penalty", "graph_weighted_sum"
}
_RRF_MODES = {"graph_rrf", "graph_rrf_mmr"}

# Max candidates fed to the cross-encoder reranker (bounds GPU memory).
_RERANK_POOL_CAP = 40


class EvidenceBuilder:
    """
    Builds evidence blocks from entity/relationship provenance and summaries.
    """

    def __init__(self, summaries_vdb: Any, vector_db_manager: Any, cache: Dict[str, Any], raw_context_lookup: Any = None) -> None:
        """
        Args:
            summaries_vdb: Summaries vector database
            vector_db_manager: Manager for accessing vector databases
            cache: Global cache with entity/relationship metadata
            raw_context_lookup: Optional RawContextLookup for use_raw_context mode
        """
        self.summaries_vdb = summaries_vdb
        self.vector_db_manager = vector_db_manager
        self.cache = cache
        self.raw_context_lookup = raw_context_lookup
        self.last_evidence_trace: dict[str, Any] = {}

    def build_evidence_block(
        self,
        context_entities: list[dict],
        context_relationships: list[dict],
        summary_topk_global: int,
        query_vec: Any,
        summary_vec_threshold: float,
        use_full_summary: bool = True,
        fallback_to_raw: bool = False,
        use_raw_context: bool = False,
        use_split_embeddings: bool = False,
        summary_direct_vector_topn: int = 0,
        summary_direct_vector_min_score: float = 0.0,
        summary_rerank_topk: int = 0,
        summary_rerank_cosine_only: bool = False,
        split_single_entry_raw: bool = False,
        query_text: Optional[str] = None,
        request_id: Optional[str] = None,
        summary_filter_mode: str = "semantic",
        scoring_weights: Optional[ScoringWeights] = None,
        hyde_vec: Any = None,
        hyde_weight: float = 0.0,
        hyde_mode: str = "blend",
        summary_per_entity_min: int = 0,
    ) -> str:
        """
        Build evidence block from provenance summaries.

        Args:
            context_entities: List of context entities
            context_relationships: List of context relationships
            summary_topk_global: Global top-K limit for evidence snippets
            query_vec: Query embedding vector
            summary_vec_threshold: Minimum similarity threshold for summaries
            use_full_summary: If True, fetch full summary text
            fallback_to_raw: If True, fallback to raw turn text when summary is truncated
            request_id: Request ID for logging
            summary_filter_mode: Ranking strategy — "semantic" | "graph_count" |
                "graph_semantic" | "graph_semantic_penalty"
            scoring_weights: ScoringWeights instance (used when mode != "semantic").
                Defaults to ScoringWeights() if None.
            hyde_vec: Optional HyDE hypothetical-summary embedding.
            hyde_weight: Blend weight for the HyDE similarity (0.0 = ignore HyDE).
            hyde_mode: How HyDE is used (semantic mode only):
                "blend" → score = (1-hyde_weight)*sim_query + hyde_weight*sim_hyde
                          (HyDE competes for top-K slots; can displace query hits).
                "fill"  → query similarity alone selects top-K; HyDE only rescues
                          summaries that fail the query threshold to fill *unused*
                          slots when query yields fewer than top-K candidates.
                          Never displaces a query hit.
            summary_per_entity_min: Minimum snippets guaranteed per source entity/
                relationship (0 = disabled, use global top-K only). When > 0, each
                source that has at least one passing candidate contributes its best
                snippet(s) before the remaining top-K slots are filled by score.
                This prevents high-scoring entities from crowding out all slots and
                improves turn-level recall for multi-session aggregation questions.

        Returns:
            Formatted evidence block text
        """
        timer_render = _StepTimer()
        _weights = scoring_weights if scoring_weights is not None else ScoringWeights()
        _jlog(
            "build_evidence_start",
            request_id,
            step="3",
            entity_count=len(context_entities or []),
            relationship_count=len(context_relationships or []),
            summary_topk_global=summary_topk_global,
            summary_vec_threshold=summary_vec_threshold,
            use_full_summary=use_full_summary,
            fallback_to_raw=fallback_to_raw,
            summary_filter_mode=summary_filter_mode,
        )
        _jlog(
            "evidence_stage_start",
            request_id,
            step="3.1",
            stage="score_events",
        )

        entity_id2meta, relationship_id2meta = build_id_to_meta_maps(self.cache)
        lines: list[str] = []

        # ── Split-embedding mode: score :u and :a entries as independent candidates ──
        if use_split_embeddings:
            return self._build_evidence_split(
                context_entities=context_entities,
                context_relationships=context_relationships,
                entity_id2meta=entity_id2meta,
                relationship_id2meta=relationship_id2meta,
                summary_topk_global=summary_topk_global,
                query_vec=query_vec,
                summary_vec_threshold=summary_vec_threshold,
                summary_direct_vector_topn=summary_direct_vector_topn,
                summary_direct_vector_min_score=summary_direct_vector_min_score,
                summary_rerank_topk=summary_rerank_topk,
                summary_rerank_cosine_only=summary_rerank_cosine_only,
                split_single_entry_raw=split_single_entry_raw,
                query_text=query_text,
                request_id=request_id,
            )

        # Deduplication: (session_id, message_id) to avoid duplicate turns
        seen_keys: set[tuple[str, str]] = set()

        # Cache: summary_id -> (snippet or None, score or None)
        snippet_cache: dict[str, tuple[str | None, float | None]] = {}
        # "fill" mode: candidates that fail the query threshold but pass it on the
        # HyDE vector, collected as (sim_hyde, ev) to backfill unused top-K slots.
        fill_events: list[tuple[float, dict]] = []
        _hyde_fill = (hyde_mode == "fill" and hyde_vec is not None and hyde_weight > 0.0)
        stage_stats = {
            "entity_events": 0,
            "relationship_events": 0,
            "dedup_skips": 0,
            "score_pass": 0,
            "score_fail": 0,
            "fetch_attempts": 0,
            "fetch_success": 0,
        }

        def key_of(ev: dict) -> tuple[str, str]:
            """Build a turn-level deduplication key from one provenance event."""
            return (str(ev.get("session_id")), str(ev.get("message_id")))

        # ========== Stage 1: Score events without fetching text ==========
        def score_event(ev: dict) -> Optional[float]:
            """
            Compute similarity score for a single event without fetching text.
            Returns None if below threshold or cannot compare.
            """
            summary_id = ev.get("summary_id")
            if not summary_id:
                return None

            # Use cached score if available
            if summary_id in snippet_cache:
                snippet, cached_score = snippet_cache[summary_id]
                _jlog(
                    "summary_cache_hit",
                    request_id,
                    step="3.1",
                    summary_id=summary_id,
                    stage="score",
                )
                return cached_score

            t = _StepTimer()

            # Always fetch the raw score so it can be logged even for below-threshold summaries.
            raw_score = self.summaries_vdb.compare_by_id_raw(
                summary_id,
                query_vec,
                request_id=request_id,
                debug_context={"step": "3.1", "source": "summary_score_event"},
            )
            if raw_score is None:
                # Vector not found
                snippet_cache[summary_id] = (None, None)
                _jlog(
                    "summary_scored",
                    request_id,
                    step="3.1",
                    summary_id=summary_id,
                    fetched=False,
                    score=None,
                    raw_score_missing=True,
                    passed_threshold=False,
                    threshold=summary_vec_threshold,
                    elapsed_sec=t.sec(),
                )
                return None

            score = float(raw_score)
            effective_threshold = summary_vec_threshold if isinstance(summary_vec_threshold, (int, float)) else 0.0
            # HyDE "blend" mode: combine query similarity with hypothetical-summary
            # similarity so HyDE-favored summaries can compete for top-K slots.
            if hyde_mode == "blend" and hyde_vec is not None and hyde_weight > 0.0:
                hyde_raw = self.summaries_vdb.compare_by_id_raw(
                    summary_id,
                    hyde_vec,
                    request_id=request_id,
                    debug_context={"step": "3.1", "source": "summary_score_hyde"},
                )
                if hyde_raw is not None:
                    score = (1.0 - hyde_weight) * score + hyde_weight * float(hyde_raw)
            passed = score >= effective_threshold

            if not passed:
                snippet_cache[summary_id] = (None, None)
                # HyDE "fill" mode: rescue query-rejected summaries for backfill only.
                if _hyde_fill:
                    hyde_raw = self.summaries_vdb.compare_by_id_raw(
                        summary_id,
                        hyde_vec,
                        request_id=request_id,
                        debug_context={"step": "3.1", "source": "summary_score_hyde_fill"},
                    )
                    if hyde_raw is not None and float(hyde_raw) >= effective_threshold:
                        fill_events.append((float(hyde_raw), ev))
                _jlog(
                    "summary_scored",
                    request_id,
                    step="3.1",
                    summary_id=summary_id,
                    fetched=False,
                    score=score,
                    passed_threshold=False,
                    threshold=summary_vec_threshold,
                    elapsed_sec=t.sec(),
                )
                return None

            # Cache score only (not text yet)
            snippet_cache[summary_id] = (None, score)
            _jlog(
                "summary_scored",
                request_id,
                step="3.1",
                summary_id=summary_id,
                fetched=False,
                score=score,
                passed_threshold=True,
                threshold=summary_vec_threshold,
                elapsed_sec=t.sec(),
            )
            return score

        # ========== Stage 2: Collect all scored candidates ==========
        scored_events: list[tuple[float, dict]] = []
        # Maps object-id(ev) → source entity/relationship id for per-entity quota.
        ev_source: dict[int, str] = {}

        # From entities
        for ent in (context_entities or []):
            entity_id = ent.get("id")
            meta = entity_id2meta.get(entity_id, {}) or {}
            events = sorted(
                Provenance.prov_to_events(meta.get("prov") or {}),
                key=lambda e: e.get("ts", 0),
                reverse=True,
            )
            _jlog(
                "evidence_iter_entity",
                request_id,
                step="3.2",
                entity_id=entity_id,
                event_count=len(events),
            )

            for ev in events:
                stage_stats["entity_events"] += 1
                k = key_of(ev)
                if k in seen_keys:
                    stage_stats["dedup_skips"] += 1
                    continue

                score = score_event(ev)
                if score is None:
                    stage_stats["score_fail"] += 1
                    continue  # Below threshold or no score

                scored_events.append((score, ev))
                ev_source[id(ev)] = entity_id or ""
                seen_keys.add(k)
                stage_stats["score_pass"] += 1

        # From relationships
        for rel in (context_relationships or []):
            relationship_id = rel.get("rel_id")
            meta = relationship_id2meta.get(relationship_id, {}) or {}
            events = sorted(
                Provenance.prov_to_events(meta.get("prov") or {}),
                key=lambda e: e.get("ts", 0),
                reverse=True,
            )
            _jlog(
                "evidence_iter_relationship",
                request_id,
                step="3.2",
                relationship_id=relationship_id,
                event_count=len(events),
            )

            for ev in events:
                stage_stats["relationship_events"] += 1
                k = key_of(ev)
                if k in seen_keys:
                    stage_stats["dedup_skips"] += 1
                    continue

                score = score_event(ev)
                if score is None:
                    stage_stats["score_fail"] += 1
                    continue

                scored_events.append((score, ev))
                ev_source[id(ev)] = relationship_id or ""
                seen_keys.add(k)
                stage_stats["score_pass"] += 1

        _jlog(
            "evidence_candidates_collected",
            request_id,
            step="3.2",
            entity_events=stage_stats["entity_events"],
            relationship_events=stage_stats["relationship_events"],
            dedup_skips=stage_stats["dedup_skips"],
            scored_candidates=len(scored_events),
            score_pass=stage_stats["score_pass"],
            score_fail=stage_stats["score_fail"],
        )

        # ========== Stage 3: Rank and take global top-K ==========
        # "semantic" mode: sort by semantic score (baseline, unchanged behaviour).
        # Graph modes: rerank via graph-linked scoring formula.
        topk = summary_topk_global if isinstance(summary_topk_global, int) else None

        if summary_filter_mode in _GRAPH_WEIGHTED_MODES:
            graph_ranked = rank_summaries_by_graph_linked_score(
                scored_events=scored_events,
                context_entities=context_entities,
                context_relationships=context_relationships,
                cache=self.cache,
                weights=_weights,
                summaries_vdb=self.summaries_vdb,
                topk=topk,
            )
            reranked: list[tuple[float, dict]] = []
            for sc, ev in graph_ranked:
                reranked.append((sc.final_score, ev))
                _jlog(
                    "summary_graph_scored",
                    request_id,
                    step="3.3",
                    **sc.to_log_dict(_weights, summary_filter_mode),
                )
            scored_events = reranked
        elif summary_filter_mode in _RRF_MODES:
            _rrf_fn = (
                select_summaries_rrf_mmr
                if summary_filter_mode == "graph_rrf_mmr"
                else select_summaries_rrf
            )
            rrf_ranked = _rrf_fn(
                scored_events=scored_events,
                context_entities=context_entities,
                context_relationships=context_relationships,
                cache=self.cache,
                weights=_weights,
                summaries_vdb=self.summaries_vdb,
                topk=topk,
            )
            reranked = []
            for sc, ev in rrf_ranked:
                reranked.append((sc.final_score, ev))
                _jlog(
                    "summary_rrf_scored",
                    request_id,
                    step="3.3",
                    **sc.to_log_dict(_weights, summary_filter_mode),
                )
            scored_events = reranked
        else:
            # Semantic mode: stable descending sort by cosine score
            scored_events.sort(key=lambda x: x[0], reverse=True)

            # Per-entity quota: guarantee at least summary_per_entity_min snippets
            # per source entity/relationship before applying global top-K.
            if summary_per_entity_min > 0 and topk is not None:
                # Step 1: collect the best `summary_per_entity_min` events per source
                source_counts: dict[str, int] = {}
                guaranteed: list[tuple[float, dict]] = []
                remainder: list[tuple[float, dict]] = []
                for score, ev in scored_events:
                    src = ev_source.get(id(ev), "")
                    if source_counts.get(src, 0) < summary_per_entity_min:
                        guaranteed.append((score, ev))
                        source_counts[src] = source_counts.get(src, 0) + 1
                    else:
                        remainder.append((score, ev))
                # Step 2: fill remaining slots from remainder (already sorted by score)
                n_fill = max(0, topk - len(guaranteed))
                scored_events = guaranteed + remainder[:n_fill]
                _jlog(
                    "evidence_per_entity_quota",
                    request_id,
                    step="3.3",
                    per_entity_min=summary_per_entity_min,
                    guaranteed_count=len(guaranteed),
                    remainder_count=len(remainder),
                    filled=min(n_fill, len(remainder)),
                    total_selected=len(scored_events),
                )
            elif topk is not None:
                scored_events = scored_events[:topk]

            # HyDE "fill" mode: backfill any unused top-K slots with HyDE-rescued
            # summaries (never displaces a query hit).
            if _hyde_fill and topk is not None and len(scored_events) < topk and fill_events:
                fill_events.sort(key=lambda x: x[0], reverse=True)
                used_keys = {key_of(ev) for _, ev in scored_events}
                n_filled = 0
                for sc, ev in fill_events:
                    if len(scored_events) >= topk:
                        break
                    k = key_of(ev)
                    if k in used_keys:
                        continue
                    scored_events.append((sc, ev))
                    used_keys.add(k)
                    n_filled += 1
                _jlog(
                    "hyde_fill_applied",
                    request_id,
                    step="3.3",
                    fill_candidates=len(fill_events),
                    slots_filled=n_filled,
                    final_count=len(scored_events),
                )

        _jlog(
            "evidence_topk_selected",
            request_id,
            step="3.3",
            requested_topk=summary_topk_global,
            selected_count=len(scored_events),
            summary_filter_mode=summary_filter_mode,
            sample_top_candidates=[
                {
                    "rank": i + 1,
                    "score": score,
                    "summary_id": ev.get("summary_id"),
                    "session_id": ev.get("session_id"),
                    "message_id": ev.get("message_id"),
                }
                for i, (score, ev) in enumerate(scored_events[:topk] if topk is not None else scored_events)
            ],
        )

        # ========== Stage 4: Fetch text for top-K only ==========
        def fetch_snippet(ev: dict, score: float) -> Optional[str]:
            """
            Fetch summary/raw text for events that passed threshold and are in top-K.
            Updates snippet_cache with text + score.
            """
            summary_id = ev.get("summary_id")
            if not summary_id:
                return None

            # Use cached snippet if available
            if summary_id in snippet_cache:
                cached_snippet, cached_score = snippet_cache[summary_id]
                if cached_snippet is not None:
                    _jlog(
                        "summary_cache_hit",
                        request_id,
                        step="3.4",
                        summary_id=summary_id,
                        stage="fetch",
                    )
                    return cached_snippet
                # If only score cached, fetch text below

            t = _StepTimer()
            snippet: Optional[str] = None
            stage_stats["fetch_attempts"] += 1

            # 1) Raw context mode: skip summary text, fetch raw turn text from CSV lookup
            if use_raw_context:
                try:
                    if self.raw_context_lookup is None:
                        raise RuntimeError("use_raw_context=True but raw_context_lookup is not configured (set raw_context_data_dir)")
                    session_id = ev.get("session_id")
                    message_id = ev.get("message_id")
                    snippet = self.raw_context_lookup.get(str(session_id), int(message_id))
                    _jlog(
                        "evidence_raw_context_fetch",
                        request_id,
                        step="3.4",
                        summary_id=summary_id,
                        session_id=session_id,
                        message_id=message_id,
                        fetched=bool(snippet),
                        raw_len=len(snippet) if snippet else 0,
                    )
                except Exception as e:
                    _jlog(
                        "evidence_raw_context_failed",
                        request_id,
                        step="3.4",
                        summary_id=summary_id,
                        error=str(e),
                    )
            else:
                # 2) Fetch summary text
                if use_full_summary:
                    try:
                        snippet = self.summaries_vdb.get_summary_text_by_id(summary_id)
                    except (AttributeError, TypeError):
                        xs = self.summaries_vdb.get_summaries_by_ids([summary_id], top_n=1)
                        snippet = xs[0] if xs else None
                else:
                    xs = self.summaries_vdb.get_summaries_by_ids([summary_id], top_n=1)
                    snippet = xs[0] if xs else None

                # 3) Fallback to raw turn text if enabled and summary is truncated
                if fallback_to_raw and (not snippet or "…" in (snippet or "")):
                    try:
                        session_id, message_id = ev.get("session_id"), ev.get("message_id")
                        raw = self.vector_db_manager.summary_store.get_raw_turn_text(session_id, message_id)
                        if raw:
                            snippet = raw
                            _jlog(
                                "evidence_fallback_to_raw",
                                request_id,
                                step="3.4",
                                summary_id=summary_id,
                                session_id=session_id,
                                message_id=message_id,
                                raw_len=len(raw),
                            )
                    except Exception as e:
                        _jlog(
                            "evidence_fallback_failed",
                            request_id,
                            step="3.4",
                            summary_id=summary_id,
                            error=str(e),
                        )

            fetched = bool(snippet)
            if fetched:
                stage_stats["fetch_success"] += 1

            # Update cache with snippet + score
            snippet_cache[summary_id] = (snippet, score if fetched else None)

            _jlog(
                "summary_fetched",
                request_id,
                step="3.4",
                summary_id=summary_id,
                fetched=fetched,
                score=score if fetched else None,
                passed_threshold=fetched,
                threshold=summary_vec_threshold,
                elapsed_sec=t.sec(),
            )

            return snippet if fetched else None

        # Collect evidence items: (score, snippet, summary_id, dialogue_datetime)
        evidence_items: list[tuple[float, str, str | None, str | None]] = []

        for score, ev in scored_events:
            summary_id = ev.get("summary_id")
            snippet = fetch_snippet(ev, score)
            if not snippet:
                continue

            # Extract dialogue_datetime from event or summary metadata
            dialogue_datetime = ev.get("dialogue_datetime")
            if dialogue_datetime is None and summary_id:
                # Try to get from summary metadata
                for meta in self.summaries_vdb._meta:
                    if meta.get("summary_id") == summary_id:
                        dialogue_datetime = meta.get("dialogue_datetime")
                        break

            evidence_items.append((score, snippet, summary_id, dialogue_datetime))

        _jlog(
            "evidence_fetch_complete",
            request_id,
            step="3.4",
            fetch_attempts=stage_stats["fetch_attempts"],
            fetch_success=stage_stats["fetch_success"],
            evidence_item_count=len(evidence_items),
        )

        # ========== Stage 5: Format evidence block ==========
        if evidence_items:
            _jlog(
                "evidence_items",
                request_id,
                step="3.5",
                count=len(evidence_items),
                items=[
                    {
                        "rank": i + 1,
                        "score": score,
                        "len": len(snippet) if snippet else 0,
                        "summary_id": summary_id,
                        "dialogue_datetime": dt,
                    }
                    for i, (score, snippet, summary_id, dt) in enumerate(evidence_items[:20])
                ],
            )

        if evidence_items:
            lines.append("### Evidence Summary")
            for score, snippet, summary_id, dialogue_datetime in evidence_items:
                score_str = f"{score:.3f}" if score is not None else "--"
                summary_id_str = summary_id or "N/A"
                dt_str = f"[{dialogue_datetime}]" if dialogue_datetime else ""
                lines.append(f"  • {dt_str}[sid={summary_id_str}][score={score_str}] {snippet} ")

        result = "\n".join(lines)
        self.last_evidence_trace = {
            "request_id": request_id,
            "selected_evidence_count": len(evidence_items),
            "selected_evidence": [
                {
                    "rank": index + 1,
                    "score": round(float(score), 6),
                    "summary_id": summary_id,
                    "dialogue_datetime": dialogue_datetime,
                    "preview": snippet[:160],
                }
                for index, (score, snippet, summary_id, dialogue_datetime) in enumerate(evidence_items)
            ],
            "score_pass_count": stage_stats["score_pass"],
            "score_fail_count": stage_stats["score_fail"],
            "dedup_skips": stage_stats["dedup_skips"],
            "fetch_attempts": stage_stats["fetch_attempts"],
            "fetch_success": stage_stats["fetch_success"],
        }

        _jlog(
            "evidence_format_complete",
            request_id,
            step="3.5",
            total_snippets=len(evidence_items),
            line_count=len(lines),
            result_length=len(result),
        )

        _jlog(
            "build_evidence_complete",
            request_id,
            step="3",
            total_snippets=len(evidence_items),
            cache_size=len(snippet_cache),
            result_length=len(result),
            elapsed_sec=timer_render.sec(),
        )

        return result

    def _fetch_split_raw_text(self, entry_id: str) -> Optional[str]:
        """Raw turn text for a split/single entry: raw_text metadata → stored
        document → raw_context_lookup (older artifacts lack text/raw_text
        metadata; fetch the raw pair text from script_data instead)."""
        snippet = (
            self.summaries_vdb.get_raw_turn_text_by_id(entry_id)
            or self.summaries_vdb.get_text_by_entry_id(entry_id)
        )
        if not snippet and self.raw_context_lookup is not None:
            sess, _, msg = str(entry_id).rpartition(":")
            try:
                snippet = self.raw_context_lookup.get(sess, int(msg))
            except (ValueError, TypeError):
                snippet = None
        if not snippet:
            # Last resort: legacy artifacts store only summary_text metadata.
            snippet = self.summaries_vdb.get_summary_text_by_id(entry_id)
        return snippet

    def _build_evidence_split(
        self,
        *,
        context_entities: list[dict],
        context_relationships: list[dict],
        entity_id2meta: dict,
        relationship_id2meta: dict,
        summary_topk_global: int,
        query_vec: Any,
        summary_vec_threshold: float,
        summary_direct_vector_topn: int = 0,
        summary_direct_vector_min_score: float = 0.0,
        summary_rerank_topk: int = 0,
        summary_rerank_cosine_only: bool = False,
        split_single_entry_raw: bool = False,
        query_text: Optional[str] = None,
        request_id: Optional[str],
    ) -> str:
        """
        Evidence building for use_split_embeddings=True.

        Each prov event expands to two VDB entries — :u (user raw) and :a (assistant
        compressed) — which are scored independently and compete for top-K slots.
        """
        timer = _StepTimer()
        topk = summary_topk_global if isinstance(summary_topk_global, int) else None
        effective_threshold = float(summary_vec_threshold) if isinstance(summary_vec_threshold, (int, float)) else 0.0

        # entry_id → score cache
        entry_score_cache: dict[str, Optional[float]] = {}

        def score_entry(entry_id: str) -> Optional[float]:
            if entry_id in entry_score_cache:
                return entry_score_cache[entry_id]
            raw = self.summaries_vdb.compare_by_id_raw(
                entry_id, query_vec, request_id=request_id,
                debug_context={"step": "3.1", "source": "split_score"},
            )
            if raw is None or float(raw) < effective_threshold:
                entry_score_cache[entry_id] = None
                return None
            score = float(raw)
            entry_score_cache[entry_id] = score
            return score

        # Collect all scored (score, entry_id, ev) candidates, dedup by entry_id
        seen_entry_ids: set[str] = set()
        scored_entries: list[tuple[float, str, dict]] = []
        ev_source: dict[str, str] = {}  # entry_id → entity/rel id

        def collect_events(events: list[dict], source_id: str) -> None:
            for ev in events:
                summary_id = ev.get("summary_id")
                if not summary_id:
                    continue
                # Single-entry mode (e.g. LoCoMo): one entry per summary_id, no :u/:a.
                suffixes = ("",) if split_single_entry_raw else (":u", ":a")
                for suffix in suffixes:
                    entry_id = f"{summary_id}{suffix}"
                    if entry_id in seen_entry_ids:
                        continue
                    seen_entry_ids.add(entry_id)
                    score = score_entry(entry_id)
                    if score is None:
                        continue
                    scored_entries.append((score, entry_id, ev))
                    ev_source[entry_id] = source_id

        for ent in (context_entities or []):
            entity_id = ent.get("id")
            meta = entity_id2meta.get(entity_id, {}) or {}
            events = sorted(
                Provenance.prov_to_events(meta.get("prov") or {}),
                key=lambda e: e.get("ts", 0),
                reverse=True,
            )
            collect_events(events, entity_id or "")

        for rel in (context_relationships or []):
            relationship_id = rel.get("rel_id")
            meta = relationship_id2meta.get(relationship_id, {}) or {}
            events = sorted(
                Provenance.prov_to_events(meta.get("prov") or {}),
                key=lambda e: e.get("ts", 0),
                reverse=True,
            )
            collect_events(events, relationship_id or "")

        # Helper: append direct-vector hits (by raw cosine) not already collected.
        def add_direct(min_score: float) -> int:
            if not (summary_direct_vector_topn and summary_direct_vector_topn > 0):
                return 0
            have = {entry_id for _, entry_id, _ in scored_entries}
            try:
                hits = self.summaries_vdb.search(
                    query_vec, top_k=int(summary_direct_vector_topn),
                    threshold=float(min_score) if min_score and min_score > 0 else None,
                )
            except Exception as exc:  # never let direct search break evidence building
                _jlog("evidence_split_direct_search_error", request_id, step="3.2", error=str(exc))
                return 0
            added = 0
            for meta, score in hits:
                if min_score and float(score) < float(min_score):
                    continue
                entry_id = str(meta.get("id") or "").strip()
                if not entry_id or entry_id in have:
                    continue
                have.add(entry_id)
                ev = {
                    "summary_id": meta.get("summary_id"),
                    "session_id": meta.get("session_id"),
                    "message_id": meta.get("message_id"),
                    "dialogue_datetime": meta.get("dialogue_datetime"),
                }
                scored_entries.append((float(score), entry_id, ev))
                ev_source[entry_id] = "direct_vector"
                added += 1
            return added

        direct_added = 0
        if summary_rerank_topk and summary_rerank_topk > 0 and query_text:
            # ── Retrieve-then-rerank: cast a wide net (prov + direct at a low
            # cosine floor), then a cross-encoder reranker picks the final top-N.
            # Decouples recall (wide, cheap cosine) from context cleanliness (rerank).
            direct_added = add_direct(summary_direct_vector_min_score)
            # Cap the pool by cosine before reranking to bound cross-encoder memory
            # (large pools OOM the GPU). Reranking the cosine-top-N is sufficient.
            cand = sorted(scored_entries, key=lambda x: x[0], reverse=True)
            pool_full = len(cand)
            if len(cand) > _RERANK_POOL_CAP:
                cand = cand[:_RERANK_POOL_CAP]
            if split_single_entry_raw:
                texts = [self._fetch_split_raw_text(eid) or "" for _, eid, _ in cand]
            else:
                texts = [self.summaries_vdb.get_text_by_entry_id(eid) or "" for _, eid, _ in cand]
            idx_text = [(i, t) for i, t in enumerate(texts) if t.strip()]
            reranked_ok = False
            if summary_rerank_cosine_only:
                # Ablation: wide net → plain cosine top-N (skip the cross-encoder).
                scored_entries = cand[:int(summary_rerank_topk)]
            elif idx_text:
                from grace_mem.utils.reranker import get_reranker
                try:
                    ranked = get_reranker().rerank(query_text, [t for _, t in idx_text], batch_size=2)
                    order = [idx_text[li][0] for li, _ in ranked][:int(summary_rerank_topk)]
                    scored_entries = [cand[i] for i in order]
                    reranked_ok = True
                except Exception as exc:  # OOM / model error → graceful cosine fallback
                    _jlog("evidence_split_rerank_error", request_id, step="3.3", error=str(exc))
                    scored_entries = cand[:int(summary_rerank_topk)]
            else:
                scored_entries = cand[:int(summary_rerank_topk)]
            _jlog(
                "evidence_split_reranked", request_id, step="3.3",
                pool_size=pool_full, reranked_pool=len(cand),
                direct_vector_added=direct_added, kept=len(scored_entries),
                reranked=reranked_ok,
            )
        else:
            # Prov keeps its full top-K budget; direct hits are EXTRA slots on top
            # (added only when they clear a high min-score), never displacing prov.
            scored_entries.sort(key=lambda x: x[0], reverse=True)
            if topk is not None:
                scored_entries = scored_entries[:topk]
            if summary_direct_vector_min_score and summary_direct_vector_min_score > 0:
                direct_added = add_direct(summary_direct_vector_min_score)
                scored_entries.sort(key=lambda x: x[0], reverse=True)

        _jlog(
            "evidence_split_selected",
            request_id,
            step="3.3",
            total_candidates=len(seen_entry_ids),
            direct_vector_added=direct_added,
            selected_count=len(scored_entries),
            sample=[
                {"rank": i + 1, "score": score, "entry_id": entry_id}
                for i, (score, entry_id, _) in enumerate(scored_entries[:10])
            ],
        )

        # Fetch text and format
        evidence_items: list[tuple[float, str, str, Optional[str]]] = []
        for score, entry_id, ev in scored_entries:
            if split_single_entry_raw:
                snippet = self._fetch_split_raw_text(entry_id)
            else:
                snippet = self.summaries_vdb.get_text_by_entry_id(entry_id)
            if not snippet:
                continue
            dialogue_datetime = ev.get("dialogue_datetime")
            evidence_items.append((score, snippet, entry_id, dialogue_datetime))

        lines: list[str] = []
        if evidence_items:
            lines.append("### Evidence Summary")
            for score, snippet, entry_id, dialogue_datetime in evidence_items:
                score_str = f"{score:.3f}"
                dt_str = f"[{dialogue_datetime}]" if dialogue_datetime else ""
                lines.append(f"  • {dt_str}[sid={entry_id}][score={score_str}] {snippet} ")

        result = "\n".join(lines)

        self.last_evidence_trace = {
            "request_id": request_id,
            "mode": "split_embeddings",
            "selected_evidence_count": len(evidence_items),
            "selected_evidence": [
                {
                    "rank": i + 1,
                    "score": round(float(score), 6),
                    "entry_id": entry_id,
                    "dialogue_datetime": dialogue_datetime,
                    "preview": snippet[:160],
                }
                for i, (score, snippet, entry_id, dialogue_datetime) in enumerate(evidence_items)
            ],
        }

        _jlog(
            "build_evidence_complete",
            request_id,
            step="3",
            mode="split_embeddings",
            total_snippets=len(evidence_items),
            result_length=len(result),
            elapsed_sec=timer.sec(),
        )
        return result
