"""
Refactored Retriever that uses modular components from retrieval/ folder.
"""
import os
import uuid
from typing import Any

import numpy as np

from grace_mem.domain.extraction import KeywordExtractionResult
from grace_mem.retrieval.ablation import flag_enabled
from grace_mem.retrieval.candidates import CandidateSet
from grace_mem.retrieval.config import RetrieverConfig
from grace_mem.retrieval.keywords import generate_query_keywords
from grace_mem.retrieval.query_rewrite import maybe_rewrite_retrieval_question
from grace_mem.retrieval.raw_turn_lookup import RawContextLookup
from grace_mem.retrieval.rendering import render_context_text

# Import modular components
from grace_mem.retrieval.steps import (
    EntityRelationshipSearcher,
    EvidenceBuilder,
    EvidenceFilter,
    SAConfig,
    SpreadingActivationEngine,
)
from grace_mem.retrieval.steps.adaptive import additive_merge
from grace_mem.retrieval.steps.narrowing import NarrowingModule
from grace_mem.retrieval.steps.temporal_relevance import date_within_coarse_range
from grace_mem.retrieval.trace import (
    build_adaptive_trace,
    build_stage_trace_snapshot,
    dedupe_preserve_order,
    format_retrieval_stage_trace_text,
)
from grace_mem.services.cache.cache import build_id_to_meta_maps
from grace_mem.temporal.query_time_parser import parse_query_time
from grace_mem.utils.logger_config import _StepTimer, make_module_jlog, setup_logger

_jlog = make_module_jlog(name="grace_mem.Retriever", filename="kg_retriever.jsonl")
_trace_jlog = make_module_jlog(name="grace_mem.Retriever.Trace", filename="kg_retrieval_trace.jsonl")
_TRACE_PRETTY_LOG_DIR = os.environ.get("KG_TRACE_PRETTY_LOG_DIR", "logs")
_trace_pretty_log = setup_logger(
    name="kg_retrieval_trace_pretty",
    log_dir=_TRACE_PRETTY_LOG_DIR,
    to_console=False,
)






#: Set to anything but 0/empty/false to make a retrieval failure raise instead of
#: degrading to an empty context. Off by default: an online caller wants the
#: degraded answer, and every existing run was produced that way.
STRICT_ENV = "KG_RETRIEVAL_STRICT"


class RetrievalFailedError(RuntimeError):
    """Retrieval raised, and the run asked to hear about it rather than degrade.

    The tolerant default returns "(no KG context)" on any exception, which is
    right for a service and wrong for a benchmark: the answering model still
    produces an answer, the row still gets written, and a FalkorDB timeout is
    scored as if the memory system had simply found nothing. That is a
    technical failure counted as a retrieval-quality result.

    Under KG_RETRIEVAL_STRICT the exception reaches the caller instead, so a
    benchmark can record the question as failed rather than as answered. Either
    way `last_retrieval_trace["retrieval_failed"]` is set, so a tolerant run can
    still tell the two apart afterwards.
    """


def strict_retrieval_enabled() -> bool:
    """True when this process was asked to raise on retrieval failure."""
    return os.getenv(STRICT_ENV, "0").lower() not in ("0", "", "false")


class Retriever:
    """
    Refactored Knowledge Graph Retriever using modular components.

    Components:
    - EntityRelationshipSearcher: Hybrid entity/relationship search
    - TemporalRelevanceCalculator: Temporal relevance scoring
    - EvidenceBuilder: Evidence block building
    - EvidenceFilter: Filtering and reranking
    """

    # Class-level default configuration
    DEFAULTS = RetrieverConfig()

    def __init__(self, *, llm: Any, graph: Any, mgr: Any, embed: Any, cache: dict[str, Any], config: dict | RetrieverConfig | None = None) -> None:
        """
        Initialize retriever with modular components.

        Args:
            llm: LLM client for keyword extraction
            graph: FalkorDB graph interface
            mgr: VDB manager
            embed: Embedding function
            cache: Global cache
            config: Optional configuration (dict or RetrieverConfig)
        """
        self.llm = llm
        self.graph = graph
        self.MGR = mgr
        self.embed = embed
        self.cache = cache

        # Process config parameter
        if config is None:
            self.cfg = Retriever.DEFAULTS
        elif isinstance(config, RetrieverConfig):
            self.cfg = config
        elif isinstance(config, dict):
            # Merge dict config into defaults
            base = {k: v for k, v in Retriever.DEFAULTS.__dict__.items()}
            base.update(config)
            self.cfg = RetrieverConfig(**base)
        else:
            raise TypeError(f"config must be dict, RetrieverConfig, or None; got {type(config)}")

        # Initialize VDB with configured dimension
        self.sum_vdb = self.MGR.get_summaries_vdb(dim=self.cfg.summary_embed_dim)

        # Initialize modular components
        self.searcher = EntityRelationshipSearcher(
            vector_db_manager=self.MGR,
            embed_function=self.embed
        )
        raw_context_lookup = (
            RawContextLookup(self.cfg.raw_context_data_dir)
            if (self.cfg.use_raw_context or self.cfg.use_split_embeddings) and self.cfg.raw_context_data_dir
            else None
        )
        self.evidence_builder = EvidenceBuilder(
            summaries_vdb=self.sum_vdb,
            vector_db_manager=self.MGR,
            cache=self.cache,
            raw_context_lookup=raw_context_lookup,
        )
        self.evidence_filter = EvidenceFilter(
            vector_db_manager=self.MGR,
            cache=self.cache
        )


        # Spreading activation engine (constructed regardless; only runs when cfg flag is on)
        self.sa_engine = SpreadingActivationEngine(
            graph=self.graph,
            vector_db_manager=self.MGR,
            cache=self.cache,
            config=SAConfig(
                max_hops=self.cfg.sa_max_hops,
                rescale_c=self.cfg.sa_rescale_c,
                tau_a=self.cfg.sa_tau_a,
                max_activated=self.cfg.sa_max_activated,
            ),
        )
        self.last_retrieval_trace: dict[str, Any] = {}
        self._last_stage_trace: dict[str, Any] = {}
        self._last_adaptive_trace: dict[str, Any] = {}

        # Narrowing module: post-evidence narrowing step (auto-filter optimization target)
        # KG_NARROWING_ENABLED=0 turns it into identity passthrough (e.g. LongMem
        # grep-agent runs, where a downstream agent does the narrowing instead).
        self.narrowing_module = NarrowingModule(
            enabled=os.getenv("KG_NARROWING_ENABLED", "1").lower() not in ("0", "false", ""),
        )

        # [LOG] Retriever initialization with effective config.
        # `inert_config` is the part of `config` that was accepted and ignored,
        # recorded so a run's own trace can be read back without knowing which
        # knobs were retired (see retrieval/config.INERT_FIELDS).
        _jlog(
            "retriever_initialized",
            request_id="INIT",
            config={k: v for k, v in self.cfg.__dict__.items()},
            inert_config=self.cfg.inert_overrides(),
        )

    def _entity_name_by_id(self, entity_id: str | None) -> str:
        """Resolve one entity ID into a human-readable display name."""
        if not entity_id:
            return "?"
        ent_id2meta, _ = build_id_to_meta_maps(self.cache)
        meta = ent_id2meta.get(entity_id, {}) or {}
        return meta.get("name") or entity_id

    def _entity_names_from_ids(self, entity_ids: list[str]) -> list[str]:
        """Resolve multiple entity IDs into deduplicated display names."""
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

    def _relationship_names_from_ids(self, relationship_ids: list[str]) -> list[str]:
        """Resolve multiple relationship IDs into deduplicated readable labels."""
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

    def _entity_names_from_relationship_ids(self, relationship_ids: list[str]) -> list[str]:
        """Resolve relationship IDs into readable endpoint entity names."""
        ent_id2meta, rel_id2meta = build_id_to_meta_maps(self.cache)
        names: list[str] = []
        seen: set[str] = set()
        for relationship_id in relationship_ids:
            meta = rel_id2meta.get(relationship_id, {}) or {}
            if not meta:
                continue
            source_id = meta.get("source_id")
            target_id = meta.get("target_id")
            source_meta = ent_id2meta.get(source_id, {}) if isinstance(source_id, str) else {}
            target_meta = ent_id2meta.get(target_id, {}) if isinstance(target_id, str) else {}
            src_name = (
                meta.get("source_entity")
                or source_meta.get("name")
                or source_id
            )
            tgt_name = (
                meta.get("target_entity")
                or target_meta.get("name")
                or target_id
            )
            for name in (src_name, tgt_name):
                if name and name not in seen:
                    names.append(name)
                    seen.add(name)
        return names


    def _relationship_name_from_edge(self, edge: dict[str, Any]) -> str:
        """Render one edge-subgraph record into a readable label."""
        src_name = edge.get("source_name") or self._entity_name_by_id(edge.get("source_id"))
        tgt_name = edge.get("target_name") or self._entity_name_by_id(edge.get("target_id"))
        desc = (edge.get("rel_desc") or edge.get("description") or "").strip()
        return f"{src_name} -> {tgt_name}" if not desc else f"{src_name} -> {tgt_name} | {desc}"

    def _entity_names_from_node_subgraph(self, node_subgraph: dict[str, dict]) -> list[str]:
        """Collect all node names present in a node subgraph."""
        names: list[str] = []
        for node_id, payload in (node_subgraph or {}).items():
            self_meta = (payload or {}).get("self") or {}
            names.append(self_meta.get("name") or node_id)
            for neighbor in (payload or {}).get("neighbors") or []:
                names.append(neighbor.get("neighbor_name") or neighbor.get("neighbor_id"))
        return dedupe_preserve_order(names)

    def _relationship_names_from_node_subgraph(self, node_subgraph: dict[str, dict]) -> list[str]:
        """Collect all edge labels present in a node subgraph."""
        labels: list[str] = []
        for node_id, payload in (node_subgraph or {}).items():
            self_meta = (payload or {}).get("self") or {}
            src_name = self_meta.get("name") or node_id
            for neighbor in (payload or {}).get("neighbors") or []:
                tgt_name = neighbor.get("neighbor_name") or neighbor.get("neighbor_id")
                desc = (neighbor.get("rel_desc") or "").strip()
                labels.append(f"{src_name} -> {tgt_name}" if not desc else f"{src_name} -> {tgt_name} | {desc}")
        return dedupe_preserve_order(labels)

    def _entity_names_from_edge_subgraph(self, edge_subgraph: list[dict[str, Any]]) -> list[str]:
        """Collect all endpoint entity names present in an edge subgraph."""
        names: list[str] = []
        for edge in edge_subgraph or []:
            names.append(edge.get("source_name") or self._entity_name_by_id(edge.get("source_id")))
            names.append(edge.get("target_name") or self._entity_name_by_id(edge.get("target_id")))
        return dedupe_preserve_order(names)

    def _relationship_names_from_edge_subgraph(self, edge_subgraph: list[dict[str, Any]]) -> list[str]:
        """Collect all readable edge labels present in an edge subgraph."""
        return dedupe_preserve_order(
            [self._relationship_name_from_edge(edge) for edge in (edge_subgraph or [])]
        )




    def _emit_retrieval_stage_trace(
        self,
        *,
        request_id: str | None,
        question: str,
        low_level_keywords: list[str],
        high_level_keywords: list[str],
        local_branch: list[dict[str, Any]],
        global_branch: list[dict[str, Any]],
        merged_branch: list[dict[str, Any]],
        graph_override: bool,
        stop_reason: str | None = None,
        elapsed_sec: float | None = None,
    ) -> None:
        """Write one single-file retrieval waterfall trace for the current request."""
        self._last_stage_trace = {
            "request_id": request_id,
            "question": question,
            "low_level_keywords": list(low_level_keywords or []),
            "high_level_keywords": list(high_level_keywords or []),
            "graph_override": graph_override,
            "stop_reason": stop_reason,
            "elapsed_sec": elapsed_sec,
            "branches": {
                "local": local_branch,
                "global": global_branch,
                "merged": merged_branch,
            },
        }
        waterfall_text = format_retrieval_stage_trace_text(
            request_id=request_id,
            question=question,
            low_level_keywords=low_level_keywords,
            high_level_keywords=high_level_keywords,
            local_branch=local_branch,
            global_branch=global_branch,
            merged_branch=merged_branch,
            graph_override=graph_override,
            stop_reason=stop_reason,
            elapsed_sec=elapsed_sec,
        )
        _trace_jlog(
            "retrieval_stage_trace",
            request_id,
            question=question,
            low_level_keywords=low_level_keywords,
            high_level_keywords=high_level_keywords,
            graph_override=graph_override,
            stop_reason=stop_reason,
            elapsed_sec=elapsed_sec,
            branches={
                "local": local_branch,
                "global": global_branch,
                "merged": merged_branch,
            },
        )
        _trace_pretty_log.info(waterfall_text)



    def _search_candidates(
        self,
        *,
        question: str,
        low_level_keywords: list[str],
        high_level_keywords: list[str],
        query_vec,
        ent_topk: int,
        rel_topk: int,
        ent_threshold: float,
        rel_threshold: float,
        graph,
        request_id: str | None,
        timer_total,
        local_branch: list,
        global_branch: list,
        append_trace,
        emit_trace,
    ) -> "CandidateSet | None":
        """Stage 1: everything that puts a candidate into the pool.

        Hybrid entity search, spreading activation, node expansion, lexical
        relationship search, edge expansion. Nothing downstream can recover a
        candidate that does not leave here, which is why loosening a threshold
        or a top-k in this stage is the first move when gold evidence is
        missing entirely rather than merely ranked badly.

        The trace callbacks are parameters rather than closures reached
        through self, because they carry the caller's per-request state.

        Returns None when the pool comes out empty, which the caller turns back
        into the empty result the whole query used to short-circuit to. That
        short circuit was a bare `return` inside this stage before it was a
        stage; making it a None keeps the early exit without letting this
        method decide what the query returns.
        """
        # 1) Search entities and relationships
        _jlog(
            "entity_hybrid_search_start",
            request_id,
            step="2.1",
            low_level_keywords=low_level_keywords,
            ent_topk=ent_topk,
            ent_threshold=ent_threshold,
        )
        ent_hits = self.searcher.search_entities_hybrid(
            query_vec=query_vec,
            low_level_keywords=low_level_keywords,
            entity_vec_threshold=ent_threshold,
            entity_top_k=ent_topk,
            request_id=request_id,
        )

        ent_ids = list({meta["id"] for hits in ent_hits.values() for meta, _ in hits})
        _jlog(
            "entity_hit_ids_collected",
            request_id,
            step="2.1",
            count=len(ent_ids),
            sample=ent_ids[:20],
            sample_names=self._entity_names_from_ids(ent_ids[:20]),
            hit_sources={source: len(hits) for source, hits in ent_hits.items()},
        )
        append_trace(
            local_branch,
            step="2.1",
            stage="entity_seeds",
            entity_names=self._entity_names_from_ids(ent_ids),
            relationship_names=[],
        )

        if not ent_ids:
            append_trace(
                local_branch,
                step="2.2",
                stage="node_subgraph",
                entity_names=[],
                relationship_names=[],
                skipped=True,
                reason="no_entity_ids",
            )
            _jlog("node_subgraph_skipped", request_id, step="2.2", reason="no_entity_ids")
            emit_trace(stop_reason="no_entity_hits")
            _jlog(
                "assemble_context_from_query_complete",
                request_id,
                step="2",
                entity_count=0,
                relationship_count=0,
                context_length=0,
                reason="no_entity_hits",
                elapsed_sec=timer_total.sec(),
            )
            return None

        # SA-RAG: optionally expand seed entities via spreading activation over the graph.
        if self.cfg.use_spreading_activation:
            seed_ent_ids = list(ent_ids)
            activated = self.sa_engine.run(
                seed_entity_ids=seed_ent_ids,
                query_vec=query_vec,
                request_id=request_id,
            )
            activated_ids = list(activated.keys())
            ent_ids = list(dict.fromkeys(seed_ent_ids + activated_ids))
            append_trace(
                local_branch,
                step="2.1.5",
                stage="sa_activated_seeds",
                entity_names=self._entity_names_from_ids(ent_ids),
                relationship_names=[],
            )
            _jlog(
                "sa_rag_expanded",
                request_id,
                step="2.1.5",
                seed_count=len(seed_ent_ids),
                activated_count=len(activated_ids),
                union_count=len(ent_ids),
                activated_names=self._entity_names_from_ids(activated_ids[:20]),
                union_names=self._entity_names_from_ids(ent_ids[:20]),
                top5=[(eid, round(sc, 4)) for eid, sc in list(activated.items())[:5]],
                top5_names=[
                    {"name": self._entity_name_by_id(eid), "score": round(sc, 4)}
                    for eid, sc in list(activated.items())[:5]
                ],
            )
        else:
            append_trace(
                local_branch,
                step="2.1.5",
                stage="sa_activated_seeds",
                entity_names=self._entity_names_from_ids(ent_ids),
                relationship_names=[],
                skipped=True,
                reason="disabled",
            )
            _jlog("sa_rag_skipped", request_id, step="2.1.5", reason="disabled")

        # Fetch node subgraph
        timer_sub_nodes = _StepTimer()
        _jlog(
            "node_subgraph_fetch_start",
            request_id,
            step="2.2",
            entity_count=len(ent_ids),
            sample_entity_ids=ent_ids[:20],
            sample_entity_names=self._entity_names_from_ids(ent_ids[:20]),
        )
        node_subgraph = graph.get_node_subgraph(ent_ids) or {}
        _jlog(
            "node_subgraph_fetched",
            request_id,
            step="2.2",
            elapsed_sec=timer_sub_nodes.sec(),
            node_count=len(node_subgraph or {}),
            sample_node_names=self._entity_names_from_ids(list((node_subgraph or {}).keys())[:20]),
        )

        # Detect partial miss: vector hits present but graph missing some IDs
        if node_subgraph:
            missing_from_graph = [eid for eid in ent_ids if eid not in node_subgraph]
            if missing_from_graph:
                _jlog(
                    "graph_index_mismatch_suspect",
                    request_id,
                    step="2.2",
                    requested_id_count=len(ent_ids),
                    returned_node_count=len(node_subgraph),
                    missing_id_count=len(missing_from_graph),
                    missing_id_sample=missing_from_graph[:10],
                )
                ent_id2meta_partial, _ = build_id_to_meta_maps(self.cache)
                for eid in missing_from_graph:
                    meta = ent_id2meta_partial.get(eid, {})
                    if meta:
                        node_subgraph[eid] = {
                            "self": {
                                "id": eid,
                                "name": meta.get("name"),
                                "type": meta.get("type"),
                                "desc": meta.get("description"),
                            },
                            "neighbors": [],
                        }

        if not node_subgraph:
            # Graph has no nodes for these IDs (graph not populated / not restored).
            # Fall back to entity metadata from the cache so VDB hits are not lost.
            ent_id2meta_fb, _ = build_id_to_meta_maps(self.cache)
            for eid in ent_ids:
                meta = ent_id2meta_fb.get(eid, {})
                if meta:
                    node_subgraph[eid] = {
                        "self": {
                            "id": eid,
                            "name": meta.get("name"),
                            "type": meta.get("type"),
                            "desc": meta.get("description"),
                        },
                        "neighbors": [],
                    }
            if not node_subgraph:
                append_trace(
                    local_branch,
                    step="2.2",
                    stage="node_subgraph",
                    entity_names=[],
                    relationship_names=[],
                    skipped=True,
                    reason="empty_after_cache_fallback",
                )
                _jlog("node_subgraph_empty", request_id, step="2.2")
                emit_trace(stop_reason="node_subgraph_empty")
                _jlog(
                    "assemble_context_from_query_complete",
                    request_id,
                    step="2",
                    entity_count=0,
                    relationship_count=0,
                    context_length=0,
                    reason="node_subgraph_empty",
                    elapsed_sec=timer_total.sec(),
                )
                return None
            _jlog(
                "node_subgraph_cache_fallback",
                request_id,
                step="2.2",
                entity_count=len(node_subgraph),
                entity_names=self._entity_names_from_ids(list(node_subgraph.keys())[:20]),
            )
        append_trace(
            local_branch,
            step="2.2",
            stage="node_subgraph",
            entity_names=self._entity_names_from_node_subgraph(node_subgraph),
            relationship_names=self._relationship_names_from_node_subgraph(node_subgraph),
        )

        # Select keyword source for relationship vector search.
        # Abstract reasoning words (high_level) match relationships poorly; concrete
        # anchors (low_level) are usually better. Configurable for ablation.
        _rel_kw_mode = getattr(self.cfg, "relation_search_keywords", "high_level")
        if _rel_kw_mode == "low_level":
            rel_search_keywords = list(low_level_keywords or [])
        elif _rel_kw_mode == "both":
            rel_search_keywords = list(dict.fromkeys(list(high_level_keywords or []) + list(low_level_keywords or [])))
        else:
            rel_search_keywords = list(high_level_keywords or [])

        # Search relationships
        _jlog(
            "relationship_search_start",
            request_id,
            step="2.3",
            rel_kw_mode=_rel_kw_mode,
            rel_search_keywords=rel_search_keywords,
            high_level_keywords=high_level_keywords,
            rel_topk=rel_topk,
            rel_threshold=rel_threshold,
        )
        rel_hits = self.searcher.search_relationships_by_vec(
            keywords=rel_search_keywords,
            relationship_top_k=rel_topk,
            relationship_vec_threshold=rel_threshold,
            request_id=request_id,
        )

        rel_ids = list({meta["id"] for hits in rel_hits.values() for meta, _ in hits})
        _jlog(
            "relationship_hit_ids_collected",
            request_id,
            step="2.3",
            count=len(rel_ids),
            sample=rel_ids[:20],
            sample_names=self._relationship_names_from_ids(rel_ids[:20]),
            hit_sources={source: len(hits) for source, hits in rel_hits.items()},
        )
        append_trace(
            global_branch,
            step="2.3",
            stage="relationship_seeds",
            entity_names=self._entity_names_from_relationship_ids(rel_ids),
            relationship_names=self._relationship_names_from_ids(rel_ids),
        )

        # Fetch edge subgraph
        timer_sub_edges = _StepTimer()
        if rel_ids:
            _jlog(
                "edge_subgraph_fetch_start",
                request_id,
                step="2.4",
                relationship_count=len(rel_ids),
                sample_relationship_ids=rel_ids[:20],
                sample_relationship_names=self._relationship_names_from_ids(rel_ids[:20]),
            )
        else:
            _jlog("edge_subgraph_skipped", request_id, step="2.4", reason="no_relationship_ids")
        edge_subgraph = (graph.get_edge_subgraph(rel_ids) or []) if rel_ids else []
        _jlog(
            "edge_subgraph_fetched",
            request_id,
            step="2.4",
            elapsed_sec=timer_sub_edges.sec(),
            edge_count=len(edge_subgraph or []),
            sample_edge_names=self._relationship_names_from_ids(
                [edge.get("rel_id") for edge in (edge_subgraph or []) if edge.get("rel_id")][:20]
            ),
        )
        append_trace(
            global_branch,
            step="2.4",
            stage="edge_subgraph",
            entity_names=self._entity_names_from_edge_subgraph(edge_subgraph),
            relationship_names=self._relationship_names_from_edge_subgraph(edge_subgraph),
            skipped=not rel_ids,
            reason="no_relationship_ids" if not rel_ids else None,
        )


        return CandidateSet(node_subgraph=node_subgraph, edge_subgraph=edge_subgraph)

    def _filter_candidates(
        self,
        *,
        intersect_entity_ids,
        intersect_rel_ids,
        filter_ent_topk,
        filter_rel_topk,
        request_id: str | None,
        merged_branch: list,
        append_trace,
    ) -> tuple:
        """Stage 3: pass the intersected pool through to the reranker.

        It does no cutting of its own: everything that survives the intersection
        goes forward, sorted so the order the reranker sees does not depend on
        set iteration -- which is the whole of its policy. The two topk values are
        taken only to be logged as requested-and-ignored.

        The metadata maps are built here rather than in stage 4 because they are
        also what the trace renders.

        Returns:
            (filtered_entity_ids, filtered_rel_ids, ent_id2meta, rel_id2meta)
        """
        # 3) Filter candidates (step 2.6)
        timer_filter = _StepTimer()
        # The two topk values are logged as what they are -- requested and not
        # applied -- so a trace reader cannot mistake them for the cut that
        # produced `filtered_entities` below.
        _jlog(
            "filter_step_start",
            request_id,
            step="2.6",
            entity_candidate_count=len(intersect_entity_ids),
            relationship_candidate_count=len(intersect_rel_ids),
            requested_ignored_filter_ent_topk=filter_ent_topk,
            requested_ignored_filter_rel_topk=filter_rel_topk,
        )


        # The reranker is the filter: everything that survived the intersection
        # goes forward, and stage 4 scores and cuts it. Sorted so the order the
        # reranker sees does not depend on set iteration.
        filtered_entity_ids = sorted(intersect_entity_ids)
        filtered_rel_ids = sorted(intersect_rel_ids)

        _jlog(
            "filter_step_done",
            request_id,
            step="2.6",
            filtered_entities=len(filtered_entity_ids),
            filtered_rels=len(filtered_rel_ids),
            elapsed_sec=timer_filter.sec(),
        )
        append_trace(
            merged_branch,
            step="2.6",
            stage="filtered",
            entity_names=self._entity_names_from_ids(filtered_entity_ids),
            relationship_names=self._relationship_names_from_ids(filtered_rel_ids),
        )

        # Convert IDs to full metadata dicts. Stage 4 renders the entity and
        # relationship records from these, over the ids the reranker returns --
        # building them here as well only produced a list that was overwritten
        # before anything read it.
        ent_id2meta, rel_id2meta = build_id_to_meta_maps(self.cache)

        return filtered_entity_ids, filtered_rel_ids, ent_id2meta, rel_id2meta

    def assemble_context_from_query(
        self,
        question: str,
        low_level_keywords: list[str],
        high_level_keywords: list[str],
        request_id: str | None = None,
        ent_topk: int | None = None,
        rel_topk: int | None = None,
        ent_threshold: float | None = None,
        rel_threshold: float | None = None,
        filter_ent_topk: int | None = None,
        filter_rel_topk: int | None = None,
        filter_ent_threshold: float | None = None,
        filter_rel_threshold: float | None = None,
        query_time: str | None = None,
        _graph: Any = None,
    ) -> tuple[list[dict], list[dict], str, np.ndarray]:
        """
        Assemble KG context from query using modular components.

        Returns:
            (context_entities, context_relationships, context_text, query_vec)
        """
        timer_total = _StepTimer()
        # Resolve params from config
        ent_topk = ent_topk if ent_topk is not None else self.cfg.ent_topk
        rel_topk = rel_topk if rel_topk is not None else self.cfg.rel_topk
        ent_threshold = ent_threshold if ent_threshold is not None else self.cfg.ent_threshold
        rel_threshold = rel_threshold if rel_threshold is not None else self.cfg.rel_threshold
        filter_ent_topk = filter_ent_topk if filter_ent_topk is not None else self.cfg.filter_ent_topk
        filter_rel_topk = filter_rel_topk if filter_rel_topk is not None else self.cfg.filter_rel_topk
        filter_ent_threshold = filter_ent_threshold if filter_ent_threshold is not None else self.cfg.filter_ent_threshold
        filter_rel_threshold = filter_rel_threshold if filter_rel_threshold is not None else self.cfg.filter_rel_threshold

        # Resolve graph: caller may supply a local graph override for adaptive pass-2
        graph = _graph if _graph is not None else self.graph
        local_branch: list[dict[str, Any]] = []
        global_branch: list[dict[str, Any]] = []
        merged_branch: list[dict[str, Any]] = []

        def append_trace(
            branch: list[dict[str, Any]],
            *,
            step: str,
            stage: str,
            entity_names: list[str],
            relationship_names: list[str],
            skipped: bool = False,
            reason: str | None = None,
        ) -> None:
            """Append one step record to the retrieval trace."""
            previous = branch[-1] if branch else None
            branch.append(
                build_stage_trace_snapshot(
                    step=step,
                    stage=stage,
                    entity_names=entity_names,
                    relationship_names=relationship_names,
                    previous=previous,
                    skipped=skipped,
                    reason=reason,
                )
            )

        def emit_trace(stop_reason: str | None = None) -> None:
            """Write the accumulated trace out as a structured log event."""
            self._emit_retrieval_stage_trace(
                request_id=request_id,
                question=question,
                low_level_keywords=low_level_keywords,
                high_level_keywords=high_level_keywords,
                local_branch=local_branch,
                global_branch=global_branch,
                merged_branch=merged_branch,
                graph_override=bool(_graph is not None),
                stop_reason=stop_reason,
                elapsed_sec=timer_total.sec(),
            )

        _jlog(
            "assemble_context_from_query_start",
            request_id,
            step="2",
            question=question,
            low_level_keywords=low_level_keywords,
            high_level_keywords=high_level_keywords,
            ent_topk=ent_topk,
            rel_topk=rel_topk,
            ent_threshold=ent_threshold,
            rel_threshold=rel_threshold,
            filter_ent_topk=filter_ent_topk,
            filter_rel_topk=filter_rel_topk,
            filter_ent_threshold=filter_ent_threshold,
            filter_rel_threshold=filter_rel_threshold,
            query_time=query_time,
            graph_override=bool(_graph is not None),
        )

        # 0) Embed query
        query_vec = self.searcher.embed_query(question, request_id=request_id)

        # 1) Search entities and relationships
        _candidates = self._search_candidates(
            question=question,
            low_level_keywords=low_level_keywords,
            high_level_keywords=high_level_keywords,
            query_vec=query_vec,
            ent_topk=ent_topk,
            rel_topk=rel_topk,
            ent_threshold=ent_threshold,
            rel_threshold=rel_threshold,
            graph=graph,
            request_id=request_id,
            timer_total=timer_total,
            local_branch=local_branch,
            global_branch=global_branch,
            append_trace=append_trace,
            emit_trace=emit_trace,
        )
        if _candidates is None:
            return [], [], "", query_vec
        node_subgraph = _candidates.node_subgraph
        edge_subgraph = _candidates.edge_subgraph

        # 2) Compute intersection (using union for now as per original code)
        intersect_entity_ids, intersect_rel_ids = self.evidence_filter.compute_subgraph_intersection(
            node_subgraph=node_subgraph,
            edge_subgraph=edge_subgraph,
            use_union=True,  # Original code uses union
            request_id=request_id,
        )
        append_trace(
            merged_branch,
            step="2.5",
            stage="union_candidates",
            entity_names=self._entity_names_from_ids(sorted(intersect_entity_ids)),
            relationship_names=self._relationship_names_from_ids(sorted(intersect_rel_ids)),
        )

        if not intersect_entity_ids and not intersect_rel_ids:
            _jlog("intersection_empty", request_id, step="2.5")
            emit_trace(stop_reason="intersection_empty")
            _jlog(
                "assemble_context_from_query_complete",
                request_id,
                step="2",
                entity_count=0,
                relationship_count=0,
                context_length=0,
                reason="intersection_empty",
                elapsed_sec=timer_total.sec(),
            )
            return [], [], "", query_vec

        # 3) Filter candidates (step 2.6)
        (
            filtered_entity_ids,
            filtered_rel_ids,
            ent_id2meta,
            rel_id2meta,
        ) = self._filter_candidates(
            intersect_entity_ids=intersect_entity_ids,
            intersect_rel_ids=intersect_rel_ids,
            filter_ent_topk=filter_ent_topk,
            filter_rel_topk=filter_rel_topk,
            request_id=request_id,
            merged_branch=merged_branch,
            append_trace=append_trace,
        )

        # 4) Optional reranker recovery / reranker-only selection
        # The reranker IS the filter: it scores every candidate that survived the
        # intersection and selects the top-K. Stage 3 does no cutting of its own.
        # It is handed stage 3's sorted lists, not the raw intersection sets: the
        # top-K cut is `passing[:top_k]` over a stable sort, so whenever scores tie
        # -- which they do wholesale on the API reranker's no-logprobs fallback,
        # where every doc is +/-1.0 -- the survivors are decided by input order. A
        # set there makes Retrieved_Context depend on PYTHONHASHSEED.
        selected_entity_ids, selected_rel_ids = self.evidence_filter.rerank_filter(
            question=question,
            entity_ids=filtered_entity_ids,
            relationship_ids=filtered_rel_ids,
            entity_top_k=self.cfg.rrk_ent_topk,
            relationship_top_k=self.cfg.rrk_rel_topk,
            threshold=self.cfg.rrk_threshold,
            request_id=request_id,
        )

        # Render the selection. There is no skip path: rerank_filter always returns
        # two lists, so the "reranker disabled" branch that used to stand here --
        # and the reranker_skipped trace state it emitted -- could never be reached.
        filtered_entities = []
        for eid in selected_entity_ids:
            meta = ent_id2meta.get(eid, {})
            if meta:
                filtered_entities.append({
                    "id": eid,
                    "name": meta.get("name"),
                    "type": meta.get("type"),
                    "desc": meta.get("description"),
                })

        filtered_rels = []
        for rid in selected_rel_ids:
            meta = rel_id2meta.get(rid, {})
            if meta:
                src_id = meta.get("source_id")
                tgt_id = meta.get("target_id")
                src_meta = ent_id2meta.get(src_id, {}) if isinstance(src_id, str) else {}
                tgt_meta = ent_id2meta.get(tgt_id, {}) if isinstance(tgt_id, str) else {}

                filtered_rels.append({
                    "rel_id": rid,
                    "rel_desc": meta.get("description"),
                    "rel_keywords": meta.get("keywords"),
                    "source_id": src_id,
                    "source_name": src_meta.get("name"),
                    "source_type": src_meta.get("type"),
                    "target_id": tgt_id,
                    "target_name": tgt_meta.get("name"),
                    "target_type": tgt_meta.get("type"),
                })
        append_trace(
            merged_branch,
            step="2.7",
            stage="reranker_final",
            entity_names=self._entity_names_from_ids(sorted(selected_entity_ids)),
            relationship_names=self._relationship_names_from_ids(sorted(selected_rel_ids)),
        )

        # 4b) Temporal containment boost: surface coarse-grained entities whose
        # stored range contains the parsed query date (MONTH/WEEK/SEASON/YEAR).
        # Ablation H: KG_ABLATION_NO_TEMPORAL_BOOST=1 -> skip this reranking.
        # Note: LoCoMo passes no query_time, so this block is structurally dead
        # there; H only means anything for LongMem.
        if flag_enabled("KG_ABLATION_NO_TEMPORAL_BOOST"):
            _jlog(
                "ablation_no_temporal_boost_applied",
                request_id,
                step="4b",
                has_query_time=bool(query_time),
            )
        elif query_time:
            try:
                _qdt = parse_query_time(query_time)
                if _qdt is not None:
                    _qdate = _qdt.date()
                    _ent_id2meta_boost, _ = build_id_to_meta_maps(self.cache)

                    def _containment_key(ent: dict) -> int:
                        _meta = _ent_id2meta_boost.get(ent.get("id", ""), {})
                        _tmeta = _meta.get("temporal") or {}
                        return 0 if date_within_coarse_range(_qdate, _tmeta) else 1

                    filtered_entities.sort(key=_containment_key)
                    _jlog(
                        "temporal_containment_boost",
                        request_id,
                        step="4b",
                        query_date=_qdate.isoformat(),
                        boosted_count=sum(
                            1 for e in filtered_entities
                            if _containment_key(e) == 0
                        ),
                    )
            except Exception:
                pass

        # 5) Render context text
        context_text = render_context_text(
            cache=self.cache,
            entities=filtered_entities,
            relationships=filtered_rels,
            request_id=request_id,
        )

        _jlog(
            "assemble_context_from_query_complete",
            request_id,
            step="2",
            entity_count=len(filtered_entities),
            relationship_count=len(filtered_rels),
            entity_names=[entity.get("name") or entity.get("id") for entity in filtered_entities[:20]],
            relationship_names=[
                (
                    f"{rel.get('source_name') or rel.get('source_id')} -> "
                    f"{rel.get('target_name') or rel.get('target_id')}"
                )
                if not (rel.get("rel_desc") or "").strip()
                else (
                    f"{rel.get('source_name') or rel.get('source_id')} -> "
                    f"{rel.get('target_name') or rel.get('target_id')} | {rel.get('rel_desc')}"
                )
                for rel in filtered_rels[:20]
            ],
            context_length=len(context_text),
            elapsed_sec=timer_total.sec(),
        )
        emit_trace()

        return filtered_entities, filtered_rels, context_text, query_vec




    # Stays a method rather than joining retrieval_steps/adaptive.py with
    # additive_merge: it calls self.assemble_context_from_query to run the
    # second pass. Moving it out would mean passing the Retriever in, which is
    # a circular dependency wearing a parameter's clothes.
    def _adaptive_research(
        self,
        *,
        question: str,
        evidence_entities: list[dict],
        evidence_rels: list[dict],
        evidence_text: str,
        query_vec: np.ndarray,
        request_id: str | None,
        ent_topk: int,
        rel_topk: int,
        ent_threshold: float,
        rel_threshold: float,
        filter_ent_topk: int,
        filter_rel_topk: int,
        filter_ent_threshold: float,
        filter_rel_threshold: float,
        query_time: str | None,
    ) -> tuple[list[dict], list[dict], str, Any]:
        """
        Post-retrieval adaptive re-search (pass 2 of at most 2 total).

        Computes confidence from pass-1 results.  If conf < tau_confidence, rewrites
        the query via an LLM and runs a second retrieval pass with relaxed thresholds.
        Merges pass-1 and pass-2 candidates (deduplicated), re-ranks all of them
        against the original query_vec, and returns the top-K.  This avoids the
        winner-take-all bias where pass-2's larger candidate pool inflated its
        confidence score.

        LLM used for rewriting: LLM_API / MODEL_NAME (from .env).
        """
        from grace_mem.retrieval.steps.adaptive import (
            build_adaptive_graph,
            build_adaptive_llm_client,
            compute_confidence,
            rewrite_query,
        )
        timer_adaptive = _StepTimer()

        _jlog(
            "adaptive_research_start",
            request_id,
            step="2b",
            entity_count=len(evidence_entities),
            relationship_count=len(evidence_rels),
            tau_confidence=self.cfg.tau_confidence,
            adaptive_threshold_scale=self.cfg.adaptive_threshold_scale,
        )

        # --- Pass-1 confidence ---
        ent_ids_1 = [e["id"] for e in evidence_entities]
        rel_ids_1 = [r["rel_id"] for r in evidence_rels]
        conf_1 = compute_confidence(ent_ids_1, rel_ids_1, query_vec, self.MGR)

        _jlog(
            "adaptive_confidence_pass1",
            request_id,
            step="2b",
            confidence=conf_1,
            tau=self.cfg.tau_confidence,
            entity_count=len(ent_ids_1),
            rel_count=len(rel_ids_1),
        )

        if conf_1 >= self.cfg.tau_confidence:
            _jlog("adaptive_skip", request_id, step="2b", reason="confidence_sufficient")
            _jlog(
                "adaptive_research_complete",
                request_id,
                step="2b",
                pass2_triggered=False,
                conf_pass1=conf_1,
                conf_final=conf_1,
                elapsed_sec=timer_adaptive.sec(),
            )
            self._last_adaptive_trace = build_adaptive_trace(
            config=self.cfg,
                pass2_triggered=False,
                pass1_entity_ids=ent_ids_1,
                pass1_relation_ids=rel_ids_1,
                conf_pass1=conf_1,
                conf_final=conf_1,
            )
            return evidence_entities, evidence_rels, evidence_text, query_vec

        # --- Rewrite query ---
        try:
            rewrite_llm = build_adaptive_llm_client()
            rewritten_q, rewrite_latency = rewrite_query(
                question, evidence_entities, evidence_rels, conf_1, rewrite_llm
            )
            _jlog(
                "adaptive_query_rewrite",
                request_id,
                step="2b",
                original_query=question,
                rewritten_query=rewritten_q,
                rewrite_latency_sec=rewrite_latency,
            )
        except Exception as exc:
            _jlog("adaptive_rewrite_error", request_id, step="2b", error=str(exc))
            rewritten_q = question

        # Skip pass-2 if the rewrite is identical to the original query — no new signal possible
        if rewritten_q.strip() == question.strip():
            _jlog("adaptive_skip", request_id, reason="rewrite_identical")
            print("[Adaptive] Rewrite returned original query — skipping pass-2.")
            self._last_adaptive_trace = build_adaptive_trace(
            config=self.cfg,
                pass2_triggered=False,
                pass1_entity_ids=ent_ids_1,
                pass1_relation_ids=rel_ids_1,
                conf_pass1=conf_1,
                conf_final=conf_1,
                rewritten_query=rewritten_q,
                adaptive_skip_reason="rewrite_identical",
            )
            return evidence_entities, evidence_rels, evidence_text, query_vec

        # --- Pass-2 graph ---
        local_graph = None
        try:
            local_graph = build_adaptive_graph()
            _jlog("adaptive_graph_opened", request_id, step="2b")
        except OSError as exc:
            _jlog("adaptive_graph_error", request_id, step="2b", error=str(exc))

        try:
            # --- Pass-2 keywords ---
            kw2 = generate_query_keywords(llm=self.llm, question=rewritten_q, request_id=request_id)

            # --- Pass-2 retrieval with relaxed filter thresholds ---
            scale = self.cfg.adaptive_threshold_scale
            timer_p2 = _StepTimer()
            _jlog(
                "adaptive_pass2_start",
                request_id,
                step="2b",
                rewritten_query=rewritten_q,
                filter_ent_threshold_scaled=filter_ent_threshold * scale,
                filter_rel_threshold_scaled=filter_rel_threshold * scale,
                graph_override=bool(local_graph is not None),
            )
            evidence2_entities, evidence2_rels, evidence2_text, query_vec2 = self.assemble_context_from_query(
                question=rewritten_q,
                low_level_keywords=kw2.low_level_keywords,
                high_level_keywords=kw2.high_level_keywords,
                request_id=request_id,
                ent_topk=ent_topk,
                rel_topk=rel_topk,
                ent_threshold=ent_threshold,
                rel_threshold=rel_threshold,
                filter_ent_topk=filter_ent_topk,
                filter_rel_topk=filter_rel_topk,
                filter_ent_threshold=filter_ent_threshold * scale,
                filter_rel_threshold=filter_rel_threshold * scale,
                query_time=query_time,
                _graph=local_graph,
            )
        finally:
            if local_graph is not None:
                local_graph.close()

        ent_ids_2 = [e["id"] for e in evidence2_entities]
        rel_ids_2 = [r["rel_id"] for r in evidence2_rels]
        conf_2 = compute_confidence(ent_ids_2, rel_ids_2, query_vec, self.MGR)

        _jlog(
            "adaptive_confidence_pass2",
            request_id,
            step="2b",
            confidence=conf_2,
            entity_count=len(ent_ids_2),
            rel_count=len(rel_ids_2),
            elapsed_sec=timer_p2.sec(),
        )
        _jlog(
            "adaptive_pass2_retrieval_done",
            request_id,
            step="2b",
            entity_count=len(ent_ids_2),
            relationship_count=len(rel_ids_2),
            context_length=len(evidence2_text),
            query_vec_dim=int(query_vec2.shape[0]) if hasattr(query_vec2, "shape") else None,
            elapsed_sec=timer_p2.sec(),
        )

        # --- Additive merge: keep all pass-1 context, append only novel pass-2 items ---
        merged_entities, merged_rels, merged_text, conf_merged = additive_merge(vdb_manager=self.MGR, cache=self.cache, cfg=self.cfg, 
            entities_1=evidence_entities,
            rels_1=evidence_rels,
            entities_2=evidence2_entities,
            rels_2=evidence2_rels,
            request_id=request_id,
            conf_1=conf_1,
            conf_2=conf_2,
            query_vec=query_vec,
        )
        _jlog(
            "adaptive_research_complete",
            request_id,
            step="2b",
            pass2_triggered=True,
            conf_pass1=conf_1,
            conf_pass2=conf_2,
            conf_final=conf_merged,
            merged_entity_count=len(merged_entities),
            merged_relationship_count=len(merged_rels),
            elapsed_sec=timer_adaptive.sec(),
        )
        self._last_adaptive_trace = build_adaptive_trace(
            config=self.cfg,
            pass2_triggered=True,
            pass1_entity_ids=ent_ids_1,
            pass1_relation_ids=rel_ids_1,
            pass2_entity_ids=ent_ids_2,
            pass2_relation_ids=rel_ids_2,
            conf_pass1=conf_1,
            conf_pass2=conf_2,
            conf_final=conf_merged,
            rewritten_query=rewritten_q,
        )
        return merged_entities, merged_rels, merged_text, query_vec


    def build_kg_context(
        self,
        question: str,
        *,
        ent_topk: int | None = None,
        rel_topk: int | None = None,
        ent_threshold: float | None = None,
        rel_threshold: float | None = None,
        filter_ent_topk: int | None = None,
        filter_rel_topk: int | None = None,
        filter_ent_threshold: float | None = None,
        filter_rel_threshold: float | None = None,
        summary_topk_per_item: int | None = None,
        summary_vec_threshold: float | None = None,
        query_time: str | None = None,
        top_k: int | None = None,
    ) -> str:
        """
        Main entry point: Build complete KG context with evidence.

        Args:
            question: User query
            ent_topk: Top-K entities in initial search
            rel_topk: Top-K relationships in initial search
            ent_threshold: Entity similarity threshold
            rel_threshold: Relationship similarity threshold
            filter_ent_topk: Top-K entities after filtering
            filter_rel_topk: Top-K relationships after filtering
            filter_ent_threshold: Entity similarity threshold for filtering
            filter_rel_threshold: Relationship similarity threshold for filtering
            summary_topk_per_item: Max evidence snippets
            summary_vec_threshold: Evidence similarity threshold
            query_time: Query timestamp for temporal relevance
            top_k: Deprecated parameter for backward compatibility

        Returns:
            Complete KG context string with evidence
        """
        request_id = str(uuid.uuid4())
        timer_total = _StepTimer()

        # Resolve params from config
        ent_topk = ent_topk if ent_topk is not None else self.cfg.ent_topk
        rel_topk = rel_topk if rel_topk is not None else self.cfg.rel_topk
        ent_threshold = ent_threshold if ent_threshold is not None else self.cfg.ent_threshold
        rel_threshold = rel_threshold if rel_threshold is not None else self.cfg.rel_threshold
        filter_ent_topk = filter_ent_topk if filter_ent_topk is not None else self.cfg.filter_ent_topk
        filter_rel_topk = filter_rel_topk if filter_rel_topk is not None else self.cfg.filter_rel_topk
        filter_ent_threshold = filter_ent_threshold if filter_ent_threshold is not None else self.cfg.filter_ent_threshold
        filter_rel_threshold = filter_rel_threshold if filter_rel_threshold is not None else self.cfg.filter_rel_threshold
        summary_topk_per_item = summary_topk_per_item if summary_topk_per_item is not None else self.cfg.summary_topk_per_item
        summary_vec_threshold = summary_vec_threshold if summary_vec_threshold is not None else self.cfg.summary_vec_threshold

        # Handle deprecated top_k parameter
        deprecated_topk_used = False
        if top_k is not None:
            ent_topk = top_k
            rel_topk = top_k
            deprecated_topk_used = True

        # [LOG] Request start
        _jlog(
            "build_kg_context_start",
            request_id,
            step="0",
            question=question,
            query_time=query_time,
            deprecated_topk_used=deprecated_topk_used,
            ent_topk=ent_topk,
            rel_topk=rel_topk,
            ent_threshold=ent_threshold,
            rel_threshold=rel_threshold,
            filter_ent_topk=filter_ent_topk,
            filter_rel_topk=filter_rel_topk,
            filter_ent_threshold=filter_ent_threshold,
            filter_rel_threshold=filter_rel_threshold,
            evidence_max_items_per=summary_topk_per_item,
            summary_vec_threshold=summary_vec_threshold,
        )
        self.last_retrieval_trace = {
            "request_id": request_id,
            "question": question,
        }
        self._last_stage_trace = {}
        self._last_adaptive_trace = {}
        self.evidence_builder.last_evidence_trace = {}

        try:
            # 0b) Rewrite relative temporal expressions in the question
            rewritten_question = maybe_rewrite_retrieval_question(
                question,
                query_time,
                request_id,
            )

            # 0c) Dynamic retrieval planning is deliberately off: the extra LLM call
            # made runs non-reproducible, and every downstream consumer already
            # treats guidance as optional. Keeping the variable (rather than
            # deleting the step) preserves the None-guidance code path that the
            # keyword and evidence stages branch on.
            retrieval_guidance = None

            # 1) Extract keywords
            # Ablation L: KG_ABLATION_NO_KEYWORDS=1 -> skip LLM keyword extraction.
            # Downstream degrades on its own: BM25 has no input so it does not run
            # (entities are left with the query-vector path only), and relation
            # vector search returns {} for empty keywords (empty edge subgraph).
            # Relationship candidates still arrive via the node subgraph's incident
            # edges (local_rel_set), so the pool never drops to zero.
            if flag_enabled("KG_ABLATION_NO_KEYWORDS"):
                _jlog("ablation_no_keywords_applied", request_id, step="1")
                kw = KeywordExtractionResult(high_level_keywords=[], low_level_keywords=[])
            else:
                kw = generate_query_keywords(llm=self.llm,
                    question=rewritten_question, request_id=request_id,
                    retrieval_guidance=retrieval_guidance,
                )

            # 2) Assemble context (entities, relationships, text)
            timer_context = _StepTimer()
            evidence_entities, evidence_rels, evidence_text, query_vec = self.assemble_context_from_query(
                question=rewritten_question,
                low_level_keywords=kw.low_level_keywords,
                high_level_keywords=kw.high_level_keywords,
                request_id=request_id,
                ent_topk=ent_topk,
                rel_topk=rel_topk,
                ent_threshold=ent_threshold,
                rel_threshold=rel_threshold,
                filter_ent_topk=filter_ent_topk,
                filter_rel_topk=filter_rel_topk,
                filter_ent_threshold=filter_ent_threshold,
                filter_rel_threshold=filter_rel_threshold,
                query_time=query_time,
            )
            _jlog(
                "context_build_done",
                request_id,
                step="2",
                entity_count=len(evidence_entities),
                relationship_count=len(evidence_rels),
                has_context=bool(evidence_text),
                elapsed_sec=timer_context.sec(),
            )

            # 2b) Adaptive re-search (pass 2) — off by default
            if self.cfg.enable_adaptive_search:
                evidence_entities, evidence_rels, evidence_text, query_vec = self._adaptive_research(
                    question=question,
                    evidence_entities=evidence_entities,
                    evidence_rels=evidence_rels,
                    evidence_text=evidence_text,
                    query_vec=query_vec,
                    request_id=request_id,
                    ent_topk=ent_topk,
                    rel_topk=rel_topk,
                    ent_threshold=ent_threshold,
                    rel_threshold=rel_threshold,
                    filter_ent_topk=filter_ent_topk,
                    filter_rel_topk=filter_rel_topk,
                    filter_ent_threshold=filter_ent_threshold,
                    filter_rel_threshold=filter_rel_threshold,
                    query_time=query_time,
                )
            else:
                _jlog("adaptive_research_skipped", request_id, step="2b", reason="disabled")

            # 2.9b) Ablation J: KG_ABLATION_NO_KG_TEXT=1 -> remove only the entity/
            # relationship text blocks from the context. evidence_entities/evidence_rels are
            # kept (the evidence provenance channel is untouched), so the answering
            # model sees the Evidence block alone.
            if flag_enabled("KG_ABLATION_NO_KG_TEXT"):
                _jlog(
                    "ablation_no_kg_text_applied",
                    request_id,
                    step="2.9",
                    dropped_ctx_text_chars=len(evidence_text or ""),
                )
                evidence_text = ""

            # 2.9) Ablation B2 (no-KG baseline): drop the graph channel entirely --
            # the evidence pool is left with direct vector search alone, and the
            # context carries no entity/relationship text blocks either.
            # The env-var convention follows KG_NARROWING_ENABLED / USE_GREP_AGENT:
            # off by default.
            if flag_enabled("KG_ABLATION_NO_GRAPH"):
                _jlog(
                    "ablation_no_graph_applied",
                    request_id,
                    step="2.9",
                    dropped_entities=len(evidence_entities),
                    dropped_rels=len(evidence_rels),
                    dropped_ctx_text_chars=len(evidence_text or ""),
                )
                evidence_entities, evidence_rels, evidence_text = [], [], ""

            # 3) Build evidence block
            timer_evidence = _StepTimer()
            # Ablation A: KG_ABLATION_NO_DIRECT_VECTOR=1 -> config closes the direct
            # search channel down to topn=0 (add_direct becomes a no-op); all that is
            # left here is a signal the smoke test can assert on.
            if flag_enabled("KG_ABLATION_NO_DIRECT_VECTOR"):
                _jlog(
                    "ablation_no_direct_vector_applied",
                    request_id,
                    step="3",
                    summary_direct_vector_topn=self.cfg.summary_direct_vector_topn,
                    summary_direct_vector_min_score=self.cfg.summary_direct_vector_min_score,
                )
            evidence_block = self.evidence_builder.build_evidence_block(
                context_entities=evidence_entities,
                context_relationships=evidence_rels,
                summary_topk_global=summary_topk_per_item,
                query_vec=query_vec,
                summary_vec_threshold=summary_vec_threshold,
                use_full_summary=self.cfg.use_full_summary,
                fallback_to_raw=self.cfg.fallback_to_raw,
                use_raw_context=self.cfg.use_raw_context,
                use_split_embeddings=self.cfg.use_split_embeddings,
                summary_direct_vector_topn=self.cfg.summary_direct_vector_topn,
                summary_direct_vector_min_score=self.cfg.summary_direct_vector_min_score,
                summary_rerank_topk=self.cfg.summary_rerank_topk,
                summary_rerank_cosine_only=self.cfg.summary_rerank_cosine_only,
                split_single_entry_raw=self.cfg.split_single_entry_raw,
                query_text=rewritten_question,
                request_id=request_id,
                summary_per_entity_min=self.cfg.summary_per_entity_min,
            )
            _jlog(
                "evidence_render_done",
                request_id,
                step="3",
                has_evidence=bool(evidence_block),
                evidence_length=len(evidence_block) if evidence_block else 0,
                elapsed_sec=timer_evidence.sec(),
            )

            # 3.5) Narrowing module: post-evidence narrowing (auto-filter optimization target)
            if evidence_block:
                _before_narrowing_len = len(evidence_block)
                evidence_block = self.narrowing_module.narrow(
                    question=question,
                    evidence_block=evidence_block,
                    request_id=request_id,
                )
                _jlog(
                    "narrowing_done",
                    request_id,
                    step="3.5",
                    before_length=_before_narrowing_len,
                    after_length=len(evidence_block) if evidence_block else 0,
                )

            # 4) Combine context and evidence
            _jlog(
                "final_context_assembly_start",
                request_id,
                step="4",
                context_text_length=len(evidence_text or ""),
                evidence_length=len(evidence_block or ""),
            )
            base_text = evidence_text or "(no KG context)"
            kg_context = f"{base_text}\n\n{evidence_block}" if evidence_block else base_text
            _jlog(
                "final_context_assembly_complete",
                request_id,
                step="4",
                base_text_length=len(base_text),
                evidence_length=len(evidence_block or ""),
                final_context_length=len(kg_context),
            )

            # [LOG] Request complete
            _jlog(
                "build_kg_context_complete",
                request_id,
                step="4",
                context_length=len(kg_context),
                success=True,
                total_elapsed_sec=timer_total.sec(),
            )
            evidence_trace = getattr(self.evidence_builder, "last_evidence_trace", {}) or {}
            stage_trace = self._last_stage_trace or {}
            adaptive_trace = self._last_adaptive_trace or {
                "pass2_triggered": False,
                "conf_pass1": None,
                "conf_final": None,
                "tau_confidence": self.cfg.tau_confidence,
            }
            self.last_retrieval_trace = {
                "request_id": request_id,
                "question": question,
                "low_level_keywords": list(kw.low_level_keywords),
                "high_level_keywords": list(kw.high_level_keywords),
                "stop_reason": stage_trace.get("stop_reason"),
                "branches": stage_trace.get("branches", {}),
                "pass2_triggered": adaptive_trace.get("pass2_triggered", False),
                "rewritten_query": adaptive_trace.get("rewritten_query"),
                "conf_pass1": adaptive_trace.get("conf_pass1"),
                "conf_pass2": adaptive_trace.get("conf_pass2"),
                "conf_final": adaptive_trace.get("conf_final"),
                "tau_confidence": adaptive_trace.get("tau_confidence", self.cfg.tau_confidence),
                "pass1_entity_ids": adaptive_trace.get("pass1_entity_ids", [entity["id"] for entity in evidence_entities]),
                "pass2_entity_ids": adaptive_trace.get("pass2_entity_ids", []),
                "pass1_relation_ids": adaptive_trace.get("pass1_relation_ids", [relationship["rel_id"] for relationship in evidence_rels]),
                "pass2_relation_ids": adaptive_trace.get("pass2_relation_ids", []),
                "entity_overlap_count": adaptive_trace.get("entity_overlap_count", 0),
                "entity_overlap_pct": adaptive_trace.get("entity_overlap_pct"),
                "relation_overlap_count": adaptive_trace.get("relation_overlap_count", 0),
                "relation_overlap_pct": adaptive_trace.get("relation_overlap_pct"),
                "final_entity_ids": [entity["id"] for entity in evidence_entities],
                "final_entity_names": [entity.get("name") or entity.get("id") for entity in evidence_entities],
                "final_relationship_ids": [relationship["rel_id"] for relationship in evidence_rels],
                "final_relationship_names": self._relationship_names_from_ids([relationship["rel_id"] for relationship in evidence_rels]),
                "final_entity_count": len(evidence_entities),
                "final_relationship_count": len(evidence_rels),
                "selected_evidence_count": evidence_trace.get("selected_evidence_count", 0),
                "selected_evidence": evidence_trace.get("selected_evidence", []),
                "evidence_score_pass_count": evidence_trace.get("score_pass_count", 0),
                "evidence_score_fail_count": evidence_trace.get("score_fail_count", 0),
                "context_length": len(kg_context),
                "evidence_length": len(evidence_block or ""),
                "has_temporal_evidence": "[mentioned_at:" in kg_context,
            }

            return kg_context

        except Exception as e:
            # [LOG] Request failed
            _jlog(
                "build_kg_context_failed",
                request_id,
                step="0",
                error=str(e),
                error_type=type(e).__name__,
                total_elapsed_sec=timer_total.sec(),
            )
            self.last_retrieval_trace = {
                "request_id": request_id,
                "question": question,
                "exception": str(e),
                "error_type": type(e).__name__,
                "stop_reason": "build_kg_context_failed",
                # Distinguishes "retrieval ran and found nothing" from
                # "retrieval never ran"; both used to return the same string.
                "retrieval_failed": True,
            }
            if strict_retrieval_enabled():
                raise RetrievalFailedError(
                    f"retrieval failed for request {request_id}: "
                    f"{type(e).__name__}: {e}"
                ) from e
            return "(no KG context)"
