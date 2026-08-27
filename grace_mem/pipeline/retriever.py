"""
Refactored Retriever that uses modular components from retrieval/ folder.
"""
import os
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from grace_mem.llm.prompts.hyde_prompting import HYDE_SYSTEM, HYDE_USER
from grace_mem.llm.prompts.keyword.extraction import KEYWORD_EXTRACTION_PROMPT

# Import modular components
from grace_mem.pipeline.retrieval_steps import (
    ContextFilter,
    EntityRelationshipSearcher,
    EvidenceBuilder,
    SAConfig,
    SpreadingActivationEngine,
    SubgraphPageRank,
    TemporalRelevanceCalculator,
)
from grace_mem.pipeline.retrieval_steps.narrowing import NarrowingModule
from grace_mem.pipeline.retrieval_steps.summary_scoring import ScoringWeights
from grace_mem.pipeline.retrieval_steps.temporal import date_within_coarse_range
from grace_mem.storage import build_id_to_meta_maps
from grace_mem.utils.common import KeywordExtractionResult
from grace_mem.utils.logger_config import _StepTimer, make_module_jlog, setup_logger
from grace_mem.utils.query_time_parser import parse_query_time
from grace_mem.utils.raw_context_lookup import RawContextLookup
from grace_mem.utils.temporal import (
    build_time_context,
    rewrite_temporal_text,
    time_rewrite_ablation_enabled,
)

_jlog = make_module_jlog(name="grace_mem.Retriever", filename="kg_retriever.jsonl")
_trace_jlog = make_module_jlog(name="grace_mem.Retriever.Trace", filename="kg_retrieval_trace.jsonl")
_TRACE_PRETTY_LOG_DIR = os.environ.get("KG_TRACE_PRETTY_LOG_DIR", "logs")
_trace_pretty_log = setup_logger(
    name="kg_retrieval_trace_pretty",
    log_dir=_TRACE_PRETTY_LOG_DIR,
    to_console=False,
)


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "0").lower() not in ("0", "", "false")


def _maybe_rewrite_retrieval_question(
    question: str,
    query_time: str | None,
    request_id: str | None,
) -> str:
    """Step 0b: rewrite relative temporal expressions for retrieval only."""
    if time_rewrite_ablation_enabled():
        _jlog(
            "query_temporal_rewrite_skipped",
            request_id,
            step="0b",
            reason="ablation_no_time_rewrite",
        )
        return question

    if not query_time:
        _jlog(
            "query_temporal_rewrite_skipped",
            request_id,
            step="0b",
            reason="no_query_time",
        )
        return question

    reference_dt = parse_query_time(query_time)
    if reference_dt is None:
        _jlog(
            "query_temporal_rewrite_failed",
            request_id,
            step="0b",
            reason="parse_query_time_failed",
            query_time=query_time,
        )
        return question

    context = build_time_context(
        reference_dt=reference_dt,
        reference_time_str=query_time,
        source="retriever",
    )
    rewritten_question, temporal_meta = rewrite_temporal_text(question, context)
    constraints = temporal_meta.get("constraints", [])
    expressions_count = temporal_meta.get("expressions_count", 0)
    if rewritten_question != question:
        _jlog(
            "query_temporal_rewrite",
            request_id,
            step="0b",
            original=question,
            rewritten=rewritten_question,
            reference_time=temporal_meta.get("reference_time"),
            expressions_count=expressions_count,
            constraints=[
                {
                    "original_text": c.get("original_text"),
                    "operator": c.get("operator"),
                    "status": (c.get("resolution") or {}).get("status"),
                    "confidence": (c.get("resolution") or {}).get("confidence"),
                    "granularity": (c.get("resolution") or {}).get("granularity"),
                    "start": (c.get("resolution") or {}).get("start"),
                    "end": (c.get("resolution") or {}).get("end"),
                    "normalized_text": (c.get("resolution") or {}).get("normalized_text"),
                }
                for c in constraints
            ],
        )
    else:
        _jlog(
            "query_temporal_rewrite_no_change",
            request_id,
            step="0b",
            original=question,
            reference_time=temporal_meta.get("reference_time"),
            expressions_count=expressions_count,
            constraints=[
                {
                    "original_text": c.get("original_text"),
                    "status": (c.get("resolution") or {}).get("status"),
                    "confidence": (c.get("resolution") or {}).get("confidence"),
                }
                for c in constraints
            ],
        )
    return rewritten_question


# --------------------------------------------------------------------------- #
# Keyword extraction cache                                                     #
# --------------------------------------------------------------------------- #
# The keyword-extraction LLM (served via vLLM/LM Studio) is not deterministic
# even with temperature=0 + seed, which makes retrieval non-reproducible across
# runs. To pin reproducibility, keyword results are cached on disk keyed by a
# hash of the exact prompt (query + guidance). Enabled by default; point
# KG_KEYWORD_CACHE_PATH at a shared file to reuse across runs, or set
# KG_KEYWORD_CACHE_DISABLE=1 to bypass.
import hashlib as _hashlib
import json as _json
import threading as _threading

_KEYWORD_CACHE_PATH = os.environ.get(
    "KG_KEYWORD_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".keyword_cache.json"),
)
_KEYWORD_CACHE_DISABLED = os.environ.get("KG_KEYWORD_CACHE_DISABLE", "") == "1"
_keyword_cache_lock = _threading.Lock()
_keyword_cache: dict[str, dict[str, list[str]]] | None = None


def _keyword_cache_key(prompt: str) -> str:
    return _hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _load_keyword_cache() -> dict[str, dict[str, list[str]]]:
    """Load the on-disk keyword-extraction cache, if one exists.

    Keyword extraction is an LLM call per question, and evaluation re-asks the
    same questions across ablation runs. Caching removes that cost from every
    run after the first -- and, more importantly, removes it as a source of
    variation between them.
    """
    global _keyword_cache
    if _keyword_cache is None:
        try:
            with open(_KEYWORD_CACHE_PATH, "r", encoding="utf-8") as f:
                _keyword_cache = _json.load(f)
        except (FileNotFoundError, ValueError):
            _keyword_cache = {}
    return _keyword_cache


def _keyword_cache_get(prompt: str) -> Optional["KeywordExtractionResult"]:
    """Look up cached keywords for a question, or None on a miss."""
    if _KEYWORD_CACHE_DISABLED:
        return None
    with _keyword_cache_lock:
        cache = _load_keyword_cache()
        hit = cache.get(_keyword_cache_key(prompt))
    if hit is None:
        return None
    return KeywordExtractionResult(
        high_level_keywords=list(hit.get("high_level_keywords", [])),
        low_level_keywords=list(hit.get("low_level_keywords", [])),
    )


def _keyword_cache_put(prompt: str, res: "KeywordExtractionResult") -> None:
    """Store a question's extracted keywords and persist the cache."""
    if _KEYWORD_CACHE_DISABLED:
        return
    with _keyword_cache_lock:
        cache = _load_keyword_cache()
        cache[_keyword_cache_key(prompt)] = {
            "high_level_keywords": list(res.high_level_keywords),
            "low_level_keywords": list(res.low_level_keywords),
        }
        tmp = f"{_KEYWORD_CACHE_PATH}.tmp"
        try:
            os.makedirs(os.path.dirname(_KEYWORD_CACHE_PATH) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                _json.dump(cache, f, ensure_ascii=False)
            os.replace(tmp, _KEYWORD_CACHE_PATH)
        except OSError:
            pass


@dataclass(frozen=True)
class RetrieverConfig:
    """Single source of truth for Retriever defaults."""
    summary_embed_dim: int = 1024
    # entity initial search
    ent_topk: int = 5
    ent_threshold: float = 0.3
    # relationship initial search
    rel_topk: int = 5
    rel_threshold: float = 0.3
    # post-intersection filtering
    filter_ent_topk: int = 3
    filter_rel_topk: int = 3
    filter_ent_threshold: float = 0.5
    filter_rel_threshold: float = 0.5
    # reranker for recovering filtered items
    use_reranker: bool = True
    reranker_threshold: float = -3.0
    reranker_topk: int = 5
    # spreading activation
    use_spreading_activation: bool = False
    sa_max_hops: int = 2
    sa_rescale_c: float = 0.4
    sa_tau_a: float = 0.5
    sa_max_activated: int = 20
    # evidence
    summary_topk_per_item: int = 5
    summary_vec_threshold: float = 0.4
    use_full_summary: bool = True
    fallback_to_raw: bool = False
    # adaptive re-search (off by default — enable per call or via custom config)
    enable_adaptive_search: bool = False
    tau_confidence: float = 0.70           # trigger threshold
    adaptive_threshold_scale: float = 0.8  # filter threshold multiplier for pass-2
    novel_ent_threshold: float = 0.35      # min similarity to original query_vec to admit a novel entity
    # Step 2.6 filter method — single axis for ablation
    filter_method: str = "similarity"      # "similarity" | "rrf" | "ppr" | "rrf+ppr" | "reranker_only"
    # RRF parameters (active when filter_method in {"rrf", "rrf+ppr"})
    rrf_k: float = 60.0
    rrf_candidate_k: int = 50             # RRF top-N fed into PPR (only for "rrf+ppr")
    # PPR parameters (active when filter_method in {"ppr", "rrf+ppr"})
    ppr_alpha: float = 0.85
    ppr_top_k: int = 10
    ppr_inverse_degree: bool = False
    # reranker-only filter params (active when filter_method == "reranker_only")
    rrk_ent_topk: int = 5          # max entities to keep
    rrk_rel_topk: int = 5          # max relationships to keep
    rrk_threshold: float = 0.0     # score cutoff — 0.0 means "Yes logit > No logit"
    # ── Summary selection strategy ────────────────────────────────────────────
    # "semantic"               → baseline cosine-similarity ranking (default)
    # "graph_count"            → graph link counts only (semantic_weight=0)
    # "graph_semantic"         → graph counts + weak semantic tie-breaker
    # "graph_semantic_penalty" → graph + semantic + popularity/redundancy penalties
    summary_filter_mode: str = "semantic"
    # Scoring weights (used when summary_filter_mode != "semantic")
    summary_relation_weight: float = 2.0
    summary_entity_weight: float = 1.0
    summary_pair_bonus_weight: float = 1.5
    summary_semantic_weight: float = 0.5
    summary_popularity_penalty_weight: float = 1.0
    summary_redundancy_penalty_weight: float = 1.0
    summary_enable_pair_bonus: bool = True
    summary_enable_popularity_penalty: bool = False
    summary_enable_redundancy_penalty: bool = False
    # RRF-specific (used when summary_filter_mode in {"graph_rrf", "graph_rrf_mmr"})
    summary_rrf_k: float = 60.0
    # ── Raw context mode ──────────────────────────────────────────────────────
    # When True, summary vectors are still used for scoring and top-K selection,
    # but the final text returned for each selected snippet is the raw turn text
    # instead of the summary text.
    use_raw_context: bool = False
    # Path to the script_data directory containing raw CSV conversation files.
    # Required when use_raw_context=True or use_split_embeddings=True.
    raw_context_data_dir: str = ""
    # When True, evidence is selected at VDB-entry level rather than turn level, via
    # EvidenceBuilder._build_evidence_split. Mutually exclusive with use_raw_context.
    # Entry granularity is controlled by split_single_entry_raw below: one entry per
    # summary_id (the default, matching what Ingestor writes), or :u (user raw) /
    # :a (assistant compressed) pairs built by rebuild_split_summaries.py.
    # Defaults to True so build_pipeline() takes the same evidence path the benchmark
    # pipelines take (experiment_config.RERANKER_PARAMS sets this too); the legacy
    # turn-level branch is kept for artifacts that predate split selection.
    use_split_embeddings: bool = True
    # Direct summary vector retrieval (split-embedding mode only). When > 0, the top-N
    # summaries by raw query similarity are pulled straight from the VDB and merged into
    # the evidence candidate pool, in parallel with entity/relationship spreading
    # activation. This recovers high-similarity gold summaries whose turn is not linked
    # to any retrieved entity (so they never enter the prov-based pool). 0 = disabled.
    summary_direct_vector_topn: int = 0
    # Min raw query-similarity for a direct-vector hit to be admitted as an EXTRA
    # evidence slot (added on top of the prov top-K, not competing for it). Requires
    # summary_direct_vector_topn > 0. 0.0 = extra-slot mode disabled.
    summary_direct_vector_min_score: float = 0.0
    # Retrieve-then-rerank (split-embedding mode). When > 0, the candidate pool
    # (prov + direct at the min-score floor) is reranked by the cross-encoder
    # reranker and the top-N are kept as final evidence. Supersedes the extra-slot
    # path. 0 = disabled.
    summary_rerank_topk: int = 0
    # Ablation: in rerank path, skip the cross-encoder and keep cosine top-N.
    summary_rerank_cosine_only: bool = False
    # Single-entry raw mode for the split path (e.g. LoCoMo): the VDB has one entry
    # per summary_id (no :u/:a suffixes), and the text fed to the LLM is the raw turn
    # text (raw_text metadata) instead of the compressed summary. Lets the rerank16
    # flow (direct-vector + cross-encoder rerank) run on datasets that don't use the
    # user/assistant split embedding scheme.
    # Defaults to True because Ingestor.summarize_and_ingest_turn writes exactly one
    # entry per summary_id and never writes :u/:a. Those pairs are a LongMem-only
    # post-processing pass (experiment/longmem/tools/rebuild_split_summaries.py), so only the
    # LongMem pipeline sets this to False — and it derives the value from
    # INGEST_PARAMS["use_split_summary"], the same flag that decides whether the rebuild
    # ran at all. Setting False against artifacts that were never rebuilt makes every
    # provenance candidate miss silently.
    split_single_entry_raw: bool = True
    # ── HyDE summary retrieval ────────────────────────────────────────────────
    # Generate hypothetical answer sentences, embed them, and blend their summary
    # similarity with the query similarity: score = (1-w)*sim_query + w*sim_hyde.
    summary_hyde_enable: bool = False
    summary_hyde_weight: float = 0.3       # w in the blend; 0.0 reproduces baseline
    summary_hyde_mode: str = "blend"       # "blend" (compete for top-K) | "fill" (backfill unused slots only)
    # Per-entity quota: guarantee this many snippets per source entity/relationship
    # before filling remaining top-K slots by score (0 = disabled).
    summary_per_entity_min: int = 0
    # Keyword source for relationship vector search:
    # "high_level" (abstract reasoning words, baseline) | "low_level" (concrete anchors) | "both"
    relation_search_keywords: str = "high_level"


class Retriever:
    """
    Refactored Knowledge Graph Retriever using modular components.

    Components:
    - EntityRelationshipSearcher: Hybrid entity/relationship search
    - TemporalRelevanceCalculator: Temporal relevance scoring
    - EvidenceBuilder: Evidence block building
    - ContextFilter: Filtering and reranking
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
        self.context_filter = ContextFilter(
            vector_db_manager=self.MGR,
            cache=self.cache
        )

        self.ppr_engine = SubgraphPageRank()

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

        # [LOG] Retriever initialization with effective config
        _jlog(
            "retriever_initialized",
            request_id="INIT",
            config={k: v for k, v in self.cfg.__dict__.items()},
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
                src_name = (
                    meta.get("source_entity")
                    or (ent_id2meta.get(meta.get("source_id"), {}) or {}).get("name")
                    or meta.get("source_id")
                    or "?"
                )
                tgt_name = (
                    meta.get("target_entity")
                    or (ent_id2meta.get(meta.get("target_id"), {}) or {}).get("name")
                    or meta.get("target_id")
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
            src_name = (
                meta.get("source_entity")
                or (ent_id2meta.get(meta.get("source_id"), {}) or {}).get("name")
                or meta.get("source_id")
            )
            tgt_name = (
                meta.get("target_entity")
                or (ent_id2meta.get(meta.get("target_id"), {}) or {}).get("name")
                or meta.get("target_id")
            )
            for name in (src_name, tgt_name):
                if name and name not in seen:
                    names.append(name)
                    seen.add(name)
        return names

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        """Deduplicate strings while preserving the original order."""
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            value = (item or "").strip()
            if not value or value in seen:
                continue
            out.append(value)
            seen.add(value)
        return out

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
        return self._dedupe_preserve_order(names)

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
        return self._dedupe_preserve_order(labels)

    def _entity_names_from_edge_subgraph(self, edge_subgraph: list[dict[str, Any]]) -> list[str]:
        """Collect all endpoint entity names present in an edge subgraph."""
        names: list[str] = []
        for edge in edge_subgraph or []:
            names.append(edge.get("source_name") or self._entity_name_by_id(edge.get("source_id")))
            names.append(edge.get("target_name") or self._entity_name_by_id(edge.get("target_id")))
        return self._dedupe_preserve_order(names)

    def _relationship_names_from_edge_subgraph(self, edge_subgraph: list[dict[str, Any]]) -> list[str]:
        """Collect all readable edge labels present in an edge subgraph."""
        return self._dedupe_preserve_order(
            [self._relationship_name_from_edge(edge) for edge in (edge_subgraph or [])]
        )

    def _build_stage_trace_snapshot(
        self,
        *,
        step: str,
        stage: str,
        entity_names: list[str],
        relationship_names: list[str],
        previous: dict[str, Any] | None = None,
        skipped: bool = False,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Build one readable stage snapshot with additions/removals from the previous stage."""
        current_entities = self._dedupe_preserve_order(entity_names)
        current_relationships = self._dedupe_preserve_order(relationship_names)
        prev_entities = (previous or {}).get("entity_names") or []
        prev_relationships = (previous or {}).get("relationship_names") or []
        prev_entity_set = set(prev_entities)
        prev_relationship_set = set(prev_relationships)
        current_entity_set = set(current_entities)
        current_relationship_set = set(current_relationships)

        return {
            "step": step,
            "stage": stage,
            "skipped": skipped,
            "reason": reason,
            "entity_count": len(current_entities),
            "relationship_count": len(current_relationships),
            "entity_names": current_entities,
            "relationship_names": current_relationships,
            "added_entity_names": [name for name in current_entities if name not in prev_entity_set],
            "removed_entity_names": [name for name in prev_entities if name not in current_entity_set],
            "added_relationship_names": [name for name in current_relationships if name not in prev_relationship_set],
            "removed_relationship_names": [name for name in prev_relationships if name not in current_relationship_set],
        }

    @staticmethod
    def _format_trace_names(names: list[str]) -> str:
        """Render a readable names list for the pretty waterfall trace."""
        if not names:
            return "-"
        return "; ".join(names)

    def _format_retrieval_stage_trace_text(
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
        stop_reason: str | None,
        elapsed_sec: float | None,
    ) -> str:
        """Format one retrieval request as a readable waterfall trace block."""
        lines = [
            "=" * 80,
            f"request_id: {request_id or '-'}",
            f"question: {question}",
            f"low_level_keywords: {self._format_trace_names(low_level_keywords)}",
            f"high_level_keywords: {self._format_trace_names(high_level_keywords)}",
            f"graph_override: {graph_override}",
            f"stop_reason: {stop_reason or '-'}",
            f"elapsed_sec: {round(elapsed_sec, 4) if elapsed_sec is not None else '-'}",
            "",
        ]

        def append_branch(branch_name: str, stages: list[dict[str, Any]]) -> None:
            """Record one retrieval branch's state into the trace.

            What the differential analysis in `derive_drop_reasons` later diffs: each
            stage appends the entities and relationships still standing, and what
            disappeared between consecutive entries is what that stage dropped. A stage
            that does not append here is invisible to failure analysis, and its drops
            get attributed to the next stage that does.
            """
            lines.append(f"[{branch_name}]")
            if not stages:
                lines.append("  (no stages)")
                lines.append("")
                return

            for stage in stages:
                header = f"{stage['step']} {stage['stage']}"
                if stage.get("skipped"):
                    header += f" (skipped: {stage.get('reason') or 'unknown'})"
                lines.append(header)
                lines.append(
                    f"  entities[{stage['entity_count']}]: "
                    f"{self._format_trace_names(stage.get('entity_names') or [])}"
                )
                lines.append(
                    f"  relationships[{stage['relationship_count']}]: "
                    f"{self._format_trace_names(stage.get('relationship_names') or [])}"
                )
                lines.append(
                    f"  + entities: {self._format_trace_names(stage.get('added_entity_names') or [])}"
                )
                lines.append(
                    f"  - entities: {self._format_trace_names(stage.get('removed_entity_names') or [])}"
                )
                lines.append(
                    "  + relationships: "
                    f"{self._format_trace_names(stage.get('added_relationship_names') or [])}"
                )
                lines.append(
                    "  - relationships: "
                    f"{self._format_trace_names(stage.get('removed_relationship_names') or [])}"
                )
                lines.append("")

        append_branch("local", local_branch)
        append_branch("global", global_branch)
        append_branch("merged", merged_branch)
        return "\n".join(lines).rstrip()

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
        waterfall_text = self._format_retrieval_stage_trace_text(
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

    def generate_query_keywords(self, question: str, request_id: str | None = None,
                                max_retries: int = 5,
                                retrieval_guidance: str | None = None) -> KeywordExtractionResult:
        """
        Extract local/global keywords from query.
        Retries up to max_retries times only if the LLM output is unparseable.
        Empty or partial keyword lists are allowed so retrieval can continue
        with whichever signals are available.
        """
        import re as _re
        timer = _StepTimer()
        guidance_section = ""
        if retrieval_guidance:
            guidance_section = f"\nRetrieval guidance:\n{retrieval_guidance}\n"
        keyword_prompt = KEYWORD_EXTRACTION_PROMPT.format(
            query=question, guidance_section=guidance_section
        )
        last_error = ""
        js = ""

        _jlog(
            "generate_query_keywords_start",
            request_id,
            step="1",
            question=question,
            max_retries=max_retries,
        )

        # Reproducibility: return cached keywords if this exact prompt was seen.
        cached = _keyword_cache_get(keyword_prompt)
        if cached is not None:
            _jlog(
                "generate_keywords_cache_hit",
                request_id,
                step="1",
                high_level_count=len(cached.high_level_keywords),
                low_level_count=len(cached.low_level_keywords),
                high_level_keywords=cached.high_level_keywords,
                low_level_keywords=cached.low_level_keywords,
                elapsed_sec=timer.sec(),
            )
            return cached

        for attempt in range(1, max_retries + 1):
            try:
                _jlog(
                    "generate_keywords_attempt_start",
                    request_id,
                    step="1",
                    attempt=attempt,
                )
                js, sec = self.llm.generate_llm_keyword(keyword_prompt)
                _jlog(
                    "generate_keywords_llm_done",
                    request_id,
                    step="1",
                    attempt=attempt,
                    latency_sec=sec,
                )

                # Strip <think>...</think> or any prose before the JSON object
                m = _re.search(r'\{.*\}', js, _re.DOTALL)
                if m:
                    js = m.group(0)

                res = KeywordExtractionResult.model_validate_json(js)

                if not res.high_level_keywords and not res.low_level_keywords:
                    _jlog(
                        "generate_keywords_empty",
                        request_id,
                        step="1",
                        attempt=attempt,
                        high_level_count=len(res.high_level_keywords),
                        low_level_count=len(res.low_level_keywords),
                    )
                    if attempt < max_retries:
                        continue
                elif not res.high_level_keywords or not res.low_level_keywords:
                    _jlog(
                        "generate_keywords_partial",
                        request_id,
                        step="1",
                        attempt=attempt,
                        high_level_count=len(res.high_level_keywords),
                        low_level_count=len(res.low_level_keywords),
                    )

                _jlog(
                    "generate_query_keywords_result",
                    request_id,
                    step="1",
                    attempt=attempt,
                    high_level_count=len(res.high_level_keywords),
                    low_level_count=len(res.low_level_keywords),
                    high_level_keywords=res.high_level_keywords,
                    low_level_keywords=res.low_level_keywords,
                    elapsed_sec=timer.sec(),
                )
                # Cache non-empty results so reruns are reproducible.
                if res.low_level_keywords or res.high_level_keywords:
                    _keyword_cache_put(keyword_prompt, res)
                return res

            except Exception as e:
                last_error = str(e)
                _jlog(
                    "generate_keywords_attempt_failed",
                    request_id,
                    step="1",
                    attempt=attempt,
                    error=last_error,
                )

        # All retries exhausted
        _jlog(
            "generate_keywords_give_up",
            request_id,
            step="1",
            max_retries=max_retries,
            last_error=last_error,
            raw_output_preview=repr(js[:500]) if js else "",
            elapsed_sec=timer.sec(),
        )
        return KeywordExtractionResult()

    def generate_hyde_vector(self, question: str, request_id: str | None = None):
        """
        HyDE: generate hypothetical answer-summary sentences for the question,
        embed them, and return a single normalized vector (mean of sentence
        embeddings). Returns None on failure so the caller falls back to the
        plain query vector.
        """
        timer = _StepTimer()
        try:
            user_prompt = HYDE_USER.format(question=question)
            raw, sec = self.llm.generate_llm_hyde(HYDE_SYSTEM, user_prompt)
            sentences = [s.strip(" -•\t") for s in (raw or "").splitlines() if s.strip()]
            if not sentences:
                _jlog("hyde_empty", request_id, step="0d", latency_sec=sec)
                return None
            vecs = self.searcher.embed(sentences)
            vec = np.asarray(vecs, dtype=np.float32).mean(axis=0)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            _jlog(
                "hyde_done",
                request_id,
                step="0d",
                latency_sec=sec,
                sentence_count=len(sentences),
                sentences=sentences,
                elapsed_sec=timer.sec(),
            )
            return vec
        except Exception as exc:
            _jlog("hyde_error", request_id, step="0d", error=str(exc))
            return None

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
                self._build_stage_trace_snapshot(
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

        # Extract entity IDs and per-source score dicts for RRF
        ent_ids = list({meta["id"] for hits in ent_hits.values() for meta, _ in hits})
        entity_emb_scores: dict[str, float] = {
            meta["id"]: float(score)
            for meta, score in ent_hits.get("__vector__", [])
            if meta.get("id")
        }
        entity_bm25_scores: dict[str, float] = {}
        for source, hits in ent_hits.items():
            if source == "__vector__":
                continue
            for meta, score in hits:
                eid = meta.get("id")
                if eid and (eid not in entity_bm25_scores or float(score) > entity_bm25_scores[eid]):
                    entity_bm25_scores[eid] = float(score)
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
            return [], [], "", query_vec

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
                return [], [], "", query_vec
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

        # Extract relationship IDs and per-source score dicts for RRF
        rel_ids = list({meta["id"] for hits in rel_hits.values() for meta, _ in hits})
        rel_emb_scores: dict[str, float] = {}
        rel_endpoint_scores: dict[str, float] = {}
        for hits in rel_hits.values():
            for meta, score in hits:
                rid = meta.get("id")
                if rid and (rid not in rel_emb_scores or float(score) > rel_emb_scores[rid]):
                    rel_emb_scores[rid] = float(score)
                for ep_key in ("source_id", "target_id"):
                    ep_id = meta.get(ep_key)
                    if ep_id and (ep_id not in rel_endpoint_scores or float(score) > rel_endpoint_scores[ep_id]):
                        rel_endpoint_scores[ep_id] = float(score)
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

        # 2) Compute intersection (using union for now as per original code)
        intersect_entity_ids, intersect_rel_ids = self.context_filter.compute_subgraph_intersection(
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

        # 3) Filter candidates (step 2.6) — dispatch on filter_method
        timer_filter = _StepTimer()
        _jlog(
            "filter_step_start",
            request_id,
            step="2.6",
            filter_method=self.cfg.filter_method,
            entity_candidate_count=len(intersect_entity_ids),
            relationship_candidate_count=len(intersect_rel_ids),
            filter_ent_topk=filter_ent_topk,
            filter_rel_topk=filter_rel_topk,
        )

        # Node-subgraph relation IDs (presence signal for relation RRF L3)
        node_subgraph_rel_ids: set = {
            nb["rel_id"]
            for b in node_subgraph.values()
            for nb in (b.get("neighbors") or [])
            if "rel_id" in nb
        }

        if self.cfg.filter_method == "similarity":
            filtered_entity_ids, filtered_rel_ids = self.context_filter.filter_by_similarity(
                entity_ids=intersect_entity_ids,
                relationship_ids=intersect_rel_ids,
                query_vec=query_vec,
                filter_entity_top_k=filter_ent_topk,
                filter_relationship_top_k=filter_rel_topk,
                filter_entity_threshold=filter_ent_threshold,
                filter_relationship_threshold=filter_rel_threshold,
                request_id=request_id,
            )

        elif self.cfg.filter_method in ("rrf", "rrf+ppr"):
            rrf_top_k = (
                filter_ent_topk
                if self.cfg.filter_method == "rrf"
                else self.cfg.rrf_candidate_k
            )
            filtered_entity_ids, filtered_rel_ids, rrf_scores = self.context_filter.filter_by_rrf(
                entity_ids=intersect_entity_ids,
                relationship_ids=intersect_rel_ids,
                query_vec=query_vec,
                rrf_k=self.cfg.rrf_k,
                filter_entity_top_k=rrf_top_k,
                filter_relationship_top_k=filter_rel_topk,
                entity_emb_scores=entity_emb_scores,
                entity_bm25_scores=entity_bm25_scores,
                rel_endpoint_scores=rel_endpoint_scores,
                rel_emb_scores=rel_emb_scores,
                node_subgraph_rel_ids=node_subgraph_rel_ids,
                filter_method=self.cfg.filter_method,
                request_id=request_id,
            )

            if self.cfg.filter_method == "rrf+ppr":
                _, rel_id2meta_pp = build_id_to_meta_maps(self.cache)
                rrf_candidate_set = set(filtered_entity_ids)
                induced_edges = [
                    (rel_id2meta_pp[rid]["source_id"], rid, rel_id2meta_pp[rid]["target_id"])
                    for rid in intersect_rel_ids
                    if rid in rel_id2meta_pp
                    and rel_id2meta_pp[rid].get("source_id") in rrf_candidate_set
                    and rel_id2meta_pp[rid].get("target_id") in rrf_candidate_set
                ]
                ppr_entities, _ = self.ppr_engine.run_ppr(
                    entity_ids=filtered_entity_ids,
                    rrf_scores=rrf_scores,
                    subgraph_edges=induced_edges,
                    alpha=self.cfg.ppr_alpha,
                    top_k=self.cfg.ppr_top_k,
                    inverse_degree_weight=self.cfg.ppr_inverse_degree,
                )
                ppr_ent_set = set(ppr_entities)
                filtered_entity_ids = ppr_entities
                filtered_rel_ids = [
                    rid for rid in intersect_rel_ids
                    if rel_id2meta_pp.get(rid, {}).get("source_id") in ppr_ent_set
                    and rel_id2meta_pp.get(rid, {}).get("target_id") in ppr_ent_set
                ]
                _jlog(
                    "ppr_done",
                    request_id,
                    step="2.6",
                    ppr_entity_count=len(filtered_entity_ids),
                    ppr_rel_count=len(filtered_rel_ids),
                )

        elif self.cfg.filter_method == "ppr":
            cosine_scores_ppr = self.context_filter.compute_cosine_scores(
                intersect_entity_ids, query_vec
            )
            _, rel_id2meta_pp = build_id_to_meta_maps(self.cache)
            ppr_candidates = sorted(intersect_entity_ids)
            ppr_candidate_set = set(ppr_candidates)
            induced_edges_ppr = [
                (rel_id2meta_pp[rid]["source_id"], rid, rel_id2meta_pp[rid]["target_id"])
                for rid in intersect_rel_ids
                if rid in rel_id2meta_pp
                and rel_id2meta_pp[rid].get("source_id") in ppr_candidate_set
                and rel_id2meta_pp[rid].get("target_id") in ppr_candidate_set
            ]
            ppr_entities_ppr, _ = self.ppr_engine.run_ppr(
                entity_ids=ppr_candidates,
                rrf_scores=cosine_scores_ppr,
                subgraph_edges=induced_edges_ppr,
                alpha=self.cfg.ppr_alpha,
                top_k=self.cfg.ppr_top_k,
                inverse_degree_weight=self.cfg.ppr_inverse_degree,
            )
            ppr_ent_set_ppr = set(ppr_entities_ppr)
            filtered_entity_ids = ppr_entities_ppr
            filtered_rel_ids = [
                rid for rid in intersect_rel_ids
                if rel_id2meta_pp.get(rid, {}).get("source_id") in ppr_ent_set_ppr
                and rel_id2meta_pp.get(rid, {}).get("target_id") in ppr_ent_set_ppr
            ]
            _jlog(
                "ppr_done",
                request_id,
                step="2.6",
                ppr_entity_count=len(filtered_entity_ids),
                ppr_rel_count=len(filtered_rel_ids),
            )

        elif self.cfg.filter_method == "reranker_only":
            # skip pre-filter; reranker handles selection in step 2.7
            filtered_entity_ids = sorted(intersect_entity_ids)
            filtered_rel_ids = sorted(intersect_rel_ids)
        else:
            raise ValueError(f"Unknown filter_method: {self.cfg.filter_method!r}")

        _jlog(
            "filter_step_done",
            request_id,
            step="2.6",
            filter_method=self.cfg.filter_method,
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

        # Convert IDs to full metadata dicts
        ent_id2meta, rel_id2meta = build_id_to_meta_maps(self.cache)

        filtered_entities = []
        for eid in filtered_entity_ids:
            meta = ent_id2meta.get(eid, {})
            if meta:
                filtered_entities.append({
                    "id": eid,
                    "name": meta.get("name"),
                    "type": meta.get("type"),
                    "desc": meta.get("description"),
                })

        filtered_rels = []
        for rid in filtered_rel_ids:
            meta = rel_id2meta.get(rid, {})
            if meta:
                # Get source and target entity info
                src_id = meta.get("source_id")
                tgt_id = meta.get("target_id")
                src_meta = ent_id2meta.get(src_id, {})
                tgt_meta = ent_id2meta.get(tgt_id, {})

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

        # 4) Optional reranker recovery / reranker-only selection
        if self.cfg.filter_method == "reranker_only":
            # reranker IS the filter — score all candidates, select top-K
            filtered_entity_ids_set, filtered_rel_ids_set = self.context_filter.rerank_filter(
                question=question,
                entity_ids=intersect_entity_ids,
                relationship_ids=intersect_rel_ids,
                entity_top_k=self.cfg.rrk_ent_topk,
                relationship_top_k=self.cfg.rrk_rel_topk,
                threshold=self.cfg.rrk_threshold,
                request_id=request_id,
            )
        elif self.cfg.use_reranker:
            filtered_entity_ids_set, filtered_rel_ids_set = self.context_filter.rerank_and_recover(
                question=question,
                all_entity_ids=intersect_entity_ids,
                all_relationship_ids=intersect_rel_ids,
                filtered_entity_ids=set(filtered_entity_ids),
                filtered_relationship_ids=set(filtered_rel_ids),
                reranker_threshold=self.cfg.reranker_threshold,
                reranker_top_k=self.cfg.reranker_topk,
                request_id=request_id,
            )
        else:
            filtered_entity_ids_set = None
            filtered_rel_ids_set = None

        if filtered_entity_ids_set is not None:
            # Re-convert IDs to metadata dicts after reranker or reranker_only
            filtered_entities = []
            for eid in filtered_entity_ids_set:
                meta = ent_id2meta.get(eid, {})
                if meta:
                    filtered_entities.append({
                        "id": eid,
                        "name": meta.get("name"),
                        "type": meta.get("type"),
                        "desc": meta.get("description"),
                    })

            filtered_rels = []
            for rid in filtered_rel_ids_set:
                meta = rel_id2meta.get(rid, {})
                if meta:
                    src_id = meta.get("source_id")
                    tgt_id = meta.get("target_id")
                    src_meta = ent_id2meta.get(src_id, {})
                    tgt_meta = ent_id2meta.get(tgt_id, {})

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
                entity_names=self._entity_names_from_ids(sorted(filtered_entity_ids_set)),
                relationship_names=self._relationship_names_from_ids(sorted(filtered_rel_ids_set)),
            )
        else:
            append_trace(
                merged_branch,
                step="2.7",
                stage="reranker_final",
                entity_names=self._entity_names_from_ids(filtered_entity_ids),
                relationship_names=self._relationship_names_from_ids(filtered_rel_ids),
                skipped=True,
                reason="disabled",
            )
            _jlog("reranker_skipped", request_id, step="2.7", reason="disabled")

        # 4b) Temporal containment boost: surface coarse-grained entities whose
        # stored range contains the parsed query date (MONTH/WEEK/SEASON/YEAR).
        # Ablation H: KG_ABLATION_NO_TEMPORAL_BOOST=1 -> skip this reranking.
        # Note: LoCoMo passes no query_time, so this block is structurally dead
        # there; H only means anything for LongMem.
        if _env_flag_enabled("KG_ABLATION_NO_TEMPORAL_BOOST"):
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
        context_text = self._render_context_text(
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

    def _render_context_text(
        self,
        entities: list[dict],
        relationships: list[dict],
        request_id: str | None = None,
    ) -> str:
        """Render entities and relationships into readable context text."""
        timer_render = _StepTimer()
        lines = []

        ent_id2meta, rel_id2meta = build_id_to_meta_maps(self.cache)

        temporal_types = {"Date", "Event", "Activity"}
        if entities:
            lines.append("=== Entities ===")
            for ent in entities:
                name = ent.get("name", "")
                ent_type = ent.get("type", "")
                desc = ent.get("desc", "")
                eid = ent.get("id", "")
                meta = ent_id2meta.get(eid, {}) or {}
                prov = meta.get("prov") or {}
                dt_str, _ = TemporalRelevanceCalculator.get_newest_dialogue_datetime(prov, request_id)
                temporal_tag = f" [mentioned_at:{dt_str}]" if dt_str and ent_type in temporal_types else ""
                temporal_meta = meta.get("temporal") or {}
                temporal_suffix = ""
                if temporal_meta:
                    parts = []
                    if temporal_meta.get("display_value"):
                        parts.append(f"display_value={temporal_meta['display_value']}")
                    if temporal_meta.get("normalized_start"):
                        parts.append(f"normalized_start={temporal_meta['normalized_start']}")
                    if temporal_meta.get("normalized_end"):
                        parts.append(f"normalized_end={temporal_meta['normalized_end']}")
                    if temporal_meta.get("original_phrase"):
                        parts.append(f"original_phrase={temporal_meta['original_phrase']}")
                    if temporal_meta.get("reference_time"):
                        parts.append(f"reference_time={temporal_meta['reference_time']}")
                    if parts:
                        temporal_suffix = " [" + "; ".join(parts) + "]"
                lines.append(f"- {name} ({ent_type}): {desc}{temporal_suffix}{temporal_tag}")

        if relationships:
            lines.append("\n=== Relationships ===")
            for rel in relationships:
                src_name = rel.get("source_name", "")
                tgt_name = rel.get("target_name", "")
                rel_desc = rel.get("rel_desc", "")
                rid = rel.get("rel_id", "")
                rmeta = rel_id2meta.get(rid, {}) or {}
                rprov = rmeta.get("prov") or {}
                rdt_str, _ = TemporalRelevanceCalculator.get_newest_dialogue_datetime(rprov, request_id)
                rtype = rmeta.get("type", "")
                temporal_tag = f" [mentioned_at:{rdt_str}]" if rdt_str and rtype in temporal_types else ""
                lines.append(f"- {src_name} -> {tgt_name}: {rel_desc}{temporal_tag}")

        result = "\n".join(lines) if lines else ""
        _jlog(
            "context_text_rendered",
            request_id,
            step="2",
            entity_count=len(entities),
            relationship_count=len(relationships),
            line_count=len(lines),
            result_length=len(result),
            elapsed_sec=timer_render.sec(),
        )
        return result

    @staticmethod
    def _compute_overlap_metrics(
        pass1_ids: list[str],
        pass2_ids: list[str],
    ) -> tuple[int, float | None]:
        """Return intersection size and Jaccard overlap for unique IDs."""
        pass1 = set(pass1_ids)
        pass2 = set(pass2_ids)
        union = pass1 | pass2
        if not union:
            return 0, None
        overlap_count = len(pass1 & pass2)
        return overlap_count, overlap_count / len(union)

    def _build_adaptive_trace(
        self,
        *,
        pass2_triggered: bool,
        pass1_entity_ids: list[str],
        pass1_relation_ids: list[str],
        pass2_entity_ids: list[str] | None = None,
        pass2_relation_ids: list[str] | None = None,
        conf_pass1: float | None = None,
        conf_pass2: float | None = None,
        conf_final: float | None = None,
        rewritten_query: str | None = None,
        adaptive_skip_reason: str | None = None,
    ) -> dict[str, Any]:
        """Build a stable trace from pre-merge pass results."""
        entity_ids_2 = list(pass2_entity_ids or []) if pass2_triggered else []
        relation_ids_2 = list(pass2_relation_ids or []) if pass2_triggered else []
        if pass2_triggered:
            entity_overlap_count, entity_overlap_pct = self._compute_overlap_metrics(
                pass1_entity_ids,
                entity_ids_2,
            )
            relation_overlap_count, relation_overlap_pct = self._compute_overlap_metrics(
                pass1_relation_ids,
                relation_ids_2,
            )
        else:
            entity_overlap_count = relation_overlap_count = 0
            entity_overlap_pct = relation_overlap_pct = None

        config = getattr(self, "cfg", None)
        trace = {
            "pass2_triggered": pass2_triggered,
            "conf_pass1": conf_pass1,
            "conf_pass2": conf_pass2,
            "conf_final": conf_final,
            "tau_confidence": getattr(config, "tau_confidence", None),
            "rewritten_query": rewritten_query,
            "adaptive_skip_reason": adaptive_skip_reason,
            "pass1_entity_ids": list(pass1_entity_ids),
            "pass2_entity_ids": entity_ids_2,
            "pass1_relation_ids": list(pass1_relation_ids),
            "pass2_relation_ids": relation_ids_2,
            "entity_overlap_count": entity_overlap_count,
            "entity_overlap_pct": entity_overlap_pct,
            "relation_overlap_count": relation_overlap_count,
            "relation_overlap_pct": relation_overlap_pct,
        }
        return trace

    def _adaptive_research(
        self,
        *,
        question: str,
        ctx_entities: list[dict],
        ctx_rels: list[dict],
        ctx_text: str,
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
        from grace_mem.pipeline.retrieval_steps.adaptive import (
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
            entity_count=len(ctx_entities),
            relationship_count=len(ctx_rels),
            tau_confidence=self.cfg.tau_confidence,
            adaptive_threshold_scale=self.cfg.adaptive_threshold_scale,
        )

        # --- Pass-1 confidence ---
        ent_ids_1 = [e["id"] for e in ctx_entities]
        rel_ids_1 = [r["rel_id"] for r in ctx_rels]
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
            self._last_adaptive_trace = self._build_adaptive_trace(
                pass2_triggered=False,
                pass1_entity_ids=ent_ids_1,
                pass1_relation_ids=rel_ids_1,
                conf_pass1=conf_1,
                conf_final=conf_1,
            )
            return ctx_entities, ctx_rels, ctx_text, query_vec

        # --- Rewrite query ---
        try:
            rewrite_llm = build_adaptive_llm_client()
            rewritten_q, rewrite_latency = rewrite_query(
                question, ctx_entities, ctx_rels, conf_1, rewrite_llm
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
            self._last_adaptive_trace = self._build_adaptive_trace(
                pass2_triggered=False,
                pass1_entity_ids=ent_ids_1,
                pass1_relation_ids=rel_ids_1,
                conf_pass1=conf_1,
                conf_final=conf_1,
                rewritten_query=rewritten_q,
                adaptive_skip_reason="rewrite_identical",
            )
            return ctx_entities, ctx_rels, ctx_text, query_vec

        # --- Pass-2 graph ---
        local_graph = None
        try:
            local_graph = build_adaptive_graph()
            _jlog("adaptive_graph_opened", request_id, step="2b")
        except OSError as exc:
            _jlog("adaptive_graph_error", request_id, step="2b", error=str(exc))

        try:
            # --- Pass-2 keywords ---
            kw2 = self.generate_query_keywords(rewritten_q, request_id=request_id)

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
            ctx2_entities, ctx2_rels, ctx2_text, query_vec2 = self.assemble_context_from_query(
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

        ent_ids_2 = [e["id"] for e in ctx2_entities]
        rel_ids_2 = [r["rel_id"] for r in ctx2_rels]
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
            context_length=len(ctx2_text),
            query_vec_dim=int(query_vec2.shape[0]) if hasattr(query_vec2, "shape") else None,
            elapsed_sec=timer_p2.sec(),
        )

        # --- Additive merge: keep all pass-1 context, append only novel pass-2 items ---
        merged_entities, merged_rels, merged_text, conf_merged = self._additive_merge(
            entities_1=ctx_entities,
            rels_1=ctx_rels,
            entities_2=ctx2_entities,
            rels_2=ctx2_rels,
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
        self._last_adaptive_trace = self._build_adaptive_trace(
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

    def _additive_merge(
        self,
        *,
        entities_1: list[dict],
        rels_1: list[dict],
        entities_2: list[dict],
        rels_2: list[dict],
        request_id: str | None,
        conf_1: float,
        conf_2: float,
        query_vec: Any = None,
    ) -> tuple[list[dict], list[dict], str, float]:
        """
        Additive context merge: preserve all pass-1 results and append only
        pass-2 items whose IDs were not already retrieved in pass-1.

        This ensures the answer-bearing context from pass-1 is never displaced.
        Novel entities are filtered by their similarity to the original query_vec
        (threshold: cfg.novel_ent_threshold) to discard noise introduced by
        rewrite drift.  Novel rels are admitted unconditionally since they are
        anchored to already-filtered entities.

        Returns:
            (merged_entities, merged_rels, merged_text, conf_merged)
        """
        # Collect pass-1 IDs to identify novel pass-2 items
        ent_ids_1: set = {e["id"] for e in entities_1}
        rel_ids_1: set = {r["rel_id"] for r in rels_1}

        novel_ents_raw = [e for e in entities_2 if e["id"] not in ent_ids_1]
        novel_rels = [r for r in rels_2 if r["rel_id"] not in rel_ids_1]

        # Filter novel entities by similarity to the original question embedding
        if query_vec is not None and novel_ents_raw:
            ent_vdb = self.MGR.get_entities_vdb(0)
            novel_ents = [
                e for e in novel_ents_raw
                if (res := ent_vdb.compare_by_id(e["id"], query_vec, threshold=0.0))
                is not None and res[1] >= self.cfg.novel_ent_threshold
            ]
        else:
            novel_ents = novel_ents_raw

        merged_entities = entities_1 + novel_ents
        merged_rels = rels_1 + novel_rels

        # Confidence unchanged from pass-1 (pass-1 context is fully preserved)
        conf_merged = conf_1

        _jlog(
            "adaptive_merge_rerank",
            request_id,
            step="2b",
            entities_pass1=len(entities_1),
            entities_pass2=len(entities_2),
            rels_pass1=len(rels_1),
            rels_pass2=len(rels_2),
            novel_entities_raw=len(novel_ents_raw),
            novel_entities=len(novel_ents),
            novel_rels=len(novel_rels),
            merged_entities=len(merged_entities),
            merged_rels=len(merged_rels),
            conf_1=conf_1,
            conf_2=conf_2,
            conf_merged=conf_merged,
            novel_ent_threshold=self.cfg.novel_ent_threshold,
        )

        merged_text = self._render_context_text(
            entities=merged_entities,
            relationships=merged_rels,
            request_id=request_id,
        )
        return merged_entities, merged_rels, merged_text, conf_merged

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
            rewritten_question = _maybe_rewrite_retrieval_question(
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
            if _env_flag_enabled("KG_ABLATION_NO_KEYWORDS"):
                _jlog("ablation_no_keywords_applied", request_id, step="1")
                kw = KeywordExtractionResult(high_level_keywords=[], low_level_keywords=[])
            else:
                kw = self.generate_query_keywords(
                    rewritten_question, request_id=request_id,
                    retrieval_guidance=retrieval_guidance,
                )

            # 2) Assemble context (entities, relationships, text)
            timer_context = _StepTimer()
            ctx_entities, ctx_rels, ctx_text, query_vec = self.assemble_context_from_query(
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
                entity_count=len(ctx_entities),
                relationship_count=len(ctx_rels),
                has_context=bool(ctx_text),
                elapsed_sec=timer_context.sec(),
            )

            # 2b) Adaptive re-search (pass 2) — off by default
            if self.cfg.enable_adaptive_search:
                ctx_entities, ctx_rels, ctx_text, query_vec = self._adaptive_research(
                    question=question,
                    ctx_entities=ctx_entities,
                    ctx_rels=ctx_rels,
                    ctx_text=ctx_text,
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
            # relationship text blocks from the context. ctx_entities/ctx_rels are
            # kept (the evidence provenance channel is untouched), so the answering
            # model sees the Evidence block alone.
            if _env_flag_enabled("KG_ABLATION_NO_KG_TEXT"):
                _jlog(
                    "ablation_no_kg_text_applied",
                    request_id,
                    step="2.9",
                    dropped_ctx_text_chars=len(ctx_text or ""),
                )
                ctx_text = ""

            # 2.9) Ablation B2 (no-KG baseline): drop the graph channel entirely --
            # the evidence pool is left with direct vector search alone, and the
            # context carries no entity/relationship text blocks either.
            # The env-var convention follows KG_NARROWING_ENABLED / USE_GREP_AGENT:
            # off by default.
            if _env_flag_enabled("KG_ABLATION_NO_GRAPH"):
                _jlog(
                    "ablation_no_graph_applied",
                    request_id,
                    step="2.9",
                    dropped_entities=len(ctx_entities),
                    dropped_rels=len(ctx_rels),
                    dropped_ctx_text_chars=len(ctx_text or ""),
                )
                ctx_entities, ctx_rels, ctx_text = [], [], ""

            # 3) Build evidence block
            timer_evidence = _StepTimer()
            # 0d) HyDE: hypothetical-summary vector to blend into summary scoring
            hyde_vec = None
            if self.cfg.summary_hyde_enable:
                hyde_vec = self.generate_hyde_vector(rewritten_question, request_id=request_id)
            _scoring_weights = ScoringWeights(
                relation_weight=self.cfg.summary_relation_weight,
                entity_weight=self.cfg.summary_entity_weight,
                pair_bonus_weight=self.cfg.summary_pair_bonus_weight,
                semantic_weight=self.cfg.summary_semantic_weight,
                popularity_penalty_weight=self.cfg.summary_popularity_penalty_weight,
                redundancy_penalty_weight=self.cfg.summary_redundancy_penalty_weight,
                enable_pair_bonus=self.cfg.summary_enable_pair_bonus,
                enable_popularity_penalty=self.cfg.summary_enable_popularity_penalty,
                enable_redundancy_penalty=self.cfg.summary_enable_redundancy_penalty,
                rrf_k=self.cfg.summary_rrf_k,
            )
            # Ablation A: KG_ABLATION_NO_DIRECT_VECTOR=1 -> config closes the direct
            # search channel down to topn=0 (add_direct becomes a no-op); all that is
            # left here is a signal the smoke test can assert on.
            if _env_flag_enabled("KG_ABLATION_NO_DIRECT_VECTOR"):
                _jlog(
                    "ablation_no_direct_vector_applied",
                    request_id,
                    step="3",
                    summary_direct_vector_topn=self.cfg.summary_direct_vector_topn,
                    summary_direct_vector_min_score=self.cfg.summary_direct_vector_min_score,
                )
            evidence_block = self.evidence_builder.build_evidence_block(
                context_entities=ctx_entities,
                context_relationships=ctx_rels,
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
                summary_filter_mode=self.cfg.summary_filter_mode,
                scoring_weights=_scoring_weights,
                hyde_vec=hyde_vec,
                hyde_weight=self.cfg.summary_hyde_weight,
                hyde_mode=self.cfg.summary_hyde_mode,
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
                context_text_length=len(ctx_text or ""),
                evidence_length=len(evidence_block or ""),
            )
            base_text = ctx_text or "(no KG context)"
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
                "pass1_entity_ids": adaptive_trace.get("pass1_entity_ids", [entity["id"] for entity in ctx_entities]),
                "pass2_entity_ids": adaptive_trace.get("pass2_entity_ids", []),
                "pass1_relation_ids": adaptive_trace.get("pass1_relation_ids", [relationship["rel_id"] for relationship in ctx_rels]),
                "pass2_relation_ids": adaptive_trace.get("pass2_relation_ids", []),
                "entity_overlap_count": adaptive_trace.get("entity_overlap_count", 0),
                "entity_overlap_pct": adaptive_trace.get("entity_overlap_pct"),
                "relation_overlap_count": adaptive_trace.get("relation_overlap_count", 0),
                "relation_overlap_pct": adaptive_trace.get("relation_overlap_pct"),
                "final_entity_ids": [entity["id"] for entity in ctx_entities],
                "final_entity_names": [entity.get("name") or entity.get("id") for entity in ctx_entities],
                "final_relationship_ids": [relationship["rel_id"] for relationship in ctx_rels],
                "final_relationship_names": self._relationship_names_from_ids([relationship["rel_id"] for relationship in ctx_rels]),
                "final_entity_count": len(ctx_entities),
                "final_relationship_count": len(ctx_rels),
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
            }
            return "(no KG context)"
