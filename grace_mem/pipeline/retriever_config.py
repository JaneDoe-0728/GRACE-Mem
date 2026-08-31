"""Every knob the Retriever reads, in one frozen dataclass.

Kept out of retriever.py because it is data, not behaviour: it changes when an
experiment sweeps a parameter, while the retrieval code around it does not.

The field groups below -- initial search, post-intersection filtering, the
filter_method dispatch and its RRF/PPR parameters, evidence assembly, summary
scoring, adaptive re-search -- are the responsibility boundaries of the
Retriever itself. If this ever splits into SearchConfig / FilterConfig /
EvidenceConfig / AdaptiveConfig, those groups are where the seams are.

Field names are load-bearing: experiment/experiment_config.py's
RETRIEVAL_PARAMS and RERANKER_PARAMS keys match them exactly, and are splatted
in as **kwargs. Renaming a field renames a config key.
"""

from dataclasses import dataclass


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
