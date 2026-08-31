"""Every knob the Retriever reads, grouped by the stage that reads it.

The four groups are the Retriever's stage boundaries made explicit: search
decides what enters the candidate pool, filtering decides what survives,
evidence decides what the LLM sees, adaptive decides whether to go round again.
A knob that fits none of them means a boundary has drifted.

`RetrieverConfig` inherits from all four instead of nesting them, which is
load-bearing:

  * experiment_config.py splats flat dicts in as `RetrieverConfig(**params)`.
    Nesting would break every call site and make every key a two-level path.
  * The Retriever reads `self.cfg.ent_topk` in 88 places. Inheritance keeps
    that flat, while a future component can still accept just `SearchConfig`
    and be handed the whole `RetrieverConfig` -- because it is one.

Field names *are* config keys. Renaming a field renames the key used by
experiment_config.py, every sweep script, and every recorded run metadata file.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchConfig:
    """Stage 1: how wide the initial entity/relationship search casts.

    Nothing downstream can recover an entity that was never retrieved here, so
    loosen these topk/threshold pairs first when gold evidence is missing
    entirely rather than merely ranked badly.
    """

    # entity initial search
    ent_topk: int = 5
    ent_threshold: float = 0.3
    # relationship initial search
    rel_topk: int = 5
    rel_threshold: float = 0.3
    # spreading activation
    use_spreading_activation: bool = False
    sa_max_hops: int = 2
    sa_rescale_c: float = 0.4
    sa_tau_a: float = 0.5
    sa_max_activated: int = 20
    # Keyword source for relationship vector search:
    # "high_level" (abstract reasoning words, baseline) | "low_level" (concrete anchors) | "both"
    relation_search_keywords: str = "high_level"


@dataclass(frozen=True)
class FilterConfig:
    """Stages 3-4: how the candidate pool is narrowed and reranked.

    `filter_method` is the single axis everything hangs off, and each value
    activates a different subset. The rrf_*/ppr_*/rrk_* knobs are inert under a
    method that ignores them -- deliberately: one flat set swept by one flag
    beats five parallel config objects.
    """

    # post-intersection filtering
    filter_ent_topk: int = 3
    filter_rel_topk: int = 3
    filter_ent_threshold: float = 0.5
    filter_rel_threshold: float = 0.5
    # reranker for recovering filtered items
    use_reranker: bool = True
    reranker_threshold: float = -3.0
    reranker_topk: int = 5
    # Step 2.6 filter method — single axis for ablation
    filter_method: str = "similarity"      # "similarity" | "rrf" | "ppr" | "rrf+ppr" | "reranker_only"
    # RRF — active for "rrf", "rrf+ppr"
    rrf_k: float = 60.0
    rrf_candidate_k: int = 50             # RRF top-N fed into PPR ("rrf+ppr" only)
    # PPR — active for "ppr", "rrf+ppr"
    ppr_alpha: float = 0.85
    ppr_top_k: int = 10
    ppr_inverse_degree: bool = False
    # Reranker-only — active for "reranker_only"
    rrk_ent_topk: int = 5          # max entities to keep
    rrk_rel_topk: int = 5          # max relationships to keep
    rrk_threshold: float = 0.0     # score cutoff — 0.0 means "Yes logit > No logit"


@dataclass(frozen=True)
class EvidenceConfig:
    """How surviving candidates become the Evidence block the LLM reads.

    The largest group, because three separable decisions grew together: which
    text to return for a summary (raw turn, summary, or the :u/:a split), which
    return for them (raw turn vs summary vs the :u/:a split), and how many
    survive (top-k, direct-vector, rerank). If this file splits further, the
    seam runs through here.
    """

    summary_embed_dim: int = 1024
    # evidence
    summary_topk_per_item: int = 5
    summary_vec_threshold: float = 0.4
    use_full_summary: bool = True
    fallback_to_raw: bool = False
    # ── Which text comes back ─────────────────────────────────────────────────
    # Score and rank on summary vectors as usual, but return the raw turn text
    # for each selected snippet instead of the summary text.
    use_raw_context: bool = False
    # script_data directory holding the raw CSV conversations.
    # Required when use_raw_context or use_split_embeddings is True.
    raw_context_data_dir: str = ""
    # Select evidence per VDB entry rather than per turn, via
    # EvidenceBuilder._build_evidence_split. Mutually exclusive with use_raw_context.
    # Entry granularity comes from split_single_entry_raw below.
    # Default True so build_pipeline() takes the same path as the benchmark
    # pipelines (RERANKER_PARAMS sets it too); the turn-level branch survives only
    # for artifacts predating split selection.
    use_split_embeddings: bool = True
    # ── Candidate pool size (split-embedding mode) ────────────────────────────
    # Pull the top-N summaries by raw query similarity straight from the VDB and
    # merge them into the pool, alongside entity/relationship spreading activation.
    # Recovers high-similarity gold summaries whose turn links to no retrieved
    # entity, so they never enter the prov-based pool. 0 = disabled.
    summary_direct_vector_topn: int = 0
    # Min raw query-similarity for a direct-vector hit to earn an EXTRA evidence
    # slot (on top of the prov top-K, not competing for it). Needs
    # summary_direct_vector_topn > 0. 0.0 = extra-slot mode off.
    summary_direct_vector_min_score: float = 0.0
    # Retrieve-then-rerank: cross-encode the whole pool (prov + direct above the
    # min-score floor) and keep the top-N. Supersedes the extra-slot path.
    # 0 = disabled.
    summary_rerank_topk: int = 0
    # Ablation: in the rerank path, skip the cross-encoder and keep cosine top-N.
    summary_rerank_cosine_only: bool = False
    # Single-entry raw mode for the split path (e.g. LoCoMo): the VDB holds one
    # entry per summary_id (no :u/:a suffixes) and the LLM is fed raw turn text
    # (raw_text metadata), not the compressed summary. Lets the rerank16 flow
    # (direct-vector + cross-encoder) run on datasets without the :u/:a scheme.
    # Default True because Ingestor.summarize_and_ingest_turn writes exactly one
    # entry per summary_id and never writes :u/:a; those pairs come only from a
    # LongMem post-processing pass (experiment/longmem/tools/rebuild_split_summaries.py).
    # So only the LongMem pipeline sets False, deriving it from
    # INGEST_PARAMS["use_split_summary"] — the same flag that decides whether the
    # rebuild ran. False against never-rebuilt artifacts makes every provenance
    # candidate miss silently.
    split_single_entry_raw: bool = True
    # ── HyDE summary retrieval ────────────────────────────────────────────────
    # Embed hypothetical answer sentences and blend their summary similarity with
    # the query's: score = (1-w)*sim_query + w*sim_hyde.
    summary_hyde_enable: bool = False
    summary_hyde_weight: float = 0.3       # w in the blend; 0.0 reproduces baseline
    summary_hyde_mode: str = "blend"       # "blend" (compete for top-K) | "fill" (backfill unused slots only)
    # Per-entity quota: guarantee this many snippets per source entity/relationship
    # before filling the remaining top-K slots by score. 0 = disabled.
    summary_per_entity_min: int = 0


@dataclass(frozen=True)
class AdaptiveConfig:
    """Pass-2 re-search: ask again when the first pass looks unconvincing.

    Off by default. `tau_confidence` is the trigger; the other three shape what
    pass 2 may do differently from pass 1.
    """

    # adaptive re-search (off by default — enable per call or via custom config)
    enable_adaptive_search: bool = False
    tau_confidence: float = 0.70           # trigger threshold
    adaptive_threshold_scale: float = 0.8  # filter threshold multiplier for pass-2
    novel_ent_threshold: float = 0.35      # min similarity to the original query_vec to admit a novel entity


@dataclass(frozen=True)
class RetrieverConfig(SearchConfig, FilterConfig, EvidenceConfig, AdaptiveConfig):
    """Every knob, flat, as the experiment configs and the Retriever expect it.

    Adds nothing of its own. It exists so the four groups can be named
    separately while callers keep one object with one flat namespace.
    """
