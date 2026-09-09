"""Every knob the Retriever reads, grouped by the stage that reads it.

The groups are the Retriever's stage boundaries made explicit: search decides
what enters the candidate pool, filtering decides what survives, evidence
decides what the LLM sees. A knob that fits none of them means a boundary has
drifted. `AdaptiveConfig` is the exception and says so: its stage is gone and
its fields are kept only so old configs still load.

`RetrieverConfig` inherits from all four instead of nesting them, which is
load-bearing:

  * experiment_config.py splats flat dicts in as `RetrieverConfig(**params)`.
    Nesting would break every call site and make every key a two-level path.
  * The Retriever reads `self.cfg.ent_topk` in 88 places. Inheritance keeps
    that flat, while a future component can still accept just `SearchConfig`
    and be handed the whole `RetrieverConfig` -- because it is one.

Field names *are* config keys. Renaming a field renames the key used by
experiment_config.py, every sweep script, and every recorded run metadata file.

The KG_ABLATION_* switches live at the bottom of this module. They are
configuration too -- each one removes a retrieval channel so a run can be
compared against the same system without it -- they just arrive through the
environment rather than through a config dict.
"""

import os
import warnings
from dataclasses import dataclass, fields

#: Knobs that are still accepted and still recorded, but no longer reach any
#: decision. Field -> why it stopped mattering.
#:
#: They are kept rather than deleted because run metadata, sweep scripts and
#: trace readers written against older runs still carry them; removing the
#: fields would turn those into a TypeError. What they must not do is stay
#: silent: a sweep over `reranker_topk` that produces a different metadata file
#: and an identical result set is how a research conclusion goes wrong. Setting
#: of them away from its default now warns once, and the set is recorded in the
#: `retriever_initialized` log for every run.
INERT_FIELDS = {
    "filter_ent_topk": "stage 3 does no cutting; the reranker is the filter",
    "filter_rel_topk": "stage 3 does no cutting; the reranker is the filter",
    "filter_ent_threshold": "stage 3 does no cutting; the reranker is the filter",
    "filter_rel_threshold": "stage 3 does no cutting; the reranker is the filter",
    "use_reranker": "stage 4 always reranks; there is no non-reranked path left",
    "reranker_threshold": "superseded by rrk_threshold",
    "reranker_topk": "superseded by rrk_ent_topk / rrk_rel_topk",
    "sa_max_activated": "spreading activation no longer caps its result set",
    "enable_adaptive_search": "adaptive re-search was removed; there is no pass 2",
    "adaptive_threshold_scale": "adaptive re-search was removed; there is no pass 2",
    "novel_ent_threshold": "adaptive re-search was removed; there is no pass 2",
}

_warned: set[tuple[str, ...]] = set()


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
    sa_max_activated: int = 20   # inert: see INERT_FIELDS
    # Keyword source for relationship vector search:
    # "high_level" (abstract reasoning words, baseline) | "low_level" (concrete anchors) | "both"
    relation_search_keywords: str = "high_level"


@dataclass(frozen=True)
class FilterConfig:
    """Stages 3-4: how the candidate pool is narrowed and reranked.

    The reranker is the filter: what survives the intersection goes to the
    cross-encoder, and the rrk_* knobs are what shape the cut. Everything above
    them is inert -- accepted, recorded, and without effect. See INERT_FIELDS.
    """

    # post-intersection filtering -- inert, see INERT_FIELDS
    filter_ent_topk: int = 3
    filter_rel_topk: int = 3
    filter_ent_threshold: float = 0.5
    filter_rel_threshold: float = 0.5
    # the pre-rrk_* reranker knobs -- inert, see INERT_FIELDS
    use_reranker: bool = True
    reranker_threshold: float = -3.0
    reranker_topk: int = 5
    # Reranker-only — active for "reranker_only"
    rrk_ent_topk: int = 5          # max entities to keep
    rrk_rel_topk: int = 5          # max relationships to keep
    rrk_threshold: float = 0.0     # score cutoff — 0.0 means "Yes logit > No logit"


@dataclass(frozen=True)
class EvidenceConfig:
    """How surviving candidates become the Evidence block the LLM reads.

    The largest group, because two separable decisions grew together: which
    text to return for a summary (raw turn, summary, or the :u/:a split), and
    how many survive (top-k, direct-vector, rerank). If this file splits
    further, the seam runs between them.
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
    # Per-entity quota: guarantee this many snippets per source entity/relationship
    # before filling the remaining top-K slots by score. 0 = disabled.
    summary_per_entity_min: int = 0


@dataclass(frozen=True)
class AdaptiveConfig:
    """What is left of pass-2 re-search, which no longer exists.

    The capability was removed: nothing reads these and no second pass runs.
    The fields stay because deleting a field turns every archived run's metadata
    and every sweep script that names it into a TypeError -- the same reason
    INERT_FIELDS exists, which is where three of them now are.

    `tau_confidence` is not inert-flagged for a different reason: it is still
    written to the `tau_confidence` column of both benchmarks' answer CSVs, so
    its default has to keep producing 0.70.
    """

    enable_adaptive_search: bool = False
    tau_confidence: float = 0.70
    adaptive_threshold_scale: float = 0.8
    novel_ent_threshold: float = 0.35


@dataclass(frozen=True)
class RetrieverConfig(SearchConfig, FilterConfig, EvidenceConfig, AdaptiveConfig):
    """Every knob, flat, as the experiment configs and the Retriever expect it.

    Adds nothing of its own. It exists so the four groups can be named
    separately while callers keep one object with one flat namespace.
    """

    def __post_init__(self) -> None:
        overrides = self.inert_overrides()
        if not overrides:
            return
        # Once per distinct set, not once per Retriever: LoCoMo builds one per
        # sample, and a warning repeated 10 times is a warning nobody reads.
        key = tuple(sorted(overrides))
        if key in _warned:
            return
        _warned.add(key)
        warnings.warn(
            "these retrieval knobs no longer affect selection and were ignored: "
            + "; ".join(f"{name}={value!r} ({INERT_FIELDS[name]})"
                        for name, value in sorted(overrides.items())),
            FutureWarning,
            stacklevel=3,
        )

    def inert_overrides(self) -> dict:
        """The inert knobs this config sets away from their default.

        Empty for a config that only sets knobs that still do something. The
        `retriever_initialized` log records it so a run's own trace says which
        of its configured values were decoration.
        """
        defaults = {f.name: f.default for f in fields(RetrieverConfig)}
        return {
            name: getattr(self, name)
            for name in INERT_FIELDS
            if getattr(self, name) != defaults[name]
        }


# ===========================================================================
# Ablation switches
# ===========================================================================
#
# Each flag removes one retrieval channel so a run can be compared against the
# same system without it. They were read in three modules with three slightly
# different expressions; the names live here so the set of ablations is
# something you can look up rather than grep for.
#
# The reader deliberately does not `.strip()`. Two of the three call sites it
# replaces did not, and `grace_mem.temporal.normalizer` -- which does, and also
# carries a legacy alias -- keeps its own reader rather than have this one
# quietly start accepting `" 1 "` where it used to reject it.

#: Every ablation switch, and the channel it removes.
ABLATIONS = {
    "KG_ABLATION_NO_BM25": "lexical half of hybrid entity search",
    "KG_ABLATION_NO_DIRECT_VECTOR": "direct summary-vector retrieval",
    "KG_ABLATION_NO_GRAPH": "the graph channel entirely",
    "KG_ABLATION_NO_KEYWORDS": "LLM keyword extraction",
    "KG_ABLATION_NO_KG_TEXT": "entity/relationship text, keeping the graph",
    "KG_ABLATION_NO_TEMPORAL_BOOST": "temporal containment reranking",
    "KG_ABLATION_NO_TIME_REWRITE": "query-side temporal rewriting",
}


def flag_enabled(name: str) -> bool:
    """True when `name` is set to anything other than 0, empty, or false."""
    return os.getenv(name, "0").lower() not in ("0", "", "false")
