"""
Single source of truth for ALL experiment parameters (ingestion + retrieval + reranker).
Change values here — all experiment scripts will use them automatically.
"""

# ── Reproducibility parameters ──────────────────────────────────────────────
# Central seed/determinism defaults used by LongMem + LoCoMo entrypoints.
# This improves reproducibility, but it does not guarantee bit-level identical
# outputs under concurrency, GPU nondeterminism, or backend/server-side batching.
REPRODUCIBILITY_PARAMS = {
    "seed": 42,
    "deterministic": True,
}

# ── Ingestion parameters ───────────────────────────────────────────────────
INGEST_PARAMS = {
    "ingest_mode": "turn_pairs",
    "prev_k": 2,
    "entity_sim_topk": 3,
    "entity_sim_threshold": 0.6,
    # ── LoCoMo only: turns per ingest chunk ────────────────────────────────
    # Each session is split into consecutive windows of this many turns, and each
    # window becomes its own summary/ingest unit (message_id = chunk index), giving
    # a finer summary-retrieval pool. 0 = one summary per whole session (the pre-
    # chunking behaviour; set this to reproduce older runs).
    # NOTE: the evidence knobs in RERANKER_PARAMS below (summary_direct_vector_topn,
    # summary_rerank_topk) were tuned against whole-session summaries. Changing
    # chunk_turns changes the candidate pool size, so re-sweep them if you tune.
    "chunk_turns": 8,
    # ── LongMem only: :u/:a split summaries ────────────────────────────────
    # True  → after ingest, each artifact's summaries_chroma is rebuilt into
    #         {sid}:u (user raw) + {sid}:a (assistant compressed) entry pairs, and
    #         retrieval scores them as independent candidates.
    # False → keep what the Ingestor wrote (one entry per summary_id) and skip the
    #         rebuild entirely.
    # This single flag drives BOTH the rebuild step and the matching retrieval
    # setting (split_single_entry_raw = not use_split_summary), so the artifact
    # layout and the retrieval config can never disagree. LoCoMo ignores it and
    # always uses one entry per summary_id.
    "use_split_summary": True,
}

# ── Parameters passed to build_kg_context() per call ──────────────────────
RETRIEVAL_PARAMS = {
    # Initial search
    "ent_topk": 20,
    "rel_topk": 10,
    "ent_threshold": 0.2,
    "rel_threshold": 0.2,
    # Post-intersection filtering
    "filter_ent_topk": 15,
    "filter_rel_topk": 15,
    "filter_ent_threshold": 0.3,
    "filter_rel_threshold": 0.3,
    "summary_topk_per_item": 16,
    "summary_vec_threshold": 0.2,
}

# ── Parameters set at Retriever init time (reranker + spreading activation) ──
RERANKER_PARAMS = {
    "use_reranker": True,
    "reranker_threshold": -3.0,
    "reranker_topk": 10,
    "rrk_ent_topk": 25,
    "rrk_rel_topk": 25,
    "rrk_threshold": -100.0,
    # Spreading Activation (SA-RAG)
    "use_spreading_activation": True,
    "sa_max_hops": 2,
    "sa_rescale_c": 0.4,
    "sa_tau_a": 0.5,
    "sa_max_activated": 20,
    # ── HyDE summary retrieval ─────────────────────────────────────────────
    "summary_hyde_enable": False,
    "summary_hyde_weight": 0.1,
    "summary_hyde_mode": "fill",
    # ── Relationship vector search keyword source ──────────────────────────
    # "high_level" → abstract reasoning words (baseline; noisy for rel search)
    # "low_level"  → concrete entity/topic anchors
    # "both"       → union of both
    "relation_search_keywords": "low_level",
    # ── Per-entity evidence quota ──────────────────────────────────────────
    # Guarantee at least this many snippets per source entity/relationship
    # before filling remaining top-K slots by score (0 = disabled).
    "summary_per_entity_min": 1,
    # ── Raw context mode ───────────────────────────────────────────────────
    # When True, summary vectors are still used for top-K selection, but the
    # text returned for each selected snippet is the raw turn text instead of
    # the summary text.
    "use_raw_context": False,
    "raw_context_data_dir": "experiment/longmem/script_data",
    # ── Split embedding mode ──────────────────────────────────────────────
    # VDB must have :u/:a entries (built by rebuild_split_summaries.py).
    # To activate: set use_raw_context=False, use_split_embeddings=True.
    "use_split_embeddings": True,
    # Direct summary vector retrieval merged into the candidate pool, parallel to
    # entity/relationship spreading activation. Recovers high-similarity gold whose
    # turn is not linked to any retrieved entity. 0 = disabled.
    "summary_direct_vector_topn": 50,
    # Extra-slot mode: direct hits above this raw cosine are added ON TOP of the
    # prov top-K (do not compete for it). 0.0 = disabled. Swept in experiments.
    "summary_direct_vector_min_score": 0.35,
    # Retrieve-then-rerank: rerank the wide pool and keep top-N.
    "summary_rerank_topk": 16,
    "summary_rerank_cosine_only": False,
    # rerank16: one entry per summary_id (no :u/:a), feed raw turn text.
    # This shared value is what LoCoMo uses (True; LoCoMo is always a single entry).
    # LongMem overrides it in processor.py / rerun.py to (not use_split_summary),
    # following INGEST_PARAMS["use_split_summary"]: the default True is overridden
    # to False, taking the :u/:a split path, and the matching artifacts are produced
    # automatically by the rebuild step driven off that same flag.
    # Do not just set this to False here -- that would send retrieval looking for
    # :u/:a entries nobody has guaranteed exist.
    "split_single_entry_raw": True,
}

# ── Grep agent (LongMem post-retrieval evidence refinement) ────────────────
# Runs an inline grep mini-harness AFTER vector+rerank top-16: the agent
# verifies candidates and greps the raw per-question corpus for missing
# literal evidence, then the Evidence Summary block is rebuilt from the
# selected sids (raw turn text). Fail-safe: any agent failure keeps the
# original context unchanged.
GREP_AGENT_PARAMS = {
    "use_grep_agent": True,
    # "filter"       → agent may only keep/drop the retrieved candidates (precision only)
    # "filter_fetch" → agent may also add corpus sids found via GREP (precision + recall)
    "grep_agent_mode": "filter_fetch",
    "grep_agent_max_calls": 10,
    "grep_agent_max_sids": 16,
    "grep_agent_grep_max_lines": 30,
    # The provenance gate was removed on 2026-07-22 (harness.py): a VECTOR hit now
    # counts as verified just like GREP/READ, so this parameter is a no-op and is
    # kept only so older traces and scripts that reference it still work.
    "grep_agent_require_verified_additions": True,
    # Entity/Relationship graph facts are independently switchable on the
    # filter prompt and on the final answer context for ablation experiments.
    "grep_agent_filter_include_graph_context": False,
    "grep_agent_answer_include_graph_context": True,
    "grep_agent_graph_context_max_chars": 12000,
    # A selected sid brings its pair partner (the other side of the same exchange)
    # into the final context, fixing the case where the agent picks the wrong side
    # of the right pair.
    "grep_agent_include_pair": True,
    # ── Answer-blind per-item adjudication (the counter to the agent asking and
    # answering itself, 2026-07-16) ──────────────────────────────────────────
    # After FINAL, an independent adjudication call (a fresh conversation that
    # cannot see the answer the agent reached -- hence answer-blind) rules KEEP or
    # DROP on each discarded seed, judging topical relevance to the question.
    # KEEPs are added back, additive only. 1 = on, 0 = off.
    "grep_agent_adjudicate": 1,
    # The agent is already optimal by nature on single-needle categories, so this
    # defaults to the multi-evidence ones only. None = every category (for ablations).
    "grep_agent_adjudicate_categories": (
        "single_session_preference", "multi_session",
        "temporal_reasoning", "knowledge_update",
    ),
    # The skill library: question-shape driven search tactics (skills.py) that
    # replace the category hint when one fires.
    # Off by default since 2026-07-22: hints are decoupled from filter_fetch, and
    # base injects no skill hint.
    "grep_agent_use_skills": False,
    # ── VECTOR tool: semantic search the agent drives itself ─────────────────
    # Gives the agent a VECTOR <query> command that searches this question's
    # summaries VDB directly (enabled only when artifact_dir holds a
    # summaries_chroma). Hits are discovery leads only and still need GREP/READ
    # verification.
    "grep_agent_vector_search": True,
    "grep_agent_vector_topn": 8,
    "grep_agent_vector_min_score": 0.30,
}
