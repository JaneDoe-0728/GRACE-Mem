"""
Single source of truth for ALL experiment parameters (ingestion + retrieval + reranker).
Change values here — all experiment scripts will use them automatically.
"""

# ── Reproducibility parameters ──────────────────────────────────────────────
# Central seed/determinism defaults used by LongMem + LoCoMo entrypoints.
# This improves reproducibility, but it does not guarantee bit-level identical
# outputs under concurrency, GPU nondeterminism, or backend/server-side batching.
REPRODUCIBILITY_PARAMS = dict(
    seed=42,
    deterministic=True,
)

# ── Ingestion parameters ───────────────────────────────────────────────────
INGEST_PARAMS = dict(
    ingest_mode="turn_pairs",
    prev_k=2,
    entity_sim_topk=3,
    entity_sim_threshold=0.6,
)

# ── Parameters passed to build_kg_context() per call ──────────────────────
RETRIEVAL_PARAMS = dict(
    # Initial search
    ent_topk=20,
    rel_topk=10,
    ent_threshold=0.2,
    rel_threshold=0.2,
    # Post-intersection filtering
    filter_ent_topk=15,
    filter_rel_topk=15,
    filter_ent_threshold=0.3,
    filter_rel_threshold=0.3,
    summary_topk_per_item=16,
    summary_vec_threshold=0.2,
)

# ── Parameters set at Retriever init time (reranker + spreading activation) ──
RERANKER_PARAMS = dict(
    use_reranker=True,
    reranker_threshold=-3.0,
    reranker_topk=10,
    # Filter policy / graph reranking
    filter_method="reranker_only",
    rrk_ent_topk=25,
    rrk_rel_topk=25,
    rrk_threshold=-100.0,
    rrf_k=60.0,
    rrf_cosine_weight=1.0,
    rrf_candidate_k=50,
    ppr_alpha=0.85,
    ppr_top_k=10,
    ppr_inverse_degree=False,
    # Spreading Activation (SA-RAG)
    use_spreading_activation=True,
    sa_max_hops=2,
    sa_rescale_c=0.4,
    sa_tau_a=0.5,
    sa_max_activated=20,
    # ── Summary selection strategy ─────────────────────────────────────────
    # "semantic"               → baseline cosine-similarity ranking
    # "graph_count"            → graph link counts only
    # "graph_semantic"         → graph counts + weak semantic tie-breaker
    # "graph_semantic_penalty" → graph + semantic + popularity/redundancy penalties
    # "graph_weighted_sum"     → alias for full weighted-sum (all terms)
    # "graph_rrf"              → RRF rank-fusion scoring
    # "graph_rrf_mmr"          → RRF + iterative MMR redundancy control
    summary_filter_mode="semantic",
    summary_relation_weight=2.0,
    summary_entity_weight=1.0,
    summary_pair_bonus_weight=1.5,
    summary_semantic_weight=0.5,
    summary_popularity_penalty_weight=1.0,
    summary_redundancy_penalty_weight=1.0,
    summary_enable_pair_bonus=False,
    summary_enable_popularity_penalty=False,
    summary_enable_redundancy_penalty=False,
    # RRF-specific (used when summary_filter_mode in {"graph_rrf", "graph_rrf_mmr"})
    summary_rrf_k=60.0,
    summary_enable_mmr_redundancy=False,
    # ── HyDE summary retrieval ─────────────────────────────────────────────
    summary_hyde_enable=False,
    summary_hyde_weight=0.1,
    summary_hyde_mode="fill",
    # ── Relationship vector search keyword source ──────────────────────────
    # "high_level" → abstract reasoning words (baseline; noisy for rel search)
    # "low_level"  → concrete entity/topic anchors
    # "both"       → union of both
    relation_search_keywords="low_level",
    # ── Per-entity evidence quota ──────────────────────────────────────────
    # Guarantee at least this many snippets per source entity/relationship
    # before filling remaining top-K slots by score (0 = disabled).
    summary_per_entity_min=1,
    # ── Raw context mode ───────────────────────────────────────────────────
    # When True, summary vectors are still used for top-K selection, but the
    # text returned for each selected snippet is the raw turn text instead of
    # the summary text.
    use_raw_context=False,
    raw_context_data_dir="experiment/longmem/script_data",
    # ── Split embedding mode ──────────────────────────────────────────────
    # VDB must have :u/:a entries (built by rebuild_split_summaries.py).
    # To activate: set use_raw_context=False, use_split_embeddings=True.
    use_split_embeddings=True,
    # Direct summary vector retrieval merged into the candidate pool, parallel to
    # entity/relationship spreading activation. Recovers high-similarity gold whose
    # turn is not linked to any retrieved entity. 0 = disabled.
    summary_direct_vector_topn=50,
    # Extra-slot mode: direct hits above this raw cosine are added ON TOP of the
    # prov top-K (do not compete for it). 0.0 = disabled. Swept in experiments.
    summary_direct_vector_min_score=0.35,
    # Retrieve-then-rerank: rerank the wide pool and keep top-N.
    summary_rerank_topk=16,
    summary_rerank_cosine_only=False,
    # rerank16: one entry per summary_id (no :u/:a), feed raw turn text.
    # 此共用值供 LoCoMo 使用(True);LongMem 在 processor.py/rerun.py 強制覆寫為
    # False(:u/:a split 路徑,artifacts 需先跑 rebuild_split_summaries.py,
    # oss-20b-0427 已於 2026-07-04 重建完成)。
    split_single_entry_raw=True,
)

# ── Grep agent (LongMem post-retrieval evidence refinement) ────────────────
# Runs an inline grep mini-harness AFTER vector+rerank top-16: the agent
# verifies candidates and greps the raw per-question corpus for missing
# literal evidence, then the Evidence Summary block is rebuilt from the
# selected sids (raw turn text). Fail-safe: any agent failure keeps the
# original context unchanged.
GREP_AGENT_PARAMS = dict(
    use_grep_agent=True,
    # "filter"       → agent may only keep/drop the retrieved candidates (precision only)
    # "filter_fetch" → agent may also add corpus sids found via GREP (precision + recall)
    grep_agent_mode="filter_fetch",
    grep_agent_max_calls=10,
    grep_agent_max_sids=16,
    grep_agent_grep_max_lines=30,
    # provenance gate 已於 2026-07-22 移除(harness.py):VECTOR 命中現與 GREP/READ
    # 同視為 verified,此參數已成 no-op,保留僅為相容舊 trace/腳本引用。
    grep_agent_require_verified_additions=True,
    # Entity/Relationship graph facts are independently switchable on the
    # filter prompt and on the final answer context for ablation experiments.
    grep_agent_filter_include_graph_context=False,
    grep_agent_answer_include_graph_context=True,
    grep_agent_graph_context_max_chars=12000,
    # evidence_floor 盲補按 rerank 原序硬塞、繞過 agent 決定,對 accuracy 零貢獻
    # (見 docs/analysis/grep-vs-adjudicate-cross-model.md §三)。全域關閉:0=不盲補。
    grep_agent_evidence_floor=0,
    # 選中 sid 的 pair 夥伴(同一 exchange 的另一側)一併放入最終 context,
    # 修復「agent 選到正確 pair 的錯誤一側」的失誤。
    grep_agent_include_pair=True,
    # ── Answer-blind 逐條裁決(自問自答對策,2026-07-16)──────────────────
    # FINAL 後用獨立裁決 call(fresh conversation,看不到 agent 已推出的答案=
    # answer-blind)對每條被丟掉的 seed 逐一 KEEP/DROP,判準=與問題主題相關。
    # KEEP 補回(只加不刪)。1=開,0=關。
    grep_agent_adjudicate=1,
    # 單針類 agent 天性已最優 → 預設僅多證據類。None=全類(ablation 用)。
    grep_agent_adjudicate_categories=(
        "single_session_preference", "multi_session",
        "temporal_reasoning", "knowledge_update",
    ),
    # KEEP-all 類別:裁決改成 recall-recovery-only(被丟 seed 全補回)。() = 關。
    grep_agent_adjudicate_keep_all_categories=(),
    # 強制 verified→FINAL:no_final 時把 verified sids 當 FINAL 走 finalize。0=關。
    grep_agent_force_verified_final=0,
    # 窄化 gate 門檻:verified 數 >= 此值(且非 _abs)才走 finalize 窄化。
    grep_agent_force_verified_min=12,
    # ── Sufficiency 迴圈 ──────────────────────────────────────────────────
    # FINAL 後由獨立 verifier 判斷證據是否足以完整回答;不足則帶著「缺什麼」
    # 的 hint 讓 agent 補搜(只加不刪,單調遞增)。0 = 關閉。
    grep_agent_verify_rounds=0,  # v3-v6 四變體皆 ≤ v2:sufficiency 蓋棺,預設關(見 agent_filter/README)
    grep_agent_verify_max_calls=4,
    grep_agent_verify_categories=("multi_session", "knowledge_update"),
    # 缺口向量補搜:verifier 判不足時,把「question+missing」embed 查 summaries VDB,
    # 撈 grep 搆不到的語意近鄰給 agent 確認(修 paraphrase gap)。0 = 關閉。
    grep_agent_gap_vector_topn=6,
    grep_agent_gap_vector_min_score=0.30,
    # Min-keep(問題驅動):彙整/最新值型問題 FINAL 少於 N 條時依 rerank 原序從 seed 補滿。
    grep_agent_min_keep_aggregation=0,  # v9 配對驗證:修復率=噪音,中性 → 預設關
    # Skill 庫:question-shape 驅動的搜尋戰術(skills.py),命中時取代 category hint。
    # 2026-07-22 預設關:hint 與 filter_fetch 解耦,base 不注入 skill hint。
    grep_agent_use_skills=False,
    # ── VECTOR 工具(agent 主動語意搜尋)──────────────────────────────────
    # 給 agent 一個 VECTOR <query> 指令直接查該題 summaries VDB(artifact_dir
    # 有 summaries_chroma 才啟用)。命中僅為 discovery lead,需 GREP/READ 驗證。
    grep_agent_vector_search=True,
    grep_agent_vector_topn=8,
    grep_agent_vector_min_score=0.30,
)
