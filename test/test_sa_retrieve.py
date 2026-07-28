"""
SA-RAG retrieval smoke test on a LoCoMo sample.
Uses real LLM keyword extraction, skips answer generation.

Usage:
    cd /path/to/gigabyte_kg
    python test/test_sa_retrieve.py
    TEST_USE_SA=true python test/test_sa_retrieve.py
"""
import os
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from KG.graph.falkordb import graph_from_env
from KG.llm.client import LLMClient
from KG.pipeline.retriever import Retriever, RetrieverConfig
from KG.storage.chroma_manager import VDBManager
from embeddings import embedder

# ── Config ────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = REPO_ROOT / "experiment" / "multisample_runs" / "gpt-oss-20b" / "sample_0" / "artifacts"
QUESTION      = "What did Caroline research?"
GOLD          = "Adoption agencies"
USE_SA        = os.getenv("TEST_USE_SA", "true").strip().lower() in {"1", "true", "yes", "on"}
# ─────────────────────────────────────────────────────────────────────────────

# Load VDB
print(f"Loading artifacts from {ARTIFACTS_DIR} ...")
mgr = VDBManager(ARTIFACTS_DIR)
mgr.initialize()
cache = mgr.cache
print(f"  entities:      {len(cache.get('entities', {}))}")
print(f"  relationships: {len(cache.get('relationships', {}))}")

# Restore graph
graph = graph_from_env().open()
entities = cache.get("entities", {})
rels     = cache.get("relationships", {})
print(f"Syncing {len(entities)} entities and {len(rels)} rels to FalkorDB...")
graph.sync_entities(entities)
if rels:
    graph.sync_relationships(list(rels.values()))
print("Graph restored.\n")

# Build retriever
llm = LLMClient()
retriever = Retriever(
    llm=llm,
    graph=graph,
    mgr=mgr,
    embed=embedder.embed,
    cache=cache,
    config=RetrieverConfig(
        ent_topk=20,
        rel_topk=10,
        ent_threshold=0.2,
        rel_threshold=0.2,
        filter_ent_topk=10,
        filter_rel_topk=10,
        filter_ent_threshold=0.4,
        filter_rel_threshold=0.4,
        summary_topk_per_item=6,
        summary_vec_threshold=0.4,
        use_reranker=False,
        use_spreading_activation=USE_SA,
        sa_max_hops=2,
        sa_rescale_c=0.4,
        sa_tau_a=0.5,
        sa_max_activated=20,
    ),
)

print(f"Question : {QUESTION}")
print(f"Gold     : {GOLD}")
print(f"SA-RAG   : {USE_SA}\n")

# Keyword extraction
kw = retriever.generate_query_keywords(QUESTION)
print(f"High-level: {kw.high_level_keywords}")
print(f"Low-level : {kw.low_level_keywords}\n")

# Retrieval
entities_ctx, rels_ctx, ctx_text, query_vec = retriever.assemble_context_from_query(
    question=QUESTION,
    low_level_keywords=kw.low_level_keywords,
    high_level_keywords=kw.high_level_keywords,
)

evidence = retriever.evidence_builder.build_evidence_block(
    context_entities=entities_ctx,
    context_relationships=rels_ctx,
    summary_topk_global=6,
    query_vec=query_vec,
    summary_vec_threshold=0.4,
    use_full_summary=True,
    fallback_to_raw=False,
)

print("── Entities ─────────────────────────────────────────────────")
for e in entities_ctx:
    print(f"  {e.get('name')} ({e.get('type')}): {e.get('desc', '')[:80]}")

print("\n── Relationships ────────────────────────────────────────────")
for r in rels_ctx:
    print(f"  {r.get('source_name')} -> {r.get('target_name')}: {r.get('rel_desc', '')[:80]}")

print("\n── Evidence ─────────────────────────────────────────────────")
print(evidence if evidence else "(no evidence)")

print("\n── Full Context ─────────────────────────────────────────────")
full = f"{ctx_text}\n\n{evidence}" if evidence else ctx_text
print(full if full.strip() else "(empty)")

# Cleanup
graph.clear_all()
