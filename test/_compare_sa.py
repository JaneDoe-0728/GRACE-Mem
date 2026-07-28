import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from KG.graph.falkordb import graph_from_env
from KG.llm.client import LLMClient
from KG.pipeline.retriever import Retriever, RetrieverConfig
from KG.storage.chroma_manager import VDBManager
from embeddings import embedder

ARTIFACTS_DIR = REPO_ROOT / "experiment" / "multisample_runs" / "gpt-oss-20b" / "sample_0" / "artifacts"

TEST_CASES = [
    {
        "question": "What did Caroline research?",
        "gold": "Adoption agencies",
        "note": "baseline: answer is 1-hop from Caroline → SA 預期無差異",
    },
    {
        "question": "What does Melanie do with her family on hikes?",
        "gold": "Roast marshmallows, tell stories",
        "note": "2-hop: Melanie→Camping trip→campfire → SA 預期有差異",
    },
    {
        "question": "What book did Melanie read from Caroline's suggestion?",
        "gold": "Becoming Nicole",
        "note": "3-hop: Caroline→Melanie→Book read→Amy Ellis Nutt → SA 預期有差異",
    },
]

if not ARTIFACTS_DIR.exists():
    raise FileNotFoundError(f"Artifacts dir not found: {ARTIFACTS_DIR}")

mgr = VDBManager(ARTIFACTS_DIR)
mgr.initialize()
cache = mgr.cache

graph = graph_from_env().open()
graph.sync_entities(cache.get("entities", {}))
if cache.get("relationships"):
    graph.sync_relationships(list(cache["relationships"].values()))

llm = LLMClient()

for tc in TEST_CASES:
    QUESTION = tc["question"]
    print(f"\n{'#'*60}")
    print(f"Q : {QUESTION}")
    print(f"Gold: {tc['gold']}")
    print(f"Note: {tc['note']}")

    kw = Retriever(llm=llm, graph=graph, mgr=mgr, embed=embedder.embed, cache=cache).generate_query_keywords(QUESTION)
    print(f"High-level: {kw.high_level_keywords}")
    print(f"Low-level : {kw.low_level_keywords}")

    for use_sa in [False, True]:
        r = Retriever(
            llm=llm, graph=graph, mgr=mgr, embed=embedder.embed, cache=cache,
            config=RetrieverConfig(
                ent_topk=20, rel_topk=10, ent_threshold=0.2, rel_threshold=0.2,
                filter_ent_topk=10, filter_rel_topk=10,
                filter_ent_threshold=0.4, filter_rel_threshold=0.4,
                summary_topk_per_item=6, summary_vec_threshold=0.4,
                use_reranker=True,
                use_spreading_activation=use_sa,
                sa_max_hops=2, sa_rescale_c=0.4, sa_tau_a=0.5, sa_max_activated=20,
            ),
        )
        ents, rels, ctx, qv = r.assemble_context_from_query(
            question=QUESTION,
            low_level_keywords=kw.low_level_keywords,
            high_level_keywords=kw.high_level_keywords,
        )
        ev = r.evidence_builder.build_evidence_block(
            ents, rels, summary_topk_global=6, query_vec=qv,
            summary_vec_threshold=0.4, use_full_summary=True, fallback_to_raw=False,
        )

        ent_names = [e["name"] for e in ents]
        rel_pairs = [f"{x['source_name']}→{x['target_name']}" for x in rels]

        print(f"\n{'='*55} SA={use_sa}")
        print(f"Entities  ({len(ents)}): {ent_names}")
        print(f"Relations ({len(rels)}): {rel_pairs}")
        print(f"Evidence snippet:\n{(ev or '')[:400]}")

graph.clear_all()
