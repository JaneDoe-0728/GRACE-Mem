# pipeline/factory.py
"""
Factory for building the KG pipeline (retriever + ingestor).

Importing this module is now side-effect-free. Call build_pipeline() to
open connections and construct the pipeline objects.

供上層 server.py 取用建構好的 retriever 與 ingestor 物件:
  from KG.pipeline.factory import build_pipeline
  _pipeline = build_pipeline()
  retriever = _pipeline["retriever"]
  ingestor  = _pipeline["ingestor"]
"""


def build_pipeline(*, retriever_config=None, ingestor_config=None) -> dict:
    """Open all connections and return a dict with retriever, ingestor, graph, mgr."""
    from KG.pipeline.retriever import Retriever
    from KG.pipeline.ingestor import Ingestor
    # from KG.pipeline.ingestor_no_ops import IngestorNoEntityOps
    from KG.storage import MGR
    from KG.llm import LLMClient
    from KG.graph.falkordb import graph_from_env
    from embeddings import embedder
    from KG.services import EntityManager, RelationshipManager, Provenance

    GLOBAL_CACHE = MGR.cache
    llm = LLMClient()
    graph = graph_from_env().open()

    ent = EntityManager(
        embedder=embedder,
        mgr=MGR,
        provenance=Provenance,
        GLOBAL_CACHE=GLOBAL_CACHE,
        processed_ent_map=GLOBAL_CACHE["entities"],
        processed_ent_full_map=GLOBAL_CACHE["entities_full"],
    )

    rel = RelationshipManager(
        embedder=embedder,
        mgr=MGR,
        provenance=Provenance,
        GLOBAL_CACHE=GLOBAL_CACHE,
        processed_rel_map=GLOBAL_CACHE["relationships"],
        processed_rel_full_map=GLOBAL_CACHE["relationships_full"],
    )

    retriever = Retriever(
        llm=llm,
        graph=graph,
        mgr=MGR,
        embed=embedder.embed,
        cache=GLOBAL_CACHE,
        config=retriever_config,
    )
    ingestor  = Ingestor(llm=llm, graph=graph, mgr=MGR, ent_svc=ent, rel_svc=rel, config=ingestor_config)
    # ingestor = IngestorNoEntityOps(llm=llm, graph=graph, mgr=MGR, ent_svc=ent, rel_svc=rel)

    return {"retriever": retriever, "ingestor": ingestor, "graph": graph, "mgr": MGR}
