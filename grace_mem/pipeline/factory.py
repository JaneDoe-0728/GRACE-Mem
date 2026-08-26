"""
Factory for building the KG pipeline (retriever + ingestor).

Importing this module is now side-effect-free. Call build_pipeline() to
open connections and construct the pipeline objects.

Hands the constructed retriever and ingestor objects to server.py above it:
  from grace_mem.pipeline.factory import build_pipeline
  with build_pipeline() as runtime:
      retriever = runtime.retriever
      ingestor = runtime.ingestor
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class PipelineRuntime(Mapping[str, Any]):
    """Constructed pipeline components with mapping compatibility and cleanup."""

    retriever: Any
    ingestor: Any
    graph: Any
    mgr: Any
    llm: Any | None = field(default=None, repr=False)
    owns_mgr: bool = field(default=True, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    _COMPONENT_NAMES: ClassVar[tuple[str, ...]] = (
        "retriever",
        "ingestor",
        "graph",
        "mgr",
    )

    def __getitem__(self, key: str) -> Any:
        if key not in self._COMPONENT_NAMES:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._COMPONENT_NAMES)

    def __len__(self) -> int:
        return len(self._COMPONENT_NAMES)

    def close(self) -> None:
        """Close runtime-owned external connections once."""
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None

        cleanups = []
        if self.owns_mgr and self.mgr is not None:
            cleanups.append(lambda: self.mgr.close(persist=True))
        if self.graph is not None:
            cleanups.append(self.graph.close)
        if self.llm is not None:
            cleanups.append(self.llm.close)

        for cleanup in cleanups:
            try:
                cleanup()
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        if first_error is not None:
            raise first_error

    def __enter__(self) -> "PipelineRuntime":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def build_pipeline(*, retriever_config=None, ingestor_config=None) -> PipelineRuntime:
    """Open connections and return the constructed pipeline runtime."""
    from grace_mem.pipeline.retriever import Retriever
    from grace_mem.pipeline.ingestor import Ingestor
    from grace_mem.storage import MGR
    from grace_mem.llm import LLMClient
    from grace_mem.graph.falkordb import graph_from_env
    from grace_mem.embeddings import embedder
    from grace_mem.services import EntityManager, RelationshipManager, Provenance

    global_cache = MGR.cache
    llm = LLMClient()
    graph = None
    try:
        graph = graph_from_env().open()
        ent = EntityManager(
            embedder=embedder,
            mgr=MGR,
            provenance=Provenance,
            global_cache=global_cache,
            processed_ent_map=global_cache["entities"],
            processed_ent_full_map=global_cache["entities_full"],
        )

        rel = RelationshipManager(
            embedder=embedder,
            mgr=MGR,
            provenance=Provenance,
            global_cache=global_cache,
            processed_rel_map=global_cache["relationships"],
            processed_rel_full_map=global_cache["relationships_full"],
        )

        retriever = Retriever(
            llm=llm,
            graph=graph,
            mgr=MGR,
            embed=embedder.embed,
            cache=global_cache,
            config=retriever_config,
        )
        ingestor = Ingestor(
            llm=llm,
            graph=graph,
            mgr=MGR,
            ent_svc=ent,
            rel_svc=rel,
            config=ingestor_config,
        )
        return PipelineRuntime(
            retriever=retriever,
            ingestor=ingestor,
            graph=graph,
            mgr=MGR,
            llm=llm,
            owns_mgr=True,
        )
    except BaseException:
        try:
            MGR.close(persist=True)
        except Exception:
            pass
        if graph is not None:
            try:
                graph.close()
            except Exception:
                pass
        try:
            llm.close()
        except Exception:
            pass
        raise
