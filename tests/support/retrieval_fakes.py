"""Shared doubles for the retrieval characterization tests.

`Retriever.assemble_context_from_query` reaches outside itself in exactly
eleven places, across five components and a cache dict. All of them are faked
here, so the tests pin the method's own behaviour -- stage ordering, the
filter_method dispatch, and how candidates become context text -- without
FalkorDB, an LLM endpoint, or the downloaded models.

Same intent as `agent_filter_fakes.py`, one layer down.

**The doubles record every call.** Snapshotting the returned value alone is
not enough for orchestration code, because a later stage can mask an earlier
one: the reranker recovery unions its result with the filter's, so changing
what the filter is asked for leaves the final entity list identical. Pinning
the conversation with the collaborators catches that. Pinning only the answer
does not -- verified by injecting exactly that change and watching an
output-only snapshot pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# The fixture graph                                                            #
# --------------------------------------------------------------------------- #
ENTITY_IDS = ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"]
RELATIONSHIP_IDS = ["r1", "r2", "r3", "r4", "r5", "r6"]

# Six of the eight clear the 0.5 default threshold while filter_ent_topk is 3,
# so both cuts bind. A fixture where only one binds cannot tell a broken top-k
# from a working one.
ENTITY_SCORES = {"e1": 0.95, "e2": 0.88, "e3": 0.81, "e4": 0.72,
                 "e5": 0.63, "e6": 0.54, "e7": 0.31, "e8": 0.14}
RELATIONSHIP_SCORES = {"r1": 0.92, "r2": 0.84, "r3": 0.76,
                       "r4": 0.58, "r5": 0.29, "r6": 0.11}

# The reranker is a cross-encoder returning logits, and reranker_threshold
# defaults to -3.0. Scoring it on the 0..1 similarity scale would put every
# candidate above the threshold and make the reranker paths indistinguishable.
RERANKER_SCORES = {"e1": 4.10, "e2": 2.30, "e3": 0.85, "e4": -1.20,
                   "e5": -2.40, "e6": -4.60, "e7": -5.80, "e8": -6.30}
RELATIONSHIP_RERANKER_SCORES = {"r1": 3.40, "r2": 1.10, "r3": -0.70,
                                "r4": -2.90, "r5": -3.90, "r6": -5.10}

# r6 dangles on purpose: its target is an entity no search returns, which is
# the case the intersection step exists to drop.
EDGES = [
    ("e1", "r1", "e2"),
    ("e2", "r2", "e3"),
    ("e3", "r3", "e4"),
    ("e4", "r4", "e5"),
    ("e5", "r5", "e6"),
    ("e1", "r6", "e9-missing"),
]


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    """A fixed unit vector. Deterministic across runs and machines."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def _stable(value: Any) -> Any:
    """Reduce an argument to a form that compares and prints deterministically."""
    if isinstance(value, np.ndarray):
        return {"ndarray_shape": list(value.shape)}
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _stable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        items = [_stable(v) for v in value]
        # A list of plain strings is sorted before recording. The production
        # code hands sets to its collaborators as `list(some_set)` -- the seed
        # list for spreading activation is one -- so the order is whatever this
        # process's string hashing produced and carries no meaning. Recording it
        # raw makes the snapshot differ between runs on the same code.
        #
        # The cost is real: this snapshot cannot see a change that only reorders
        # a list of ids. The final entity_ids and relationship_ids in the
        # captured output are recorded unsorted and do catch that.
        if items and all(_is_orderless(i) for i in items):
            return sorted(items, key=_sort_key)
        return items
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return type(value).__name__


def _is_orderless(item: Any) -> bool:
    """True for the element shapes the production code derives from sets.

    Ids are strings; induced graph edges are (src, tgt, rel) triples built by a
    set comprehension. Both reach a collaborator in whatever order this
    process's hashing produced.
    """
    if isinstance(item, str):
        return True
    return isinstance(item, list) and all(isinstance(x, str) for x in item)


def _sort_key(item: Any) -> tuple:
    return (0, item, ()) if isinstance(item, str) else (1, "", tuple(item))


class CallLog:
    """Every call the code under test makes into its collaborators, in order."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def record(self, name: str, **kwargs: Any) -> None:
        self.entries.append({"call": name, "args": {k: _stable(v) for k, v in sorted(kwargs.items())}})

    @property
    def names(self) -> list[str]:
        return [e["call"] for e in self.entries]


def entity_meta(entity_id: str) -> dict[str, Any]:
    return {
        "id": entity_id,
        "name": f"Entity {entity_id.upper()}",
        "entity_type": "Concept",
        "description": f"description of {entity_id}",
        "prov": {"events": [{"session_id": "s1", "message_id": int(entity_id[1:])}]},
    }


def relationship_meta(rel_id: str) -> dict[str, Any]:
    src, _, tgt = next(e for e in EDGES if e[1] == rel_id)
    return {
        "id": rel_id,
        "rel_id": rel_id,
        "source_id": src,
        "target_id": tgt,
        "rel_desc": f"{src} relates to {tgt}",
        "rel_keywords": "alpha|beta",
        "rel_strength": 0.5,
        "prov": {"events": [{"session_id": "s1", "message_id": int(rel_id[1:])}]},
    }


def cache() -> dict[str, dict]:
    """A cache in the shape `build_id_to_meta_maps` inverts: keyed by name."""
    return {
        "entities": {f"entity {e}": entity_meta(e) for e in ENTITY_IDS},
        "relationships": {f"rel {r}": relationship_meta(r) for r in RELATIONSHIP_IDS},
    }


class _Logged:
    def __init__(self, log: CallLog) -> None:
        self.log = log


class FakeSearcher(_Logged):
    """Vector + BM25 entity search and relationship vector search.

    Returns the fixture's scores unchanged; applying the threshold and top-k is
    the code under test's job, not the double's.
    """

    def embed_query(self, question, request_id=None):
        self.log.record("searcher.embed_query", question=question)
        return _vec(0)

    def search_entities_hybrid(self, *, query_vec, low_level_keywords, entity_vec_threshold,
                               entity_top_k, request_id=None):
        self.log.record("searcher.search_entities_hybrid",
                        low_level_keywords=low_level_keywords,
                        entity_vec_threshold=entity_vec_threshold, entity_top_k=entity_top_k)
        hits = [(entity_meta(e), s) for e, s in ENTITY_SCORES.items() if e != "e8"]
        return {"vector": hits, "bm25": hits[:2]}

    def search_relationships_by_vec(self, keywords, relationship_top_k,
                                    relationship_vec_threshold, request_id=None):
        self.log.record("searcher.search_relationships_by_vec", keywords=keywords,
                        relationship_top_k=relationship_top_k,
                        relationship_vec_threshold=relationship_vec_threshold)
        return {"vector": [(relationship_meta(r), s) for r, s in RELATIONSHIP_SCORES.items()]}


class FakeGraph(_Logged):
    """Node and edge subgraph expansion over the fixture edges."""

    def get_node_subgraph(self, entity_ids):
        self.log.record("graph.get_node_subgraph", entity_ids=entity_ids)
        out: dict[str, dict] = {}
        for src, _rel, tgt in EDGES:
            if src in entity_ids or tgt in entity_ids:
                out.setdefault(src, entity_meta(src))
                if tgt in ENTITY_IDS:
                    out.setdefault(tgt, entity_meta(tgt))
        return out

    def get_edge_subgraph(self, rel_ids):
        self.log.record("graph.get_edge_subgraph", rel_ids=rel_ids)
        return [relationship_meta(r) for r in rel_ids if r in RELATIONSHIP_IDS]


class FakeEvidenceFilter(_Logged):
    """Subgraph intersection and cross-encoder reranking.

    Both methods apply the threshold and top-k they were handed, so a stage that
    forgets to pass one shows up as a changed result as well as a changed call.
    """

    @staticmethod
    def _cut(ids, scores, threshold, top_k):
        kept = sorted((i for i in ids if scores.get(i, 0.0) >= threshold),
                      key=lambda i: -scores.get(i, 0.0))
        return kept[:top_k] if top_k else kept

    def compute_subgraph_intersection(self, node_subgraph, edge_subgraph, use_union=True, request_id=None):
        self.log.record("filter.compute_subgraph_intersection",
                        node_ids=sorted(node_subgraph),
                        edge_ids=[e.get("rel_id") for e in edge_subgraph], use_union=use_union)
        return set(node_subgraph), {e.get("rel_id") for e in edge_subgraph if e.get("rel_id")}

    def rerank_filter(self, question, entity_ids, relationship_ids,
                      entity_top_k, relationship_top_k, threshold, request_id=None):
        self.log.record("filter.rerank_filter", question=question, entity_ids=entity_ids,
                        relationship_ids=relationship_ids, entity_top_k=entity_top_k,
                        relationship_top_k=relationship_top_k, threshold=threshold)
        return (self._cut(entity_ids, RERANKER_SCORES, threshold, entity_top_k),
                self._cut(relationship_ids, RELATIONSHIP_RERANKER_SCORES, threshold,
                          relationship_top_k))


class FakeSpreadingActivation(_Logged):
    def run(self, seed_entity_ids, query_vec, request_id=None):
        self.log.record("sa.run", seed_entity_ids=seed_entity_ids)
        # e8 is only reachable this way; that is what makes the SA path visible.
        return {"e8": 0.42}
