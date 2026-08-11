from __future__ import annotations

import numpy as np
import sys
import types

if "nltk" not in sys.modules:
    nltk_stub = types.ModuleType("nltk")
    nltk_stub.word_tokenize = lambda text: text.split()
    nltk_stub.pos_tag = lambda toks: [(tok, "NN") for tok in toks]
    sys.modules["nltk"] = nltk_stub

from KG.services.entity_manager import EntityManager


class _DummyEmbedder:
    def embed(self, texts):
        return np.zeros((len(texts), 2), dtype=np.float32)


class _DummyVDB:
    def __init__(self, batch_results):
        self._batch_results = batch_results

    def batch_search(self, vecs, top_k=5, threshold=0.6):
        return self._batch_results[: len(vecs)]


class _DummyBM25:
    metas = []
    size = 0

    def get_scores(self, _tokens):
        return [], []


class _DummyMgr:
    def __init__(self, batch_results):
        self._vdb = _DummyVDB(batch_results)
        self._bm25 = _DummyBM25()

    def get_entities_vdb(self, _dim):
        return self._vdb

    def get_entities_bm25(self, load_if_empty=True):
        return self._bm25


class _DummyProv:
    @staticmethod
    def merge_prov(old, new):
        return new or old


def _build_manager(*, cache_entities=None, batch_results=None):
    cache = {"entities": cache_entities or {}, "entities_full": {}, "relationships": {}, "relationships_full": {}}
    return EntityManager(
        embedder=_DummyEmbedder(),
        mgr=_DummyMgr(batch_results or []),
        provenance=_DummyProv(),
        GLOBAL_CACHE=cache,
        processed_ent_map=cache["entities"],
        processed_ent_full_map=cache["entities_full"],
    )


def test_temporal_entity_prefers_exact_name_match_only():
    exact_meta = {"id": "date_0705", "name": "2023-07-05", "type": "Date", "description": "The calendar date 2023-07-05."}
    mgr = _build_manager(
        cache_entities={"2023-07-05::date": exact_meta},
        batch_results=[
            [({"id": "date_0706", "name": "2023-07-06", "type": "Date", "description": "The calendar date 2023-07-06."}, 0.99)]
        ],
    )

    similar = mgr.find_similar_for_hybrid(
        [{"entity_name": "2023-07-05", "entity_type": "Date", "entity_description": "The calendar date 2023-07-05."}]
    )

    hits = similar[("2023-07-05", "Date")]
    assert len(hits) == 1
    assert hits[0][0]["name"] == "2023-07-05"
    assert hits[0][0]["_source"] == "exact_name"


def test_temporal_entity_with_no_exact_match_does_not_fall_back_to_semantic_candidates():
    mgr = _build_manager(
        batch_results=[
            [({"id": "time_1530", "name": "2023-07-06T15:30", "type": "Time", "description": "The clock time 2023-07-06T15:30."}, 0.99)]
        ],
    )

    similar = mgr.find_similar_for_hybrid(
        [{"entity_name": "2023-07-06T15:00", "entity_type": "Time", "entity_description": "The clock time 2023-07-06T15:00."}]
    )

    assert similar[("2023-07-06T15:00", "Time")] == []


def test_non_temporal_entities_still_use_hybrid_similarity_search():
    mgr = _build_manager(
        batch_results=[
            [({"id": "alice", "name": "Alice", "type": "Person", "description": "A person."}, 0.88)]
        ],
    )

    similar = mgr.find_similar_for_hybrid(
        [{"entity_name": "Alice", "entity_type": "Person", "entity_description": "A person."}]
    )

    hits = similar[("Alice", "Person")]
    assert len(hits) == 1
    assert hits[0][0]["name"] == "Alice"
    assert hits[0][0]["_source"] == "vector"
