"""Shared doubles for the evidence characterization tests.

`EvidenceBuilder.build_evidence_block` is 597 lines at 3% coverage -- the worst
single method left in grace_mem, worse in proportion than
assemble_context_from_query was before it was split. These doubles exist so it
can be taken apart the same way.

The boundary is eight calls across two collaborators: the summaries vector store
and the raw-turn lookup, plus a cache dict. Nothing here needs Chroma, FalkorDB,
an LLM or the reranker model.

Reuses the CallLog from retrieval_fakes for the same reason it exists there: a
later stage can mask an earlier one, so the snapshot has to record the
conversation, not only the answer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tests.retrieval_fakes import CallLog, _stable  # noqa: F401  (re-exported)

# --------------------------------------------------------------------------- #
# The fixture corpus                                                           #
# --------------------------------------------------------------------------- #
# Six summaries with descending similarity. The top-k defaults cut at 3, and
# summary_vec_threshold at 0.4 cuts at s5, so both bind -- a fixture where only
# one bound could not tell a broken top-k from a working one.
SUMMARY_IDS = ["s1", "s2", "s3", "s4", "s5", "s6"]
SUMMARY_SCORES = {"s1": 0.93, "s2": 0.85, "s3": 0.71, "s4": 0.58, "s5": 0.36, "s6": 0.12}

SUMMARY_TEXT = {s: f"compressed summary for {s}" for s in SUMMARY_IDS}
RAW_TURN_TEXT = {s: f"raw turn text for {s}, longer than the summary" for s in SUMMARY_IDS}

ENTITY_IDS = ["e1", "e2", "e3"]
RELATIONSHIP_IDS = ["r1", "r2"]


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def entity_meta(entity_id: str, summary_ids: list[str]) -> dict[str, Any]:
    """An entity whose provenance points at the summaries it came from."""
    return {
        "id": entity_id,
        "name": f"Entity {entity_id.upper()}",
        "entity_type": "Concept",
        "description": f"description of {entity_id}",
        "prov": {
            "events": [
                {"session_id": "s1", "message_id": int(s[1:]), "summary_id": s}
                for s in summary_ids
            ]
        },
    }


def relationship_meta(rel_id: str, src: str, tgt: str, summary_ids: list[str]) -> dict[str, Any]:
    return {
        "id": rel_id,
        "rel_id": rel_id,
        "source_id": src,
        "target_id": tgt,
        "rel_desc": f"{src} relates to {tgt}",
        "rel_keywords": "alpha|beta",
        "prov": {
            "events": [
                {"session_id": "s1", "message_id": int(s[1:]), "summary_id": s}
                for s in summary_ids
            ]
        },
    }


def context_entities() -> list[dict[str, Any]]:
    return [
        entity_meta("e1", ["s1", "s2"]),
        entity_meta("e2", ["s3"]),
        entity_meta("e3", ["s4", "s5"]),
    ]


def context_relationships() -> list[dict[str, Any]]:
    return [
        relationship_meta("r1", "e1", "e2", ["s1", "s3"]),
        relationship_meta("r2", "e2", "e3", ["s6"]),
    ]


def cache() -> dict[str, dict]:
    return {
        "entities": {f"entity {e['id']}": e for e in context_entities()},
        "relationships": {f"rel {r['rel_id']}": r for r in context_relationships()},
    }


class _Logged:
    def __init__(self, log: CallLog) -> None:
        self.log = log


class FakeSummariesVDB(_Logged):
    """The summaries vector store, with the real SummariesVDB's signatures.

    Entry ids carry a :u / :a suffix in split mode and are bare otherwise; both
    are answered here so the split and single-entry paths run from one fixture.
    """

    #: Read directly by evidence.py:549 to recover a summary's dialogue_datetime.
    #: A private attribute of the store, reached through the builder -- recorded
    #: here because faking it is the only way to drive that path, and because a
    #: double that has to reproduce a collaborator's internals is telling you
    #: something about the coupling.
    _meta = [
        {"summary_id": s, "dialogue_datetime": f"2023/02/{10 + int(s[1:])} (Sat) 08:00"}
        for s in SUMMARY_IDS
    ]

    def search(self, query_vec, top_k: int = 5, threshold: float | None = None):
        self.log.record("vdb.search", top_k=top_k, threshold=threshold)
        hits = [({"id": s, "summary_id": s, "text": SUMMARY_TEXT[s],
                  "raw_text": RAW_TURN_TEXT[s]}, SUMMARY_SCORES[s])
                for s in SUMMARY_IDS]
        if threshold is not None:
            hits = [h for h in hits if h[1] >= threshold]
        return hits[:top_k]

    def get_summaries_by_ids(self, summary_ids, max_len: int = 3000, top_n: int = 10):
        self.log.record("vdb.get_summaries_by_ids", summary_ids=summary_ids, top_n=top_n)
        return [SUMMARY_TEXT[s] for s in summary_ids if s in SUMMARY_TEXT][:top_n]

    def get_summary_text_by_id(self, summary_id):
        self.log.record("vdb.get_summary_text_by_id", summary_id=summary_id)
        return SUMMARY_TEXT.get(str(summary_id).split(":")[0])

    def get_raw_turn_text_by_id(self, summary_id):
        self.log.record("vdb.get_raw_turn_text_by_id", summary_id=summary_id)
        return RAW_TURN_TEXT.get(str(summary_id).split(":")[0])

    def get_text_by_entry_id(self, entry_id):
        self.log.record("vdb.get_text_by_entry_id", entry_id=entry_id)
        base = str(entry_id).split(":")[0]
        if str(entry_id).endswith(":u"):
            return RAW_TURN_TEXT.get(base)
        return SUMMARY_TEXT.get(base)

    def compare_by_id_raw(self, mid, query_vec, request_id=None, debug_context=None):
        self.log.record("vdb.compare_by_id_raw", mid=mid)
        return SUMMARY_SCORES.get(str(mid).split(":")[0])


class FakeRawTurnLookup(_Logged):
    """The CSV-backed lookup that reconstructs pre-compression turn text."""

    def get(self, session_id, message_id):
        self.log.record("raw_turn_lookup.get", session_id=session_id, message_id=message_id)
        return RAW_TURN_TEXT.get(f"s{message_id}", "")
