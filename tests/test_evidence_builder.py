"""Characterization tests for the evidence block builder.

build_evidence_block is 597 of evidence.py's 900 lines and sits at 3% coverage.
These tests record what it produces so it can be split without changing it, the
same way tests/test_retrieval_pipeline.py did for assemble_context_from_query.

They assert almost nothing about what the values *should* be. A characterization
test records what the code does today; judgements about whether that is right
belong elsewhere.

Each mode gets its own snapshot, because the modes are the branches a reviewer
cannot check by reading a large diff.

Regenerate after an intentional behaviour change, never to make a red test
green:

    KG_UPDATE_EVIDENCE_SNAPSHOTS=1 uv run pytest tests/test_evidence_builder.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from grace_mem.retrieval.evidence import EvidenceBuilder
from tests.evidence_fakes import (
    CallLog,
    FakeRawTurnLookup,
    FakeSummariesVDB,
    _vec,
    cache,
    context_entities,
    context_relationships,
)

SNAPSHOT_DIR = Path(__file__).parent / "fixtures"
QUERY = "What did they say about the marathon?"

#: The evidence paths that differ in what text they return and how they select it.
MODES = {
    "summary": {},
    "raw_context": {"use_raw_context": True},
    "split_embeddings": {"use_split_embeddings": True},
    "split_single_entry_raw": {"use_split_embeddings": True, "split_single_entry_raw": True},
    "direct_vector": {"use_split_embeddings": True, "summary_direct_vector_topn": 4},
    "per_entity_quota": {"summary_per_entity_min": 2},
    "hyde_blend": {"hyde_vec": _vec(1), "hyde_weight": 0.3, "hyde_mode": "blend"},
    "rerank_cosine_only": {
        "use_split_embeddings": True,
        "summary_direct_vector_topn": 4,
        "summary_rerank_topk": 3,
        "summary_rerank_cosine_only": True,
    },
}


def _builder() -> EvidenceBuilder:
    log = CallLog()
    b = EvidenceBuilder(
        summaries_vdb=FakeSummariesVDB(log),
        vector_db_manager=None,
        cache=cache(),
        raw_context_lookup=FakeRawTurnLookup(log),
    )
    b.log = log
    return b


def _capture(**overrides) -> dict:
    b = _builder()
    text = b.build_evidence_block(
        context_entities=context_entities(),
        context_relationships=context_relationships(),
        summary_topk_global=3,
        query_vec=_vec(0),
        summary_vec_threshold=0.4,
        query_text=QUERY,
        request_id="snapshot",
        **overrides,
    )
    return {"evidence_text": text, "calls": b.log.entries}


@pytest.mark.parametrize("mode", sorted(MODES))
def test_evidence_block_matches_snapshot(mode: str) -> None:
    """The rendered evidence and the store conversation are unchanged for this mode."""
    path = SNAPSHOT_DIR / f"evidence_{mode}.json"
    actual = _capture(**MODES[mode])

    if os.getenv("KG_UPDATE_EVIDENCE_SNAPSHOTS") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"snapshot rewritten: {path.name}")

    assert path.exists(), (
        f"no snapshot for mode={mode}. Generate with KG_UPDATE_EVIDENCE_SNAPSHOTS=1."
    )
    assert actual == json.loads(path.read_text(encoding="utf-8"))


def test_every_mode_actually_produces_evidence() -> None:
    """A snapshot of an empty block would pass and prove nothing."""
    for mode, overrides in MODES.items():
        captured = _capture(**overrides)
        assert captured["evidence_text"].strip(), f"{mode} produced no evidence text"
        assert captured["calls"], f"{mode} never touched the summaries store"


def test_the_modes_do_not_all_agree() -> None:
    """If every mode rendered the same block, the snapshots could not tell a
    broken mode from a working one."""
    rendered = {m: _capture(**o)["evidence_text"] for m, o in MODES.items()}
    assert len(set(rendered.values())) > 1, "all modes produced identical evidence"
