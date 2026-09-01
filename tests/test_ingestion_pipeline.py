"""Characterization tests for the ingestion pipeline.

summarize_and_ingest_turn is 260 lines, and _repair_temporal_entities beside it
is 316 more, in a file that delegates five times in total -- the same shape
retriever.py had, one size smaller. These tests record what one turn produces so
that can be taken apart without changing it.

They assert almost nothing about what the values *should* be. A characterization
test records what the code does today.

The Ingestor is built through object.__new__ and its collaborators set directly,
because __init__ constructs them from an LLM client, a graph and a vector
manager. Pinning summarize_and_ingest_turn means faking what it talks to, not
what builds it.

Regenerate after an intentional behaviour change, never to make a red test
green:

    KG_UPDATE_INGESTION_SNAPSHOTS=1 uv run pytest tests/test_ingestion_pipeline.py
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from grace_mem.adapters.vector_store.chroma_manager import VDBManager
from grace_mem.ingestion.pipeline import Ingestor, IngestorConfig
from grace_mem.runtime.paths import resolve_project_root
from tests.ingestion_fakes import (
    ASSISTANT_TEXT,
    DIALOGUE_DATETIME,
    MESSAGE_ID,
    SESSION_ID,
    USER_TEXT,
    CallLog,
    FakeCompressor,
    FakeEntityExtractor,
    FakeRelationshipExtractor,
    FakeSyncer,
    FakeVectorDBManager,
)

SNAPSHOT_DIR = Path(__file__).parent / "fixtures"

#: The ingest paths that differ in how a turn becomes graph state.
MODES = {
    "turn_pairs": {},
    "no_dialogue_datetime": {"_no_datetime": True},
    "sim_topk_and_threshold": {"entity_sim_topk": 5, "entity_sim_threshold": 0.8},
    "prev_k_zero": {"prev_k": 0},
}


def _ingestor() -> Ingestor:
    log = CallLog()
    ing = object.__new__(Ingestor)
    ing.cfg = IngestorConfig()
    ing.llm = None
    ing.graph = None
    ing.vector_db_manager = FakeVectorDBManager(log)
    ing.summaries_vdb = ing.vector_db_manager.summaries_vdb
    ing.entity_service = None
    ing.relationship_service = None
    ing._lock = threading.Lock()
    ing._compressor = FakeCompressor(log)
    ing._entity_extractor = FakeEntityExtractor(log)
    ing._rel_extractor = FakeRelationshipExtractor(log)
    ing._syncer = FakeSyncer(log)
    ing.log = log
    return ing


def _capture(**overrides) -> dict:
    no_datetime = overrides.pop("_no_datetime", False)
    ing = _ingestor()
    result = ing.summarize_and_ingest_turn(
        session_id=SESSION_ID,
        message_id=MESSAGE_ID,
        user_text=USER_TEXT,
        assistant_text=ASSISTANT_TEXT,
        dialogue_datetime=None if no_datetime else DIALOGUE_DATETIME,
        **overrides,
    )
    return {
        "result_keys": sorted(result) if isinstance(result, dict) else type(result).__name__,
        "result": _scrub(result),
        "calls": ing.log.entries,
    }


def _scrub(value):
    """Drop the per-run request_id before snapshotting.

    The Ingestor mints a fresh uuid4 per turn, so it differs on every run and
    would make the snapshot compare unequal to itself. Nothing downstream keys
    on its value -- it exists to correlate log lines within one turn -- so
    dropping it loses nothing the snapshot was pinning.
    """
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items() if k != "request_id"}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@pytest.mark.parametrize("mode", sorted(MODES))
def test_ingest_turn_matches_snapshot(mode: str) -> None:
    """What one turn produces, and the conversation it has, are unchanged."""
    path = SNAPSHOT_DIR / f"ingestion_{mode}.json"
    actual = _capture(**MODES[mode])

    if os.getenv("KG_UPDATE_INGESTION_SNAPSHOTS") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, indent=2, sort_keys=True, default=str) + "\n",
                        encoding="utf-8")
        pytest.skip(f"snapshot rewritten: {path.name}")

    assert path.exists(), (
        f"no snapshot for mode={mode}. Generate with KG_UPDATE_INGESTION_SNAPSHOTS=1."
    )
    assert json.loads(json.dumps(actual, sort_keys=True, default=str)) == json.loads(
        path.read_text(encoding="utf-8")
    )


def test_every_mode_drives_the_whole_pipeline() -> None:
    """A snapshot of a run that stopped after compression would prove nothing."""
    for mode, overrides in MODES.items():
        names = [c["call"] for c in _capture(**overrides)["calls"]]
        for required in ("compressor.summarize_turn", "entity_extractor.extract",
                         "relationship_extractor.extract", "syncer.sync"):
            assert required in names, f"{mode} never called {required}: {names}"


def test_the_modes_do_not_all_agree() -> None:
    """If every mode had the same conversation, the snapshots could not tell a
    broken parameter from a working one."""
    convos = {m: json.dumps(_capture(**o)["calls"], sort_keys=True, default=str)
              for m, o in MODES.items()}
    assert len(set(convos.values())) > 1, "all modes produced the same conversation"


def test_repository_paths_resolve_above_grace_mem_package() -> None:
    """Adapters must find the root .env and downloaded models from any cwd."""
    project_root = resolve_project_root()

    assert project_root == Path(__file__).resolve().parent.parent
    assert (project_root / "grace_mem").is_dir()
    assert (project_root / ".env.example").is_file()


def test_persist_requests_never_write_the_same_store_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back-to-back entity/relationship persists must serialize disk writes."""
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_request_entered = threading.Event()
    second_request_completed = threading.Event()

    class BlockingStore:
        def __init__(self) -> None:
            self.calls = 0
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def save(self) -> None:
            with self.lock:
                self.calls += 1
                call_number = self.calls
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                if call_number == 1:
                    first_started.set()
                    assert release_first.wait(timeout=2)
                else:
                    second_started.set()
            finally:
                with self.lock:
                    self.active -= 1

        def export_metadatas_jsonl(self, _path: str) -> None:
            pass

    monkeypatch.setattr(
        "grace_mem.adapters.vector_store.chroma_manager.CacheStore.save",
        lambda *_args, **_kwargs: None,
    )
    manager = VDBManager(tmp_path / "artifacts")
    store = BlockingStore()
    manager._entities_vdb = store

    manager.persist_async()
    assert first_started.wait(timeout=2)

    def request_second_persist() -> None:
        second_request_entered.set()
        manager.persist_async()
        second_request_completed.set()

    requester = threading.Thread(target=request_second_persist)
    requester.start()
    assert second_request_entered.wait(timeout=2)
    assert not second_started.wait(timeout=0.1)
    assert not second_request_completed.is_set()

    release_first.set()
    requester.join(timeout=2)
    assert not requester.is_alive()
    manager._wait_for_persist()

    assert second_started.is_set()
    assert store.max_active == 1
