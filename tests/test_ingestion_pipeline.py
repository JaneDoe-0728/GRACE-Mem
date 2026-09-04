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
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from grace_mem.data_model.extraction import ExtractionResult
from grace_mem.ingestion.pipeline import IngestionFailedError, Ingestor, IngestorConfig
from grace_mem.ingestion.steps.sync import ExtractionSyncer
from grace_mem.services.vector_store.chroma_manager import VDBManager
from grace_mem.utils.paths import resolve_project_root
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
        "grace_mem.services.vector_store.chroma_manager.CacheStore.save",
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


def test_failed_primary_and_fallback_ingest_raise_instead_of_returning_success() -> None:
    """A failed extraction must stop callers before they write a checkpoint."""
    ingestor = _ingestor()
    ingestor._entity_extractor = Mock()
    ingestor._entity_extractor.extract.return_value = (False, "model returned invalid JSON")

    with pytest.raises(IngestionFailedError, match="fallback ingest failed"):
        ingestor.summarize_and_ingest_turn(
            session_id=SESSION_ID,
            message_id=MESSAGE_ID,
            user_text=USER_TEXT,
            assistant_text=ASSISTANT_TEXT,
            dialogue_datetime=DIALOGUE_DATETIME,
        )


def test_longmem_ingest_failure_preserves_checkpoint_for_retry() -> None:
    """The failed session must not join the resume set or permit QA to start."""
    from experiment.longmem.pipeline.runner import DatasetRunner

    runner = object.__new__(DatasetRunner)
    runner.current_ingestor = object()
    runner.current_mgr = Mock()
    runner.ingest_stage = Mock()
    runner.ingest_stage.normalize_sessions.side_effect = lambda data: data
    runner.ingest_stage.ingest_by_turn_pairs.side_effect = RuntimeError("ingest failed")
    runner._load_checkpoint = Mock(
        return_value={"processed_session_ids": ["session-1"]}
    )
    runner._save_checkpoint = Mock()
    runner._record_session_failure = Mock()
    config = SimpleNamespace(
        name="dataset-a",
        resume=True,
        prev_k=2,
        entity_sim_topk=3,
        entity_sim_threshold=0.6,
        checkpoint_every_n_sessions=5,
    )
    frame = pd.DataFrame(
        [
            {"session_id": "session-1", "content": "already complete"},
            {"session_id": "session-2", "content": "must retry"},
        ]
    )

    with pytest.raises(RuntimeError, match="checkpoint preserved for retry"):
        runner._ingest_by_turn_pairs(frame, config)

    runner.current_mgr.flush_persist.assert_called_once_with()
    assert runner._save_checkpoint.call_args.args[1] == {"session-1"}
    assert runner._save_checkpoint.call_args.kwargs["stage"] == "ingest_in_progress"


def test_graph_verification_uses_stable_entity_ids_not_manager_lookup_keys() -> None:
    """FalkorDB stores metadata ids, not EntityManager's normalized dict keys."""
    graph = Mock()
    graph.sync_entities.return_value = 1
    graph.sync_relationships.return_value = 1
    graph.check_entity_ids.side_effect = lambda ids: ids
    graph.check_relationship_ids.side_effect = lambda ids: ids

    entity_service = Mock()
    entity_service.find_similar_for_hybrid.return_value = {}
    entity_service.apply_ops.return_value = (
        {"user::person": {"id": "person_user", "name": "User"}},
        {},
        {"added": 1, "updated": 0},
    )
    relationship_service = Mock()
    relationship_service.upsert_from_extraction.return_value = [{"id": "rel_user_event"}]
    llm = Mock()
    llm.generate_llm_extract.return_value = {"results": []}
    syncer = ExtractionSyncer(
        llm=llm,
        graph=graph,
        entity_service=entity_service,
        relationship_service=relationship_service,
        cfg=SimpleNamespace(similar_entity_top_k=3, entity_sim_threshold=0.6),
    )

    result = syncer.sync(
        ExtractionResult(entities=[], relationships=[]),
        provenance=None,
        request_id="verification-test",
    )

    graph.check_entity_ids.assert_called_once_with(["person_user"])
    assert result["graph_sync_ok"] is True
    assert result["graph_sync_missing_entity_count"] == 0


def test_the_token_log_is_written_at_the_repository_root() -> None:
    """The one path anchor the resolve_project_root sweep missed.

    `parents[2]` was the repo root while this lived at KG/llm/client.py and became
    the grace_mem package directory in the move, so every run's aggregate token
    accounting silently relocated to grace_mem/logs/token_usage.jsonl -- and
    importing the adapter created a stray logs/ dir inside the package.
    """
    from grace_mem.services.llm.token_tracking import _TOKEN_LOG_PATH

    root = resolve_project_root()

    assert _TOKEN_LOG_PATH == root / "logs" / "token_usage.jsonl"
    assert _TOKEN_LOG_PATH.parent.parent == root
    assert "grace_mem" not in _TOKEN_LOG_PATH.parts


def test_a_transient_relationship_failure_is_retried_rather_than_failing_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Since 04cba26 a relationship failure costs the caller its whole turn.

    _require_successful_ingest turns it into an IngestionFailedError, which ends a
    LongMem dataset or a LoCoMo sample, so three back-to-back attempts inside the
    same few milliseconds -- all landing on the same rate limit or the same dropped
    connection -- were not buying what they looked like they were buying.
    """
    from grace_mem.ingestion.extractors import relationship_extractor as module

    monkeypatch.setattr(module, "_RETRY_BACKOFF_SEC", 0.0)
    calls = {"n": 0}

    class FlakyLLM:
        def generate_llm_extract(self, prompt: str):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise ConnectionError("connection reset by peer")
            return ("", 0.1)

    extractor = module.RelationshipExtractor(
        llm=FlakyLLM(),
        lock=threading.Lock(),
        cfg=SimpleNamespace(
            llm_tuple_delim="<|>", llm_record_delim="##", llm_completion_delim="<|COMPLETE|>"
        ),
    )

    success, result = extractor.extract({"tuple_delimiter": "<|>"}, "{entities_text}", [], "RID")

    assert success, f"gave up after {calls['n']} attempts: {result}"
    assert calls["n"] == 4


def test_an_over_long_prompt_fails_without_spending_the_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retry budget is for the transport. A context-length error is
    deterministic -- the same prompt is resent unchanged -- so waiting on it only
    spends the backoff to reach the same answer."""
    from grace_mem.ingestion.extractors import relationship_extractor as module

    monkeypatch.setattr(module, "_RETRY_BACKOFF_SEC", 30.0)
    calls = {"n": 0}

    class OverContextLLM:
        def generate_llm_extract(self, prompt: str):
            calls["n"] += 1
            raise ValueError("This model's maximum context length is 8192 tokens")

    extractor = module.RelationshipExtractor(
        llm=OverContextLLM(),
        lock=threading.Lock(),
        cfg=SimpleNamespace(
            llm_tuple_delim="<|>", llm_record_delim="##", llm_completion_delim="<|COMPLETE|>"
        ),
    )

    success, result = extractor.extract({"tuple_delimiter": "<|>"}, "{entities_text}", [], "RID")

    assert not success
    assert calls["n"] == 1
    assert "maximum context length" in str(result)
