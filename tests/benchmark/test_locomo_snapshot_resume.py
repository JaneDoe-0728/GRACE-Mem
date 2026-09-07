"""Regression tests for LoCoMo session snapshot and resume guarantees."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from experiment.locomo.artifacts import snapshot
from experiment.locomo.pipeline.worker import _ingest_remaining_sessions


class _FakeGraph:
    cfg = SimpleNamespace(entity_label="Entity", rel_type="KG_REL")

    def _run_read(self, query, _params):
        if "RETURN e.id" in query:
            return [
                {
                    "id": "entity-1",
                    "name": "Alice",
                    "type": "Person",
                    "description": "A test entity",
                }
            ]
        return []


def _write_valid_vdb_artifacts(base_dir):
    for directory_name in (
        "entities_chroma",
        "relationships_chroma",
        "summaries_chroma",
    ):
        directory = base_dir / directory_name
        directory.mkdir(parents=True)
        (directory / "chroma.sqlite3").write_bytes(b"sqlite-test-data")

    for file_name in (
        "entities_cache.pkl",
        "relationships_cache.pkl",
        "entities_bm25.pkl",
    ):
        (base_dir / file_name).write_bytes(b"persisted-test-data")


@pytest.fixture
def saved_snapshot(tmp_path, monkeypatch):
    working_dir = tmp_path / "working-artifacts"
    working_dir.mkdir()
    _write_valid_vdb_artifacts(working_dir)
    monkeypatch.setattr(snapshot, "ARTIFACTS_SRC", working_dir)

    sample_dir = tmp_path / "sample_0"
    compatibility = {
        "dataset": "locomo",
        "sample_index": 0,
        "sample_id": "conv-26",
        "dataset_sha256": "dataset-hash",
        "session_source_sha256": "session-source-hash",
        "ingest_config": {"chunk_turns": 8, "prev_k": 2},
    }
    path = snapshot.save_snapshot(
        sample_dir,
        1,
        _FakeGraph(),
        compatibility=compatibility,
    )
    return sample_dir, compatibility, path


def test_interrupted_ingest_resumes_after_last_completed_session():
    attempts = []
    flushed = []
    saved = []

    class FakeIngest:
        fail_on_session = 3

        @classmethod
        def sessions_to_one_turn_df(cls, records, **_kwargs):
            return SimpleNamespace(empty=False, session_id=records[0]["session_id"])

        @classmethod
        def ingest_by_session_one_turn(cls, _ingestor, frame, **_kwargs):
            attempts.append(frame.session_id)
            if frame.session_id == cls.fail_on_session:
                raise KeyboardInterrupt("simulated interruption")
            return {str(frame.session_id): []}

    records = [
        {"sample_index": 0, "session_id": session_id}
        for session_id in (1, 2, 3, 4)
    ]
    arguments = {
        "records": records,
        "ingest_module": FakeIngest,
        "ingestor": object(),
        "prev_k": 2,
        "entity_sim_topk": 3,
        "entity_sim_threshold": 0.6,
        "chunk_turns": 8,
        "flush_persist": lambda: flushed.append(True),
        "save_after_session": saved.append,
    }

    with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
        _ingest_remaining_sessions(resume_from=0, **arguments)

    assert attempts == [1, 2, 3]
    assert saved == [1, 2]
    assert len(flushed) == 2

    FakeIngest.fail_on_session = None
    _ingest_remaining_sessions(resume_from=max(saved), **arguments)

    assert attempts == [1, 2, 3, 3, 4]
    assert saved == [1, 2, 3, 4]
    assert len(flushed) == 4


def test_resume_rejects_incompatible_ingest_settings(saved_snapshot):
    sample_dir, compatibility, _path = saved_snapshot
    incompatible = {
        **compatibility,
        "ingest_config": {"chunk_turns": 4, "prev_k": 2},
    }

    with pytest.raises(snapshot.SnapshotCompatibilityError, match="ingest_config"):
        snapshot.highest_existing_snapshot(
            sample_dir,
            [1, 2],
            expected_compatibility=incompatible,
        )


def test_resume_rejects_snapshot_payload_corruption(saved_snapshot):
    sample_dir, compatibility, path = saved_snapshot
    (path / "entities_cache.pkl").write_bytes(b"tampered")

    with pytest.raises(snapshot.SnapshotCorruptionError, match="manifest mismatch"):
        snapshot.validate_snapshot(
            sample_dir,
            1,
            expected_compatibility=compatibility,
        )


def test_resume_rejects_corrupt_snapshot_metadata(saved_snapshot):
    sample_dir, compatibility, path = saved_snapshot
    (path / "snapshot_meta.json").write_text("{truncated", encoding="utf-8")

    with pytest.raises(snapshot.SnapshotCorruptionError, match="metadata is unreadable"):
        snapshot.validate_snapshot(
            sample_dir,
            1,
            expected_compatibility=compatibility,
        )


def test_resume_rejects_non_contiguous_snapshot_directories(saved_snapshot):
    sample_dir, compatibility, path = saved_snapshot
    path.rename(snapshot.snapshot_dir(sample_dir, 2))

    with pytest.raises(snapshot.SnapshotCorruptionError, match="Non-contiguous"):
        snapshot.highest_existing_snapshot(
            sample_dir,
            [1, 2],
            expected_compatibility=compatibility,
        )
