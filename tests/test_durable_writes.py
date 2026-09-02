"""Writes that must survive being interrupted, and one that must not raise.

Both are about the same failure shape: the work is already done, and the thing
that records it is what breaks. A truncated pickle costs a re-extraction; a
summary write that raises ended a LongMem run that had already been judged
correct.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from experiment.longmem.utils.io import write_json_file
from grace_mem.domain.entities import Entity, EntityType
from grace_mem.runtime.atomic_write import atomic_write

# ── atomic_write ─────────────────────────────────────────────────────────────

def test_the_target_appears_only_once_the_write_finished(tmp_path: Path) -> None:
    target = tmp_path / "index.pkl"

    with atomic_write(target, "wb") as handle:
        pickle.dump({"docs": ["a", "b"]}, handle)
        # Mid-write: the target must not exist yet, half-written or otherwise.
        assert not target.exists()

    assert pickle.loads(target.read_bytes()) == {"docs": ["a", "b"]}


def test_a_failed_write_leaves_the_previous_copy_intact(tmp_path: Path) -> None:
    target = tmp_path / "index.pkl"
    target.write_bytes(pickle.dumps({"docs": ["old"]}))

    with pytest.raises(RuntimeError):
        with atomic_write(target, "wb") as handle:
            pickle.dump({"docs": ["new"]}, handle)
            raise RuntimeError("killed mid-write")

    assert pickle.loads(target.read_bytes()) == {"docs": ["old"]}
    assert list(tmp_path.iterdir()) == [target]  # no .tmp left behind


def test_the_parent_directory_is_created(tmp_path: Path) -> None:
    target = tmp_path / "artifacts" / "nested" / "meta.jsonl"

    with atomic_write(target, "w", encoding="utf-8") as handle:
        handle.write("{}\n")

    assert target.read_text() == "{}\n"


# ── write_json_file ──────────────────────────────────────────────────────────

def test_a_summary_holding_a_domain_object_still_writes(tmp_path: Path) -> None:
    # The LongMem run summary carried the raw ingest payloads, so json.dumps
    # met an Entity and raised -- after the run had been ingested, answered and
    # judged correct.
    entity = Entity(
        entity_name="asylum application",
        entity_type=EntityType.Event,
        entity_description="approved after over a year",
    )
    path = tmp_path / "processing_summary.json"

    write_json_file(path, [{"dataset": "001be529", "entity": entity}])

    written = json.loads(path.read_text())
    assert written[0]["dataset"] == "001be529"
    assert "asylum application" in written[0]["entity"]


def test_ordinary_values_are_untouched(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    payload = {"dataset": "a", "ingest_results": {"s1": 12}, "num_questions": 1}

    write_json_file(path, payload)

    assert json.loads(path.read_text()) == payload
