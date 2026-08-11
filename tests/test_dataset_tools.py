from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiment.longmem.tools.convert_dataset import (
    convert_longmem_dataset,
    iter_json_array,
    sha256_file,
)
from tools.download_datasets import LOCOMO, LONGMEM_FILES


def _record(
    question_id: str = "gpt4_1234abcd",
    question_type: str = "temporal-reasoning",
) -> dict:
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": "What happened first?",
        "answer": "The appointment.",
        "question_date": "2023/04/10 (Mon) 23:07",
        "haystack_session_ids": ["session-a", "session-b"],
        "haystack_dates": [
            "2023/04/08 (Sat) 09:00",
            "2023/04/09 (Sun) 10:00",
        ],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I booked an appointment.", "has_answer": True},
                {"role": "assistant", "content": "Noted.", "has_answer": False},
            ],
            [
                {"role": "user", "content": "I went shopping."},
                {"role": "assistant", "content": "Sounds good."},
            ],
        ],
        "answer_session_ids": ["session-a"],
    }


def _write_source(path: Path, records: list[dict]) -> str:
    path.write_text(json.dumps(records), encoding="utf-8")
    return sha256_file(path)


def test_pinned_dataset_urls_use_immutable_revisions_and_sha256():
    specs = [LOCOMO, *LONGMEM_FILES.values()]
    for spec in specs:
        assert spec.revision in spec.url
        assert "/main/" not in spec.url
        assert len(spec.sha256) == 64
        int(spec.sha256, 16)
        assert spec.size > 0


def test_streaming_json_array_parser_handles_small_chunks(tmp_path: Path):
    source = tmp_path / "source.json"
    records = [_record(), _record("second_abs", "single-session-user")]
    _write_source(source, records)

    assert list(iter_json_array(source, chunk_size=7)) == records


def test_converter_writes_runner_compatible_question_csvs(tmp_path: Path):
    source = tmp_path / "longmem.json"
    records = [_record(), _record("second_abs", "single-session-user")]
    digest = _write_source(source, records)
    output = tmp_path / "script_data"

    summary = convert_longmem_dataset(
        source,
        output,
        source_revision="a" * 40,
        source_sha256=digest,
        variant="s",
    )

    assert summary.records == 2
    assert summary.rows == 8
    assert summary.category_counts == {
        "single_session_user": 1,
        "temporal_reasoning": 1,
    }
    target = output / "temporal_reasoning" / "gpt4_1234abcd.csv"
    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["turn_index"] for row in rows[:2]] == ["1", "2"]
    assert {row["question"] for row in rows} == {"What happened first?"}
    assert rows[0]["dialogue_datetime"] == "2023/04/08 (Sat) 09:00"
    assert rows[0]["has_answer"] == "True"
    assert rows[0]["is_answer_session"] == "True"
    assert rows[2]["is_answer_session"] == "False"
    assert (output / "single_session_user" / "second_abs.csv").is_file()

    repeated = convert_longmem_dataset(
        source,
        output,
        source_revision="a" * 40,
        source_sha256=digest,
        variant="s",
    )
    assert repeated.skipped is True


def test_converter_preserves_numeric_gold_answers_as_text(tmp_path: Path):
    source = tmp_path / "numeric.json"
    record = _record()
    record["answer"] = 3
    digest = _write_source(source, [record])
    output = tmp_path / "script_data"

    convert_longmem_dataset(
        source,
        output,
        source_revision="d" * 40,
        source_sha256=digest,
        variant="s",
    )

    with (output / "temporal_reasoning" / "gpt4_1234abcd.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["answer"] == "3"


def test_converter_rejects_misaligned_session_arrays(tmp_path: Path):
    source = tmp_path / "bad.json"
    record = _record()
    record["haystack_dates"] = []
    digest = _write_source(source, [record])

    with pytest.raises(ValueError, match="mismatched"):
        convert_longmem_dataset(
            source,
            tmp_path / "output",
            source_revision="b" * 40,
            source_sha256=digest,
            variant="s",
        )


def test_converter_rejects_wrong_source_checksum(tmp_path: Path):
    source = tmp_path / "source.json"
    _write_source(source, [_record()])

    with pytest.raises(ValueError, match="checksum mismatch"):
        convert_longmem_dataset(
            source,
            tmp_path / "output",
            source_revision="c" * 40,
            source_sha256="0" * 64,
            variant="oracle",
        )
