"""Convert an official LongMemEval JSON release into GRACE-Mem question CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator


CATEGORY_MAP = {
    "single-session-user": "single_session_user",
    "single-session-assistant": "single_session_assistant",
    "single-session-preference": "single_session_preference",
    "multi-session": "multi_session",
    "temporal-reasoning": "temporal_reasoning",
    "knowledge-update": "knowledge_update",
}

CSV_COLUMNS = (
    "question_id",
    "question_type",
    "session_id",
    "turn_index",
    "role",
    "content",
    "dialogue_datetime",
    "question",
    "answer",
    "question_date",
    "has_answer",
    "is_answer_session",
    "answer_session_ids",
)

MANIFEST_NAME = "dataset_manifest.json"
SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class ConversionSummary:
    source_file: str
    source_revision: str
    source_sha256: str
    variant: str
    records: int
    rows: int
    category_counts: dict[str, int]
    generated_files: list[str]
    generated_sha256: dict[str, str]
    skipped: bool = False


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_json_array(path: Path, *, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """Yield values from one top-level JSON array without loading it all."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    first = True
    eof = False

    with path.open("r", encoding="utf-8") as handle:
        while True:
            if not eof and len(buffer) < chunk_size:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            buffer = buffer.lstrip()
            if not started:
                if not buffer and not eof:
                    continue
                if not buffer.startswith("["):
                    raise ValueError(f"LongMemEval source must be a JSON array: {path}")
                buffer = buffer[1:]
                started = True
                continue

            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                buffer = buffer[1:] + handle.read()
                if buffer.strip():
                    raise ValueError(f"Unexpected content after JSON array: {path}")
                return

            value_buffer = buffer
            if not first:
                if not buffer:
                    if eof:
                        raise ValueError(f"Unterminated JSON array: {path}")
                    continue
                if buffer[0] != ",":
                    raise ValueError(f"Expected ',' between LongMemEval records: {path}")
                value_buffer = buffer[1:].lstrip()

            try:
                value, end = decoder.raw_decode(value_buffer)
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError(f"Invalid LongMemEval JSON in {path}: {exc}") from exc
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue

            yield value
            buffer = value_buffer[end:]
            first = False


def _required(record: dict[str, Any], key: str, expected_type: type) -> Any:
    value = record.get(key)
    if not isinstance(value, expected_type):
        raise ValueError(
            f"Question {record.get('question_id', '<unknown>')} has invalid {key!r}; "
            f"expected {expected_type.__name__}"
        )
    return value


def _text_scalar(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    raise ValueError(
        f"Question {record.get('question_id', '<unknown>')} has invalid {key!r}; "
        "expected a string or number"
    )


def _rows_for_record(record: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    question_id = _required(record, "question_id", str).strip()
    if not question_id or not _SAFE_ID.fullmatch(question_id):
        raise ValueError(f"Unsafe or empty LongMemEval question_id: {question_id!r}")

    question_type = _required(record, "question_type", str).strip()
    try:
        category = CATEGORY_MAP[question_type]
    except KeyError as exc:
        raise ValueError(
            f"Question {question_id} has unsupported question_type: {question_type!r}"
        ) from exc

    question = _required(record, "question", str)
    answer = _text_scalar(record, "answer")
    question_date = _required(record, "question_date", str)
    session_ids = _required(record, "haystack_session_ids", list)
    session_dates = _required(record, "haystack_dates", list)
    sessions = _required(record, "haystack_sessions", list)
    answer_session_ids = _required(record, "answer_session_ids", list)

    if not (len(session_ids) == len(session_dates) == len(sessions)):
        raise ValueError(
            f"Question {question_id} has mismatched haystack_session_ids, "
            "haystack_dates, and haystack_sessions lengths"
        )
    if not sessions:
        raise ValueError(f"Question {question_id} has no haystack sessions")

    answer_session_set = {str(value) for value in answer_session_ids}
    encoded_answer_sessions = json.dumps(answer_session_ids, ensure_ascii=True)
    rows: list[dict[str, Any]] = []

    for session_id_value, session_date, turns in zip(
        session_ids, session_dates, sessions, strict=True
    ):
        session_id = str(session_id_value).strip()
        if not session_id:
            raise ValueError(f"Question {question_id} contains an empty session id")
        if not isinstance(session_date, str) or not session_date.strip():
            raise ValueError(f"Question {question_id} session {session_id} has no date")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"Question {question_id} session {session_id} has no turns")

        for turn_index, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                raise ValueError(
                    f"Question {question_id} session {session_id} has a non-object turn"
                )
            role = turn.get("role")
            content = turn.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                raise ValueError(
                    f"Question {question_id} session {session_id} turn {turn_index} "
                    "must contain a user/assistant role and string content"
                )
            has_answer = turn.get("has_answer", False)
            if not isinstance(has_answer, bool):
                raise ValueError(
                    f"Question {question_id} session {session_id} turn {turn_index} "
                    "has a non-boolean has_answer"
                )

            rows.append(
                {
                    "question_id": question_id,
                    "question_type": question_type,
                    "session_id": session_id,
                    "turn_index": turn_index,
                    "role": role,
                    "content": content,
                    "dialogue_datetime": session_date.strip(),
                    "question": question,
                    "answer": answer,
                    "question_date": question_date.strip(),
                    "has_answer": has_answer,
                    "is_answer_session": session_id in answer_session_set,
                    "answer_session_ids": encoded_answer_sessions,
                }
            )

    return question_id, category, rows


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _matching_complete_manifest(
    output_dir: Path,
    *,
    source_revision: str,
    source_sha256: str,
    variant: str,
) -> ConversionSummary | None:
    manifest = _read_manifest(output_dir / MANIFEST_NAME)
    if not manifest:
        return None
    expected = {
        "schema_version": SCHEMA_VERSION,
        "source_revision": source_revision,
        "source_sha256": source_sha256,
        "variant": variant,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    files = manifest.get("generated_files")
    if not isinstance(files, list) or not files:
        return None
    if not all((output_dir / rel).is_file() for rel in files):
        return None

    summary_data = {key: manifest[key] for key in ConversionSummary.__dataclass_fields__ if key in manifest}
    summary_data["skipped"] = True
    return ConversionSummary(**summary_data)


def _write_question_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        digest = sha256_file(temp_path)
        temp_path.replace(path)
        return digest
    finally:
        temp_path.unlink(missing_ok=True)


def convert_longmem_dataset(
    source: Path,
    output_dir: Path,
    *,
    source_revision: str,
    source_sha256: str,
    variant: str,
    force: bool = False,
) -> ConversionSummary:
    source = source.resolve()
    output_dir = output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"LongMemEval source does not exist: {source}")

    actual_sha256 = sha256_file(source)
    if actual_sha256 != source_sha256:
        raise ValueError(
            f"LongMemEval checksum mismatch for {source}: "
            f"expected {source_sha256}, got {actual_sha256}"
        )

    if not force:
        existing = _matching_complete_manifest(
            output_dir,
            source_revision=source_revision,
            source_sha256=source_sha256,
            variant=variant,
        )
        if existing:
            return existing
        existing_csvs = list(output_dir.glob("*/*.csv")) if output_dir.exists() else []
        if existing_csvs:
            raise FileExistsError(
                f"LongMem output already contains {len(existing_csvs)} CSV files: {output_dir}. "
                "Use --force to replace the generated dataset."
            )

    previous_manifest = _read_manifest(output_dir / MANIFEST_NAME) or {}
    previous_files = {
        rel for rel in previous_manifest.get("generated_files", []) if isinstance(rel, str)
    }
    seen_ids: set[str] = set()
    generated_files: list[str] = []
    generated_sha256: dict[str, str] = {}
    category_counts: Counter[str] = Counter()
    row_count = 0

    for index, record in enumerate(iter_json_array(source), start=1):
        if not isinstance(record, dict):
            raise ValueError(f"LongMemEval record {index} is not an object")
        question_id, category, rows = _rows_for_record(record)
        if question_id in seen_ids:
            raise ValueError(f"Duplicate LongMemEval question_id: {question_id}")
        seen_ids.add(question_id)

        relative_path = f"{category}/{question_id}.csv"
        output_path = output_dir / relative_path
        if output_path.exists() and not force:
            raise FileExistsError(f"LongMem output already exists: {output_path}")
        generated_sha256[relative_path] = _write_question_csv(output_path, rows)
        generated_files.append(relative_path)
        category_counts[category] += 1
        row_count += len(rows)

        if index % 25 == 0:
            print(f"Converted {index} LongMemEval questions...")

    if not generated_files:
        raise ValueError(f"LongMemEval source contains no records: {source}")

    for relative_path in previous_files - set(generated_files):
        stale_path = (output_dir / relative_path).resolve()
        if output_dir in stale_path.parents:
            stale_path.unlink(missing_ok=True)

    summary = ConversionSummary(
        source_file=source.name,
        source_revision=source_revision,
        source_sha256=source_sha256,
        variant=variant,
        records=len(generated_files),
        rows=row_count,
        category_counts=dict(sorted(category_counts.items())),
        generated_files=generated_files,
        generated_sha256=generated_sha256,
    )
    manifest = {"schema_version": SCHEMA_VERSION, **asdict(summary)}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    temp_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temp_manifest.replace(manifest_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert official LongMemEval JSON into one GRACE-Mem CSV per question."
    )
    parser.add_argument("--input", type=Path, required=True, help="Official LongMemEval JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--variant", choices=("s", "m", "oracle"), required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = convert_longmem_dataset(
        args.input,
        args.output_dir,
        source_revision=args.source_revision,
        source_sha256=args.source_sha256,
        variant=args.variant,
        force=args.force,
    )
    action = "Verified existing conversion" if summary.skipped else "Converted"
    print(
        f"{action}: {summary.records} questions, {summary.rows} turns -> "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
