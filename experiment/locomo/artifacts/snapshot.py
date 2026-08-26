"""Validated, per-session snapshots for a standard LoCoMo sample.

Each worker keeps snapshots inside its own sample directory::

    sample_<index>/snapshots/session_<id>/
        <vector-store artifacts>
        graph_export.json
        snapshot_meta.json

The metadata records the dataset and ingest settings that produced the state,
plus a SHA-256 manifest for every payload file. A snapshot is never restored
until its compatibility, manifest, graph export, and vector-store artifacts
have all been validated.

Call order is deliberate:

* flush persistent storage, then :func:`save_snapshot`;
* :func:`load_snapshot_files_only` before constructing the pipeline;
* :func:`restore_graph` after the pipeline has opened its graph connection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiment.locomo.utils.graph import (
    ARTIFACTS_SRC,
    GRAPH_EXPORT_FILE,
    SNAPSHOT_META_FILE,
    restore_graph_from_export_file,
    validate_graph_export,
    validate_vdb_artifacts,
    write_graph_export,
)

logger = logging.getLogger(__name__)

SNAPSHOT_FORMAT_VERSION = 1
SNAPSHOTS_DIR = "snapshots"


class SnapshotError(RuntimeError):
    """Base error for snapshot validation and restore failures."""


class SnapshotCompatibilityError(SnapshotError):
    """The snapshot was produced by a different dataset or ingest config."""


class SnapshotCorruptionError(SnapshotError):
    """The snapshot is missing files or its persisted bytes have changed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_sha256(path: str | Path) -> str:
    """Return a stable content fingerprint for a dataset/session source file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Snapshot source file not found: {source}")
    return _sha256_file(source)


def build_snapshot_compatibility(
    *,
    sample_index: int,
    sample_id: str,
    dataset_json_path: str | Path,
    session_source_path: str | Path,
    ingest_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the metadata subset that must match before a resume is allowed."""
    try:
        normalized_config = json.loads(
            json.dumps(dict(ingest_config), ensure_ascii=False, sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("ingest_config must contain JSON-serializable values") from exc

    return {
        "dataset": "locomo",
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "dataset_sha256": dataset_sha256(dataset_json_path),
        "session_source_sha256": dataset_sha256(session_source_path),
        "ingest_config": normalized_config,
    }


def snapshot_dir(sample_dir: str | Path, session_id: int) -> Path:
    """Return the directory holding one session's persisted state."""
    return Path(sample_dir) / SNAPSHOTS_DIR / f"session_{int(session_id)}"


def snapshot_exists(sample_dir: str | Path, session_id: int) -> bool:
    """Return whether any snapshot directory exists for ``session_id``.

    Existence is intentionally weaker than validity. Call
    :func:`validate_snapshot` before using the state.
    """
    return snapshot_dir(sample_dir, session_id).is_dir()


def _payload_manifest(base_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(base_dir).as_posix(): _sha256_file(path)
        for path in sorted(base_dir.rglob("*"))
        if path.is_file() and path.name != SNAPSHOT_META_FILE
    }


def _read_metadata(snapshot_path: Path) -> dict[str, Any]:
    meta_path = snapshot_path / SNAPSHOT_META_FILE
    if not meta_path.is_file():
        raise SnapshotCorruptionError(f"Snapshot metadata missing: {meta_path}")
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SnapshotCorruptionError(
            f"Snapshot metadata is unreadable: {meta_path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise SnapshotCorruptionError(f"Snapshot metadata must be an object: {meta_path}")
    return metadata


def validate_snapshot(
    sample_dir: str | Path,
    session_id: int,
    *,
    expected_compatibility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one snapshot and return its metadata.

    Compatibility failures are distinct from damaged files so callers can
    explain whether the user changed settings or the checkpoint itself broke.
    """
    path = snapshot_dir(sample_dir, session_id)
    if not path.is_dir():
        raise FileNotFoundError(f"Snapshot not found: {path}")

    metadata = _read_metadata(path)
    if metadata.get("format_version") != SNAPSHOT_FORMAT_VERSION:
        raise SnapshotCompatibilityError(
            "Snapshot format mismatch: "
            f"expected {SNAPSHOT_FORMAT_VERSION}, got {metadata.get('format_version')!r} "
            f"at {path}"
        )
    if metadata.get("session_id") != int(session_id):
        raise SnapshotCorruptionError(
            f"Snapshot session mismatch at {path}: "
            f"expected {session_id}, got {metadata.get('session_id')!r}"
        )

    for key, expected in (expected_compatibility or {}).items():
        actual = metadata.get(key)
        if actual != expected:
            raise SnapshotCompatibilityError(
                f"Snapshot setting mismatch for {key!r} at {path}: "
                f"expected {expected!r}, got {actual!r}"
            )

    recorded_manifest = metadata.get("files")
    if not isinstance(recorded_manifest, dict) or not recorded_manifest:
        raise SnapshotCorruptionError(f"Snapshot file manifest missing or empty: {path}")
    actual_manifest = _payload_manifest(path)
    if actual_manifest != recorded_manifest:
        missing = sorted(set(recorded_manifest) - set(actual_manifest))
        unexpected = sorted(set(actual_manifest) - set(recorded_manifest))
        changed = sorted(
            name
            for name in set(recorded_manifest) & set(actual_manifest)
            if recorded_manifest[name] != actual_manifest[name]
        )
        raise SnapshotCorruptionError(
            f"Snapshot file manifest mismatch at {path}: "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )

    try:
        validate_graph_export(path / GRAPH_EXPORT_FILE)
        validate_vdb_artifacts(path)
    except RuntimeError as exc:
        raise SnapshotCorruptionError(f"Snapshot payload is invalid at {path}: {exc}") from exc
    return metadata


def highest_existing_snapshot(
    sample_dir: str | Path,
    session_ids: Sequence[int],
    *,
    expected_compatibility: Mapping[str, Any] | None = None,
) -> int:
    """Return the highest valid snapshot in a contiguous session prefix.

    A later snapshot after a gap is treated as corrupt state rather than being
    silently ignored, since it cannot be reached by replaying from the prefix.
    """
    highest = 0
    missing_seen = False
    for session_id in sorted(set(int(value) for value in session_ids)):
        if not snapshot_exists(sample_dir, session_id):
            missing_seen = True
            continue
        if missing_seen:
            raise SnapshotCorruptionError(
                "Non-contiguous snapshots: found "
                f"session {session_id} after an earlier gap in {Path(sample_dir) / SNAPSHOTS_DIR}"
            )
        validate_snapshot(
            sample_dir,
            session_id,
            expected_compatibility=expected_compatibility,
        )
        highest = session_id
    return highest


def save_snapshot(
    sample_dir: str | Path,
    session_id: int,
    graph: Any,
    *,
    compatibility: Mapping[str, Any],
) -> Path:
    """Atomically persist vector stores and the graph after one session."""
    destination = snapshot_dir(sample_dir, session_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing snapshot: {destination}")

    temp_path = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        if ARTIFACTS_SRC.exists():
            shutil.copytree(
                ARTIFACTS_SRC,
                temp_path,
                ignore=shutil.ignore_patterns(GRAPH_EXPORT_FILE, SNAPSHOT_META_FILE),
                dirs_exist_ok=True,
            )
        validate_vdb_artifacts(temp_path)

        result = write_graph_export(temp_path / GRAPH_EXPORT_FILE, graph, validate=True)
        if result is None:
            raise SnapshotError(
                f"Graph export failed for session {session_id}; FalkorDB may be unreachable"
            )

        metadata = {
            **dict(compatibility),
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "session_id": int(session_id),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": _payload_manifest(temp_path),
        }
        (temp_path / SNAPSHOT_META_FILE).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.rename(destination)
        return destination
    except Exception:
        logger.exception(
            "Session snapshot save failed: session_id=%s destination=%s source=%s",
            session_id,
            destination,
            ARTIFACTS_SRC,
        )
        shutil.rmtree(temp_path, ignore_errors=True)
        raise


def load_snapshot_files_only(
    sample_dir: str | Path,
    session_id: int,
    *,
    expected_compatibility: Mapping[str, Any] | None = None,
) -> None:
    """Validate and install snapshot files before pipeline construction."""
    source = snapshot_dir(sample_dir, session_id)
    validate_snapshot(
        sample_dir,
        session_id,
        expected_compatibility=expected_compatibility,
    )

    ARTIFACTS_SRC.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(
        tempfile.mkdtemp(prefix=f".{ARTIFACTS_SRC.name}.restore.", dir=ARTIFACTS_SRC.parent)
    )
    try:
        shutil.copytree(
            source,
            temp_path,
            ignore=shutil.ignore_patterns(GRAPH_EXPORT_FILE, SNAPSHOT_META_FILE),
            dirs_exist_ok=True,
        )
        if ARTIFACTS_SRC.exists():
            shutil.rmtree(ARTIFACTS_SRC)
        temp_path.rename(ARTIFACTS_SRC)
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise


def restore_graph(
    sample_dir: str | Path,
    session_id: int,
    graph: Any,
    *,
    expected_compatibility: Mapping[str, Any] | None = None,
) -> None:
    """Validate and restore the graph after pipeline construction."""
    validate_snapshot(
        sample_dir,
        session_id,
        expected_compatibility=expected_compatibility,
    )
    restore_graph_from_export_file(
        graph,
        snapshot_dir(sample_dir, session_id) / GRAPH_EXPORT_FILE,
    )
