"""Append-only writers for the per-run analysis artifacts.

These are the primitives, not the analysis. `_JSONL_FILES` is a closed set of
artifact names because the analysis scripts load these files by name; a free
filename would let a typo create an orphan nothing ever reads. Records are
appended and never rewritten: writes come from concurrent workers, and append is
the only operation that stays coherent without coordination.

Lives in `utils/` rather than with the analysis that reads these files, because
the ingestion pipeline writes here too. The analysis itself belongs to the
benchmark harness -- see `experiment/common/error_analysis.py`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

_JSONL_FILES = {
    "ingest_delta": "error_analysis_ingest_delta.jsonl",
    "retrieval_summary": "error_analysis_retrieval_summary.jsonl",
    "failure_verdict": "error_analysis_failure_verdicts.jsonl",
    "drop_reasons": "error_analysis_drop_reasons.jsonl",
    "top_miss": "error_analysis_top_miss_candidates.jsonl",
    "anomaly_flags": "error_analysis_anomaly_flags.jsonl",
    "evidence_bridge": "error_analysis_evidence_bridge.jsonl",
    "grep_agent": "error_analysis_grep_agent.jsonl",
}


# ---------- IO helpers ----------

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)


# ---------- Public API ----------

def timestamp_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_analysis_record(log_dir: str | Path, artifact: str, record: dict[str, Any]) -> Path:
    """Append one record to the JSONL file for `artifact`.

    Args:
        artifact: Key into `_JSONL_FILES`. Deliberately a closed set rather
            than a free filename -- the analysis scripts read these by name, so
            a typo would create an orphan file that silently never gets read.

    Returns:
        The file written to.

    Raises:
        KeyError: If `artifact` is not a known artifact type.
    """
    target_dir = _ensure_dir(Path(log_dir))
    filename = _JSONL_FILES[artifact]
    payload = {"logged_at": timestamp_now(), **record}
    path = target_dir / filename
    _append_jsonl_record(path, payload)
    return path


def append_pretty_block(log_dir: str | Path, filename: str, text: str) -> Path:
    """Append a human-readable block, blank-line separated from the previous one.

    The separator is only written when the file is non-empty, so the file never
    opens with a stray blank line -- these are read by eye, and consistent
    block boundaries are what make them scannable.
    """
    target_dir = _ensure_dir(Path(log_dir))
    path = target_dir / filename
    if path.exists() and path.stat().st_size > 0:
        _append_text(path, "\n")
    _append_text(path, text.rstrip() + "\n")
    return path

