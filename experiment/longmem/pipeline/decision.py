from __future__ import annotations

from collections import defaultdict
from pathlib import Path


TERMINAL_STAGES = {"qa_complete"}


def should_treat_output_as_complete(output_path: Path) -> bool:
    return output_path.exists()


def should_reset_legacy_skipped_stage(checkpoint: dict) -> bool:
    return checkpoint.get("stage") == "skipped_by_watchdog"


def next_resume_stage(*, processed_count: int, checkpoint_every_n_sessions: int) -> str:
    if checkpoint_every_n_sessions <= 0:
        return "ingest_in_progress"
    if processed_count % checkpoint_every_n_sessions == 0:
        return "ingest_in_progress"
    return "new"


def retrieval_context_needs_rerun(context: str) -> bool:
    value = str(context or "").strip()
    return value in ("", "nan") or "(no KG context)" in value


def checkpoint_is_terminal(checkpoint: dict) -> bool:
    return checkpoint.get("stage") in TERMINAL_STAGES


def read_child_manifest(manifest_path: str | Path) -> list[tuple[str, str]]:
    path = Path(manifest_path)
    if not path.exists():
        raise ValueError(f"Child manifest not found: {manifest_path}")

    entries: list[tuple[str, str]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid child manifest line {line_no}: {raw_line}")
        entries.append((parts[0], parts[1]))

    if not entries:
        raise ValueError(f"No valid child entries found in {manifest_path}")
    return entries


def filter_child_entries(entries: list[tuple[str, str]], type_name: list[str] | str | None = None) -> list[tuple[str, str]]:
    if not type_name:
        return entries
    allowed = set(type_name) if isinstance(type_name, list) else {type_name}
    return [(dataset_id, category) for dataset_id, category in entries if category in allowed]


def group_child_entries(entries: list[tuple[str, str]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for dataset_id, category in entries:
        grouped[category].append(dataset_id)
    return dict(grouped)
