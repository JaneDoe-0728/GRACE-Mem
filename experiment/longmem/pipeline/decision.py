"""Resume predicates: what a previous run already finished, and what it did not.

Every function here answers one question about existing state, and they are
gathered so the answers are testable without a run and consistent between the
runner and the rerun tool.

The distinction that matters is between complete and merely present. An output
file exists as soon as work starts on it, so `should_treat_output_as_complete`
and `retrieval_context_needs_rerun` look at contents, not existence -- treating
a truncated artifact as finished silently reports partial results as final.

`should_reset_legacy_skipped_stage` handles checkpoints written by an older
version whose stage names no longer mean the same thing.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def should_treat_output_as_complete(output_path: Path) -> bool:
    """Whether an existing output means this dataset can be skipped.

    Existence alone, deliberately. The per-question outputs are written whole
    at the end of a dataset rather than streamed, so a file that exists is a
    file that finished. Content checks live in `retrieval_context_needs_rerun`,
    which is applied per row where partial writes are actually possible.
    """
    return output_path.exists()


def should_reset_legacy_skipped_stage(checkpoint: dict) -> bool:
    """Whether a checkpoint was left behind by the watchdog and must be redone.

    The watchdog marks a stalled dataset "skipped_by_watchdog" so the run can
    move on. That is a record of abandonment, not of completion, so resuming
    must discard it -- treating it as a finished stage would permanently skip
    a dataset that never actually ran.
    """
    return checkpoint.get("stage") == "skipped_by_watchdog"


def retrieval_context_needs_rerun(context: str) -> bool:
    """Whether a stored retrieval context is unusable and must be recomputed.

    Three ways a context is empty in practice, and all three have to be caught:
    a genuinely empty string, the literal "nan" that pandas writes for a
    missing cell on round-trip, and the explicit "(no KG context)" marker the
    retriever emits when it found nothing. Missing the "nan" case is the
    subtle one -- it is a non-empty string, so a naive truthiness check keeps
    a row that has no context at all.
    """
    value = str(context or "").strip()
    return value in ("", "nan") or "(no KG context)" in value


def read_child_manifest(manifest_path: str | Path) -> list[tuple[str, str]]:
    """Parse a child manifest of "category,dataset" lines.

    Strict: a malformed line raises rather than being skipped, and an empty
    manifest is an error too. A sweep driven by this file must not quietly
    cover fewer datasets than intended -- the missing ones would simply be
    absent from the results, indistinguishable from datasets that scored zero.

    Blank lines and "#" comments are ignored.

    Raises:
        ValueError: Manifest missing, a line malformed, or no entries found.
    """
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
