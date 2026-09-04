"""Filesystem helpers shared by the LoCoMo runner and its stages.

Everything here is append-only or create-if-absent. Samples run concurrently
and write into the same run root, so a helper that truncated or rewrote a
shared file would let one worker discard another's output. `EVAL_COLUMNS`
fixes the evaluation CSV's column order, which is what allows per-sample CSVs
to be concatenated later without reconciling headers.
"""

import csv
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiment.common.reproducibility import attach_reproducibility_metadata
from grace_mem.utils.paths import resolve_artifacts_dir

EVAL_COLUMNS = [
    "question",
    "gold_answer",
    "gold_evidence_source",
    "model_answer",
    "retrieved_context",
    "rendered_evidence",
    "retrieval_request_id",
    "retrieval_stop_reason",
    "retrieval_failure_type",
    "retrieval_confidence",
    "retrieval_tau",
    "selected_evidence_count",
    "selected_evidence_ids",
    "selected_evidence_preview",
    "final_entity_names",
    "final_relationship_names",
    "anomaly_flags",
    "pass2_triggered",
    "pass1_entity_ids",
    "pass2_entity_ids",
    "pass1_relation_ids",
    "pass2_relation_ids",
    "entity_overlap_count",
    "relation_overlap_count",
    "entity_overlap_pct",
    "relation_overlap_pct",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def copy_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def remove_if_exists(path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        target.unlink()


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON file expected to hold a list of records."""
    target = Path(path)
    with target.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        records = [data]
    elif isinstance(data, list):
        records = [item for item in data if isinstance(item, dict)]
    else:
        raise ValueError(f"{target} must contain a JSON object or array of objects")
    if not records:
        raise ValueError(f"{target} does not contain any records")
    return records


def load_csv_rows(path: str | Path, *, encoding: str = "utf-8-sig") -> list[dict[str, Any]]:
    target = Path(path)
    with target.open("r", encoding=encoding, newline="") as fh:
        return list(csv.DictReader(fh))


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping unparseable lines.

    Tolerant for the same reason as the LongMem equivalent: trace files are
    appended to during a run, so the last line is frequently incomplete.
    """
    target = Path(path)
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON decode error at line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{target} line {line_number} must contain a JSON object")
            records.append(payload)
    return records


def append_jsonl_record(path: str | Path, record: dict[str, Any]) -> None:
    target = ensure_parent_dir(Path(path))
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_text(path: str | Path, text: str) -> None:
    target = ensure_parent_dir(Path(path))
    with target.open("a", encoding="utf-8") as fh:
        fh.write(text)


def token_usage_log_path(run_root: Path, sample_index: int) -> Path:
    path = run_root / "logs" / "tokens_usages" / f"tokens_usage{sample_index}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def backup_artifacts_and_logs(
    sample_dir: Path,
    *,
    also_copy: Sequence[Path],
    include_artifacts: bool = True,
) -> None:
    """Copy a sample's artifacts and logs aside before the next sample overwrites them.

    Samples reuse one working artifacts directory, so without this each sample
    destroys the previous one's evidence -- and a failure is usually only
    diagnosable from the state of the run that produced it.
    """
    if include_artifacts:
        copy_dir(resolve_artifacts_dir(), sample_dir / "artifacts")
    copy_dir(Path("./logs"), sample_dir / "logs")
    for path in also_copy:
        if path.exists():
            dest = sample_dir / path.name
            if path.is_dir():
                copy_dir(path, dest)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if path.resolve() != dest.resolve():
                    shutil.copy2(path, dest)


def write_stats_json(path: str | Path, stats: dict[str, Any]) -> None:
    stats_path = Path(path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(attach_reproducibility_metadata(stats), fh, ensure_ascii=False, indent=2)


def write_eval_csv(
    *,
    pandas_module: Any,
    eval_csv: str | Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    """Write evaluation rows with the canonical column order.

    EVAL_COLUMNS fixes the order so per-sample CSVs concatenate later without
    reconciling headers.
    """
    output_path = Path(eval_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pandas_module.DataFrame(list(rows), columns=EVAL_COLUMNS).to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
    )


def write_empty_eval_csv(*, pandas_module: Any, eval_csv: str | Path) -> None:
    write_eval_csv(pandas_module=pandas_module, eval_csv=eval_csv, rows=[])
