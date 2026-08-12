import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

from grace_mem.storage.paths import resolve_artifacts_dir
from experiment.common.reproducibility import attach_reproducibility_metadata


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


def load_json_object(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    with target.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{target} must contain a JSON object")
    return data


def load_csv_rows(path: str | Path, *, encoding: str = "utf-8-sig") -> list[dict[str, Any]]:
    target = Path(path)
    with target.open("r", encoding=encoding, newline="") as fh:
        return list(csv.DictReader(fh))


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
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


def append_csv_with_sample(src_csv: Path, dst_csv: Path, *, sample_index: int) -> None:
    if not src_csv.exists():
        return
    with src_csv.open("r", encoding="utf-8", newline="") as src_fh:
        reader = csv.DictReader(src_fh)
        fieldnames = reader.fieldnames or []
        out_fieldnames = ["sample"] + [name for name in fieldnames if name != "sample"]
        write_header = not dst_csv.exists()
        dst_csv.parent.mkdir(parents=True, exist_ok=True)
        with dst_csv.open("a", encoding="utf-8", newline="") as dst_fh:
            writer = csv.DictWriter(dst_fh, fieldnames=out_fieldnames, quoting=csv.QUOTE_ALL)
            if write_header:
                writer.writeheader()
            for row in reader:
                merged = {"sample": f"sample_{sample_index}"}
                merged.update(row)
                writer.writerow(merged)


def sync_logs(run_root: Path) -> None:
    copy_dir(Path("./logs"), run_root / "logs")


def token_usage_log_path(run_root: Path, sample_index: int) -> Path:
    path = run_root / "logs" / "tokens_usages" / f"tokens_usage{sample_index}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_summary_map(path: Path, per_sample_stats: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "per_sample": per_sample_stats,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def backup_artifacts_and_logs(
    sample_dir: Path,
    *,
    also_copy: Sequence[Path],
    include_artifacts: bool = True,
) -> None:
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
