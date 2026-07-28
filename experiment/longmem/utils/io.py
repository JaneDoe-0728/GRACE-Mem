from __future__ import annotations

import csv
import fcntl
import json
import os
import shutil
import tempfile
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_tree(path: Path, *, missing_ok: bool = True) -> None:
    if path.exists():
        shutil.rmtree(path)
    elif not missing_ok:
        raise FileNotFoundError(path)


def remove_file(path: Path, *, missing_ok: bool = True) -> None:
    if path.exists():
        path.unlink()
    elif not missing_ok:
        raise FileNotFoundError(path)


def move_file(src: Path, dst: Path, *, ensure_parent: bool = True) -> Path:
    if ensure_parent:
        ensure_dir(dst.parent)
    return Path(shutil.move(str(src), str(dst)))


def glob_sorted(folder: Path, pattern: str) -> list[Path]:
    return sorted(folder.glob(pattern))


def latest_glob_match(folder: Path, pattern: str) -> Path | None:
    matches = glob_sorted(folder, pattern)
    return matches[-1] if matches else None


def read_text_file(path: Path, *, encoding: str = "utf-8", default: str | None = None) -> str | None:
    try:
        return path.read_text(encoding=encoding)
    except Exception:
        return default


def write_text_file(path: Path, text: str, *, ensure_parent: bool = True, encoding: str = "utf-8") -> None:
    if ensure_parent:
        ensure_dir(path.parent)
    path.write_text(text, encoding=encoding)


def read_json_file(path: Path, *, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json_file(path: Path, data: Any, *, ensure_parent: bool = True, indent: int = 2) -> None:
    if ensure_parent:
        ensure_dir(path.parent)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )


def append_jsonl(path: Path, record: dict, *, ensure_parent: bool = True) -> None:
    if ensure_parent:
        ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl_file(path: Path, *, encoding: str = "utf-8") -> list[dict]:
    if not path.exists():
        return []
    lines: list[dict] = []
    with open(path, encoding=encoding) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return lines


def read_csv_frame(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


@contextmanager
def file_lock(path: Path):
    ensure_dir(path.parent)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_replace(temp_path: Path, target_path: Path) -> None:
    with open(temp_path, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp_path, target_path)


def write_csv_frame(df: pd.DataFrame, path: Path, **kwargs) -> None:
    ensure_dir(path.parent)
    options = {"index": False, "encoding": "utf-8-sig"}
    options.update(kwargs)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tmp",
        prefix=f"{path.name}.",
        dir=path.parent,
        delete=False,
        encoding=options.get("encoding", "utf-8-sig"),
        newline="",
    ) as handle:
        temp_path = Path(handle.name)
        df.to_csv(handle, **options)
    _atomic_replace(temp_path, path)


def upsert_csv_row(
    path: Path,
    row: dict[str, Any],
    *,
    key_columns: list[str],
    read_kwargs: dict[str, Any] | None = None,
    write_kwargs: dict[str, Any] | None = None,
) -> None:
    row_str = {key: str(value) for key, value in row.items()}
    if path.exists():
        df = read_csv_frame(path, **(read_kwargs or {"dtype": str}))
    else:
        df = pd.DataFrame()

    if all(column in df.columns for column in key_columns) and key_columns:
        mask = pd.Series(True, index=df.index)
        for column in key_columns:
            mask = mask & (df[column].astype(str) == row_str.get(column, ""))
    else:
        mask = pd.Series([], dtype=bool)

    if mask.any():
        for key, value in row_str.items():
            df.loc[mask, key] = value
    else:
        df = pd.concat([df, pd.DataFrame([row_str])], ignore_index=True)

    write_csv_frame(df, path, **(write_kwargs or {}))


def read_csv_dict_rows(path: Path, *, encoding: str = "utf-8-sig", newline: str = "") -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline=newline, encoding=encoding) as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv_dict_rows(
    path: Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
    encoding: str = "utf-8-sig",
    newline: str = "",
) -> None:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tmp",
        prefix=f"{path.name}.",
        dir=path.parent,
        delete=False,
        encoding=encoding,
        newline=newline,
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _atomic_replace(temp_path, path)


def write_status_file(status_path: Path, data: dict[str, Any]) -> None:
    payload = dict(data)
    payload["updated_at"] = datetime.now().isoformat()
    try:
        write_json_file(status_path, payload)
    except Exception:
        pass


def list_run_targets(data_root: Path) -> list[Path]:
    subfolders = sorted(path for path in data_root.iterdir() if path.is_dir())
    return subfolders if subfolders else [data_root]


def has_subfolders(data_root: Path) -> bool:
    return any(path.is_dir() for path in data_root.iterdir())


def resolve_output_dir(data_root: Path, output_root: Path, folder: Path) -> Path:
    if has_subfolders(data_root):
        return output_root / folder.name
    return output_root / data_root.name


def resolve_batch_output_root(data_root: Path, output_root: Path) -> Path:
    if has_subfolders(data_root):
        return output_root
    return output_root / data_root.name


def append_type_subdir(base_dir: Path, type_name: str | None) -> Path:
    if not type_name:
        return base_dir
    return base_dir / type_name


def append_progress_stuck_history(output_dir: Path, dataset_name: str, stuck_entry: str) -> None:
    from experiment.longmem.helpers.progress import append_stuck_history_entry

    append_stuck_history_entry(
        output_dir,
        dataset=dataset_name,
        entry=stuck_entry,
        filename="progress.csv",
    )
