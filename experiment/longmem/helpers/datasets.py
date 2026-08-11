from __future__ import annotations

from pathlib import Path
import re

from experiment.longmem.pipeline.decision import (
    filter_child_entries,
    group_child_entries,
    read_child_manifest,
    retrieval_context_needs_rerun,
)
from experiment.longmem.utils.io import glob_sorted, read_csv_frame


def discover_csv_datasets(folder_path: str, file_pattern: str = "*.csv") -> list[Path]:
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")
    csv_files = glob_sorted(folder, file_pattern)
    if not csv_files:
        raise ValueError(f"No files matching '{file_pattern}' found in {folder_path}")
    return csv_files


def resolve_child_datasets(
    data_root: str | Path,
    manifest_path: str | Path,
    *,
    type_name: list[str] | str | None = None,
) -> dict[str, list[Path]]:
    root = Path(data_root)
    entries = filter_child_entries(read_child_manifest(manifest_path), type_name)
    if not entries:
        label = (", ".join(type_name) if isinstance(type_name, list) else type_name) or "requested filter"
        raise ValueError(f"No child datasets found for {label}")

    grouped_ids = group_child_entries(entries)
    grouped_paths: dict[str, list[Path]] = {}
    missing: list[str] = []
    for category, dataset_ids in grouped_ids.items():
        paths: list[Path] = []
        category_dir = root / category
        for dataset_id in dataset_ids:
            csv_path = category_dir / f"{dataset_id}.csv"
            if csv_path.exists():
                paths.append(csv_path)
            else:
                missing.append(f"{category}/{dataset_id}.csv")
        if paths:
            grouped_paths[category] = sorted(paths)

    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise ValueError(f"Missing child dataset CSVs: {preview}{suffix}")

    return grouped_paths


_RANGE_PATTERN = re.compile(r"^(?P<start>\d+)-(?P<end>\d+)$")


def select_dataset_names(
    dataset_names: list[str],
    selector: str | None,
    *,
    scope_label: str,
) -> list[str]:
    if not selector:
        return list(dataset_names)

    ordered_names = sorted(dataset_names)
    by_name = {name: name for name in ordered_names}
    selected: list[str] = []
    seen: set[str] = set()

    def add_name(name: str) -> None:
        if name in seen:
            return
        selected.append(name)
        seen.add(name)

    for raw_token in selector.split(","):
        token = raw_token.strip()
        if not token:
            continue

        if token in by_name:
            add_name(by_name[token])
            continue

        match = _RANGE_PATTERN.fullmatch(token)
        if match:
            start = int(match.group("start"))
            end = int(match.group("end"))
            if start > end:
                raise ValueError(f"Invalid dataset range '{token}' for {scope_label}: start must be <= end")
            if end >= len(ordered_names):
                raise ValueError(
                    f"Dataset range '{token}' is out of bounds for {scope_label}: "
                    f"valid indices are 0-{max(len(ordered_names) - 1, 0)}"
                )
            for index in range(start, end + 1):
                add_name(ordered_names[index])
            continue

        if token.isdigit():
            index = int(token)
            if index >= len(ordered_names):
                raise ValueError(
                    f"Dataset index '{token}' is out of bounds for {scope_label}: "
                    f"valid indices are 0-{max(len(ordered_names) - 1, 0)}"
                )
            add_name(ordered_names[index])
            continue

        raise ValueError(
            f"Unknown dataset selector '{token}' for {scope_label}. "
            "Use dataset IDs or numeric indices/ranges like 0,3-5."
        )

    if not selected:
        raise ValueError(f"No datasets matched selector '{selector}' for {scope_label}")
    return selected


def select_datasets(
    csv_paths: list[Path],
    selector: str | None,
    *,
    scope_label: str,
) -> list[Path]:
    selected_names = select_dataset_names(
        [path.stem for path in csv_paths],
        selector,
        scope_label=scope_label,
    )
    by_name = {path.stem: path for path in sorted(csv_paths)}
    return [by_name[name] for name in selected_names]


def get_question_info(dataset_name: str, data_folder: Path | None, output_csv: Path) -> tuple[str, str | None, str]:
    import pandas as pd

    if data_folder is not None:
        src = data_folder / f"{dataset_name}.csv"
        if src.exists():
            df = read_csv_frame(src)
            q_col = "question" if "question" in df.columns else df.columns[1]
            question = str(df[q_col].dropna().iloc[0]).strip()
            question_date = None
            for col in ("question_date", "dialogue_datetime", "date", "timestamp"):
                if col in df.columns and not df[col].dropna().empty:
                    question_date = str(df[col].dropna().iloc[0]).strip()
                    break
            gold = str(df["answer"].dropna().iloc[0]) if "answer" in df.columns and not df["answer"].dropna().empty else ""
            return question, question_date, gold

    df = read_csv_frame(output_csv)
    question = str(df["question"].iloc[0]).strip() if "question" in df.columns else ""
    question_date = str(df["question_date"].iloc[0]).strip() if "question_date" in df.columns else None
    gold = str(df["answer"].iloc[0]).strip() if "answer" in df.columns else ""
    if question_date in ("", "nan"):
        question_date = None
    return question, question_date, gold


def output_csv_needs_rerun(csv_path: Path) -> bool:
    import pandas as pd

    try:
        df = read_csv_frame(csv_path)
        if "Retrieved_Context" not in df.columns or len(df) == 0:
            return True
        context = str(df["Retrieved_Context"].iloc[0]).strip()
        return retrieval_context_needs_rerun(context)
    except Exception:
        return True
