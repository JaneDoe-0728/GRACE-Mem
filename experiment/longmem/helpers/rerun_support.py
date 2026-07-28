from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from experiment.longmem.decision import retrieval_context_needs_rerun
from experiment.longmem.utils.io import glob_sorted, read_csv_frame, read_json_file, upsert_csv_row


ARTIFACT_MARKERS = (
    "entities_cache.pkl",
    "relationships_cache.pkl",
    "entities_meta.jsonl",
    "relationships_meta.jsonl",
    "summaries_meta.jsonl",
)


def looks_like_artifact_dir(path: Path) -> bool:
    return path.is_dir() and any((path / marker).exists() for marker in ARTIFACT_MARKERS)


def candidate_artifact_dirs(root: Path, dataset_name: str) -> list[Path]:
    return [
        root / dataset_name,
        root / f"artifacts_{dataset_name}",
    ]


def resolve_artifact_dir(root: Path, dataset_name: str) -> Path | None:
    for path in candidate_artifact_dirs(root, dataset_name):
        if looks_like_artifact_dir(path):
            return path
    return None


def failed_datasets(output_dir: Path, specified: list[str] | None) -> list[str]:
    if specified:
        return specified

    progress_path = output_dir / "progress.csv"
    if not progress_path.exists():
        raise FileNotFoundError(f"progress.csv not found: {progress_path}")

    df = read_csv_frame(progress_path, dtype=str)
    failed = df[df["correctness"].astype(str).str.strip() == "0"]
    return list(failed["dataset"].astype(str).str.strip())


def retrieval_datasets(
    output_dir: Path,
    specified: list[str] | None,
    *,
    artifact_dir: Path | None = None,
    force: bool = False,
) -> list[str]:
    """Return dataset names that need retrieval rerun.

    artifact_dir: when set, scans this directory for artifacts_<name>/ subdirs
    instead of output_dir. Useful for retrieval-only runs against pre-built stores.
    """
    scan_dir = artifact_dir if artifact_dir is not None else output_dir

    if specified:
        candidates = [output_dir / f"{name}.csv" for name in specified]
    else:
        skip = {"all_answers", "all_answers_judged_0316", "progress0323"}
        candidates = [path for path in glob_sorted(output_dir, "*.csv") if path.stem not in skip]

    selected: list[str] = []
    for csv_path in candidates:
        if force or output_csv_needs_rerun(csv_path):
            if resolve_artifact_dir(scan_dir, csv_path.stem) is not None:
                selected.append(csv_path.stem)
    return selected


def retrieval_datasets_from_artifacts(
    artifact_dir: Path,
    specified: list[str] | None,
) -> list[str]:
    """Discover dataset names by scanning artifact_dir for artifacts_<name>/ subdirs.

    Used when output CSVs don't exist yet (pure retrieval-only run with no prior output).
    """
    if specified:
        return [name for name in specified if resolve_artifact_dir(artifact_dir, name) is not None]

    names: set[str] = set()
    for subdir in sorted(artifact_dir.iterdir()):
        if not looks_like_artifact_dir(subdir):
            continue
        if subdir.name.startswith("artifacts_"):
            names.add(subdir.name[len("artifacts_"):])
        else:
            names.add(subdir.name)
    return sorted(names)


def output_csv_needs_rerun(csv_path: Path) -> bool:
    try:
        df = read_csv_frame(csv_path)
        if "Retrieved_Context" not in df.columns or len(df) == 0:
            return True
        context = str(df["Retrieved_Context"].iloc[0]).strip()
        return retrieval_context_needs_rerun(context)
    except Exception:
        return True


def setup_retrieval_loggers(dataset_name: str, log_dir: Path) -> None:
    from KG.utils.logger_config import make_module_jlog

    import KG.pipeline.retrieval_steps.evidence as evidence_module
    import KG.pipeline.retrieval_steps.filtering as filtering_module
    import KG.pipeline.retrieval_steps.search as search_module
    import KG.pipeline.retrieval_steps.temporal as temporal_module
    import KG.pipeline.retriever as retriever_module

    retriever_module._jlog = make_module_jlog(
        name=f"KG.Retriever.{dataset_name}",
        filename="kg_retriever.jsonl",
        log_dir=str(log_dir),
    )
    search_module._jlog = make_module_jlog(
        name=f"KG.Retrieval.Search.{dataset_name}",
        filename="kg_retrieval_search.jsonl",
        log_dir=str(log_dir),
    )
    filtering_module._jlog = make_module_jlog(
        name=f"KG.Retrieval.Filtering.{dataset_name}",
        filename="kg_retrieval_filtering.jsonl",
        log_dir=str(log_dir),
    )
    temporal_module._jlog = make_module_jlog(
        name=f"KG.Retrieval.Temporal.{dataset_name}",
        filename="kg_retrieval_temporal.jsonl",
        log_dir=str(log_dir),
    )
    evidence_module._jlog = make_module_jlog(
        name=f"KG.Retrieval.Evidence.{dataset_name}",
        filename="kg_retrieval_evidence.jsonl",
        log_dir=str(log_dir),
    )

    # Redirect trace_pretty_log — module-level logger initialised at import time,
    # so handlers must be replaced to point at this dataset's log_dir.
    trace_logger = logging.getLogger("kg_retrieval_trace_pretty")
    for handler in list(trace_logger.handlers):
        trace_logger.removeHandler(handler)
        handler.close()
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        str(log_dir / "kg_retrieval_trace_pretty.log"),
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    trace_logger.setLevel(logging.INFO)
    trace_logger.addHandler(fh)

    # Redirect reranker CSV to absolute path so it lands in log_dir regardless of cwd.
    filtering_module._RERANKER_SCORE_CSV = str(log_dir / "reranker_scores.csv")


def cleanup_retrieval_loggers(log_dir: Path) -> None:
    from KG.utils.logger_config import close_event_loggers

    close_event_loggers(log_dir=str(log_dir))

    trace_logger = logging.getLogger("kg_retrieval_trace_pretty")
    for handler in list(trace_logger.handlers):
        trace_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def rerun_accuracy(results: list[dict]) -> tuple[int, int]:
    judged = [row for row in results if str(row.get("correctness", "")).strip() in ("0", "1")]
    correct = sum(1 for row in judged if str(row.get("correctness", "")).strip() == "1")
    return correct, len(judged)


def read_summary_accuracy(summary_path: Path) -> tuple[int, int]:
    data = read_json_file(summary_path, default=[]) or []
    return rerun_accuracy(data)


def upsert_result_csv(path: Path, row: dict) -> None:
    upsert_csv_row(path, row, key_columns=["variant"])
