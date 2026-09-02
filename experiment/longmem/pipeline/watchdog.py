"""
Watchdog runner for multi-dataset QA.
Automatically restarts the LongMem batch module until all datasets complete.

Usage:
  python -m experiment.longmem.pipeline.watchdog --run-tag my-run --type single_session_user
  python -m experiment.longmem.pipeline.watchdog --run-tag my-run --child
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

# Allow importing experiment.* and tools.* when this file is run directly.
_PIPELINE_DIR = Path(__file__).resolve().parent
_LONGMEM_ROOT = _PIPELINE_DIR.parent
if __package__ in (None, ""):
    repo_root = _PIPELINE_DIR.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiment.common.run_metadata import namespace_to_dict, write_run_metadata
from experiment.longmem.helpers.args import (
    DEFAULT_STAGES,
    add_child_args,
    add_data_args,
    add_rerun_args,
    add_run_args,
    resolve_stages,
)
from experiment.longmem.helpers.datasets import (
    resolve_child_datasets,
    select_dataset_names,
    select_datasets,
)
from experiment.longmem.helpers.progress import append_stuck_history_entry
from experiment.longmem.utils.io import (
    append_jsonl,
    append_type_subdir,
    ensure_dir,
    glob_sorted,
    list_run_targets,
    read_json_file,
    read_jsonl_file,
    resolve_batch_output_root,
    resolve_output_dir,
    write_json_file,
    write_status_file,
)

DEFAULT_OUTPUT_BASE = _LONGMEM_ROOT / "output"
BATCH_MODULE = "experiment.longmem.pipeline.batch"


class RerunTarget(NamedTuple):
    """A dataset the watchdog has decided to re-run, and why."""
    category: str | None
    data_folder: Path | None
    artifact_dir: Path
    output_root: Path


def default_output_root(run_tag: str) -> Path:
    return DEFAULT_OUTPUT_BASE / run_tag


def default_log_dir(run_tag: str) -> Path:
    return default_output_root(run_tag) / "_watchdog"


def _write_watchdog_metadata(
    metadata_path: Path,
    *,
    args,
    run_tag: str,
    data_root: Path,
    data_folder: Path,
    child_file: Path,
    output_root: Path,
    artifact_dir: Path | None,
    log_dir: Path,
    batch_module: str,
    selected_stages: list[str],
    dataset_selector: str | None,
    rerun_mode: bool,
) -> None:
    """Record what the watchdog was configured to supervise, and how.

    Written alongside the run so an interrupted or partly-skipped sweep can be
    read back later: which datasets were in scope, what counted as stuck, and
    what the watchdog was permitted to do about it.
    """
    write_run_metadata(
        metadata_path,
        {
            "entrypoint": "experiment.longmem.pipeline.watchdog",
            "run_tag": run_tag,
            "run_root": str(output_root.resolve()),
            "mode": "rerun" if rerun_mode else "batch",
            "watchdog": {
                "cli": {
                    "argv": list(getattr(args, "raw_argv", [])),
                    "resolved_args": namespace_to_dict(args),
                },
                "resolved_config": {
                    "data_root": str(data_root.resolve()),
                    "data_folder": str(data_folder.resolve()),
                    "child": bool(args.child),
                    "child_file": str(child_file.resolve()),
                    "type": args.type,
                    "output_root": str(output_root.resolve()),
                    "artifact_dir": str(artifact_dir.resolve()) if artifact_dir is not None else None,
                    "log_dir": str(log_dir.resolve()),
                    "batch_module": batch_module,
                    "python": args.python,
                    "stages": list(selected_stages),
                    "dataset_id": dataset_selector,
                    "num": args.num,
                    "no_judge": bool(args.no_judge),
                    "file_pattern": args.file_pattern,
                    "sleep": args.sleep,
                    "max_restarts": args.max_restarts,
                    "io_cooldown": args.io_cooldown,
                    "timeout": args.timeout,
                },
            },
        },
    )


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(log_dir: Path) -> logging.Logger:
    """Configure the watchdog's own log, separate from the run's.

    Separate because the watchdog outlives individual datasets and its decisions
    -- what it skipped, what it restarted -- need to stay legible next to a run
    log that is being written by several workers at once.
    """
    ensure_dir(log_dir)
    log_path = log_dir / "watchdog.log"

    logger = logging.getLogger("watchdog")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — always append so history is preserved across runs
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("=" * 60)
    logger.info("Watchdog started. Log file: %s", log_path)
    logger.info("=" * 60)
    return logger


# ---------------------------------------------------------------------------
# Dataset completion detection (unchanged logic from original)
# ---------------------------------------------------------------------------

def discover_csvs(folder: Path, pattern: str) -> list[Path]:
    return glob_sorted(folder, pattern)


TERMINAL_STAGES = {"qa_complete", "error"}


def dataset_complete(output_dir: Path, dataset_name: str) -> bool:
    """Whether a dataset finished, judged from its output rather than its status.

    The watchdog cannot trust the progress table here: a worker that died mid
    write may have left a status that never advanced past in-progress. Checking
    the output directly is what distinguishes a stalled run from a finished one
    whose bookkeeping was lost.
    """
    output_csv = output_dir / f"{dataset_name}.csv"
    if output_csv.exists():
        return True
    checkpoint = output_dir / f"checkpoint_{dataset_name}.json"
    if checkpoint.exists():
        try:
            data = read_json_file(checkpoint, default={})
            return data.get("stage") in TERMINAL_STAGES
        except Exception:
            return False
    return False


CHECKPOINT_EVERY_N = 5  # Must match DatasetConfig.checkpoint_every_n_sessions
MAX_SAME_PROGRESS_RESUMES = 3


def count_same_progress_stuck_events(
    stuck_log_path: Path,
    dataset_name: str,
    processed_count: int,
    *,
    category: str | None = None,
) -> int:
    """Count recent stuck events for the same dataset at the same progress point."""
    events = read_jsonl_file(stuck_log_path)
    matching_history = [
        entry for entry in events
        if entry.get("dataset") == dataset_name and entry.get("category") == category
    ]
    count = 0
    for entry in reversed(matching_history):
        if entry.get("processed_sessions") != processed_count:
            break
        count += 1
    return count


def mark_stuck_datasets_as_skipped(
    data_folder: Path,
    output_root: Path,
    pattern: str,
    dataset_selector: str | None,
    logger: logging.Logger,
) -> list[str]:
    """Find datasets stuck mid-run.

    For each stuck dataset:
    - Logs the event (name, processed/total, timestamp) to watchdog_stuck.jsonl.
    - Resets the stage to resume from the saved checkpoint (up to MAX_SAME_PROGRESS_RESUMES times).
    - If the same progress point is seen more than MAX_SAME_PROGRESS_RESUMES times,
      marks the checkpoint stage as "error" and skips the entire dataset.
    """
    skipped = []
    targets = list_run_targets(data_folder)
    stuck_log_path = output_root / "watchdog_stuck.jsonl"
    ensure_dir(stuck_log_path.parent)

    for folder in targets:
        csvs = select_datasets(
            discover_csvs(folder, pattern),
            dataset_selector,
            scope_label=f"folder '{folder.name}'",
        )
        if not csvs:
            continue
        output_dir = resolve_output_dir(data_folder, output_root, folder)
        for csv in csvs:
            name = csv.stem
            output_csv = output_dir / f"{name}.csv"
            if output_csv.exists():
                continue  # already done
            checkpoint_path = output_dir / f"checkpoint_{name}.json"
            if not checkpoint_path.exists():
                continue  # never started
            try:
                data = read_json_file(checkpoint_path, default={})
                stage = data.get("stage", "")
                if stage in TERMINAL_STAGES:
                    continue  # already terminal

                processed_ids = data.get("processed_session_ids", [])
                processed_count = len(processed_ids)
                total = data.get("total_sessions")
                in_sync = (processed_count % CHECKPOINT_EVERY_N == 0)

                prior_same_progress = count_same_progress_stuck_events(
                    stuck_log_path,
                    name,
                    processed_count,
                )
                action = "skip_dataset" if prior_same_progress >= MAX_SAME_PROGRESS_RESUMES else "resume"

                # Write log entry
                log_entry = {
                    "dataset": name,
                    "stuck_at": datetime.now().isoformat(),
                    "previous_stage": stage,
                    "processed_sessions": processed_count,
                    "total_sessions": total,
                    "in_sync": in_sync,
                    "same_progress_retries": prior_same_progress,
                    "action": action,
                }
                append_jsonl(stuck_log_path, log_entry)

                # Update progress.csv stuck_history
                stuck_entry = (
                    f"{datetime.now().strftime('%Y-%m-%dT%H:%M')} "
                    f"(processed {processed_count}/{total} sessions)"
                )
                try:
                    append_stuck_history_entry(output_dir, dataset=name, entry=stuck_entry)
                except Exception as exc:
                    logger.error("Failed to update progress.csv stuck_history for '%s': %s", name, exc)

                if action == "resume":
                    data["stage"] = "ingest_in_progress"
                    data["updated_at"] = datetime.now().isoformat() + "Z"
                    write_json_file(checkpoint_path, data)
                    logger.warning(
                        "Stuck dataset '%s' (was: %s) — (%d/%s), will resume next run (resume #%d).",
                        name, stage, processed_count, total, prior_same_progress + 1,
                    )
                else:
                    # Exhausted MAX_SAME_PROGRESS_RESUMES resumes at same point — mark as error
                    data["stage"] = "error"
                    data["updated_at"] = datetime.now().isoformat() + "Z"
                    write_json_file(checkpoint_path, data)
                    logger.error(
                        "Stuck dataset '%s' (was: %s) — same checkpoint (%d/%s) after %d resumes; "
                        "marking as error and skipping entire dataset.",
                        name, stage, processed_count, total, prior_same_progress + 1,
                    )

                skipped.append(name)
            except Exception as exc:
                logger.error("Failed to handle stuck dataset '%s': %s", name, exc)
    return skipped


def count_completion_selected(
    data_folder: Path,
    output_root: Path,
    pattern: str,
    dataset_selector: str | None,
) -> tuple[int, int]:
    """Returns (completed, total) for the selected datasets."""
    total = 0
    completed = 0
    targets = list_run_targets(data_folder)
    for folder in targets:
        csvs = select_datasets(
            discover_csvs(folder, pattern),
            dataset_selector,
            scope_label=f"folder '{folder.name}'",
        )
        if not csvs:
            continue
        output_dir = resolve_output_dir(data_folder, output_root, folder)
        for csv in csvs:
            total += 1
            if dataset_complete(output_dir, csv.stem):
                completed += 1
    return completed, total


def count_completion_child(
    data_root: Path,
    output_root: Path,
    child_file: Path,
    type_name: list[str] | str | None,
    dataset_selector: str | None = None,
) -> tuple[int, int]:
    """Count how many of a child manifest's datasets have finished."""
    grouped = resolve_child_datasets(data_root, child_file, type_name=type_name)
    total = 0
    completed = 0
    for category, raw_csvs in grouped.items():
        csvs = select_datasets(
            raw_csvs,
            dataset_selector,
            scope_label=f"category '{category}'",
        )
        output_dir = output_root / category
        for csv in csvs:
            total += 1
            if dataset_complete(output_dir, csv.stem):
                completed += 1
    return completed, total


def mark_stuck_child_datasets_as_skipped(
    data_root: Path,
    output_root: Path,
    child_file: Path,
    type_name: list[str] | str | None,
    dataset_selector: str | None,
    logger: logging.Logger,
) -> list[str]:
    """Mark datasets that stopped making progress as skipped, so the sweep continues.

    A sweep spans many datasets and a single wedged one -- a hung backend, a
    pathological conversation -- would otherwise block all of them indefinitely.

    Skipping is recorded, never silent: each is appended to watchdog_stuck.jsonl
    and written into the dataset's stuck_history. That record is what keeps a
    skipped dataset from being mistaken for one that legitimately scored zero,
    and `should_reset_legacy_skipped_stage` makes a later resume redo it rather
    than trusting the marker as completion.

    Returns:
        The datasets marked skipped.
    """
    skipped = []
    grouped = resolve_child_datasets(data_root, child_file, type_name=type_name)
    stuck_log_path = output_root / "watchdog_stuck.jsonl"
    ensure_dir(stuck_log_path.parent)

    for category, raw_csvs in grouped.items():
        csvs = select_datasets(
            raw_csvs,
            dataset_selector,
            scope_label=f"category '{category}'",
        )
        output_dir = output_root / category
        for csv in csvs:
            name = csv.stem
            output_csv = output_dir / f"{name}.csv"
            if output_csv.exists():
                continue
            checkpoint_path = output_dir / f"checkpoint_{name}.json"
            if not checkpoint_path.exists():
                continue
            try:
                data = read_json_file(checkpoint_path, default={})
                stage = data.get("stage", "")
                if stage in TERMINAL_STAGES:
                    continue

                processed_ids = data.get("processed_session_ids", [])
                processed_count = len(processed_ids)
                total = data.get("total_sessions")
                in_sync = (processed_count % CHECKPOINT_EVERY_N == 0)

                prior_same_progress = count_same_progress_stuck_events(
                    stuck_log_path,
                    name,
                    processed_count,
                    category=category,
                )
                action = "skip_dataset" if prior_same_progress >= MAX_SAME_PROGRESS_RESUMES else "resume"

                log_entry = {
                    "dataset": name,
                    "category": category,
                    "stuck_at": datetime.now().isoformat(),
                    "previous_stage": stage,
                    "processed_sessions": processed_count,
                    "total_sessions": total,
                    "in_sync": in_sync,
                    "same_progress_retries": prior_same_progress,
                    "action": action,
                }
                append_jsonl(stuck_log_path, log_entry)

                stuck_entry = (
                    f"{datetime.now().strftime('%Y-%m-%dT%H:%M')} "
                    f"(processed {processed_count}/{total} sessions)"
                )
                try:
                    append_stuck_history_entry(output_dir, dataset=name, entry=stuck_entry)
                except Exception as exc:
                    logger.error("Failed to update progress.csv stuck_history for '%s': %s", name, exc)

                if action == "resume":
                    data["stage"] = "ingest_in_progress"
                    data["updated_at"] = datetime.now().isoformat() + "Z"
                    write_json_file(checkpoint_path, data)
                    logger.warning(
                        "Stuck child dataset '%s' in %s (was: %s) — (%d/%s), will resume next run (resume #%d).",
                        name, category, stage, processed_count, total, prior_same_progress + 1,
                    )
                else:
                    # Exhausted MAX_SAME_PROGRESS_RESUMES resumes at same point — mark as error
                    data["stage"] = "error"
                    data["updated_at"] = datetime.now().isoformat() + "Z"
                    write_json_file(checkpoint_path, data)
                    logger.error(
                        "Stuck child dataset '%s' in %s (was: %s) — same checkpoint (%d/%s) after %d resumes; "
                        "marking as error and skipping entire dataset.",
                        name, category, stage, processed_count, total, prior_same_progress + 1,
                    )

                skipped.append(f"{category}/{name}")
            except Exception as exc:
                logger.error("Failed to handle stuck child dataset '%s' in %s: %s", name, category, exc)
    return skipped


# ---------------------------------------------------------------------------
# Subprocess runner (with timeout / kill)
# ---------------------------------------------------------------------------

class RunResult(NamedTuple):
    """Outcome of one watchdog pass: what completed and what remains."""
    return_code: int
    timed_out: bool


def run_once(py: str, module: str, env: dict, logger: logging.Logger, timeout_sec: int) -> RunResult:
    """Perform one supervision pass: check progress, skip what is wedged.

    Called on a poll interval rather than continuously. Each pass is independent
    and reads state from disk, so the watchdog can be restarted mid-sweep
    without losing track of what it had already decided.
    """
    import signal

    cmd = [py, "-m", module]
    logger.info("Launching subprocess: %s", " ".join(cmd))
    logger.info("Timeout: %ds (%.1fh)", timeout_sec, timeout_sec / 3600)
    start = time.time()

    try:
        proc = subprocess.Popen(cmd, env=env)
    except Exception as exc:
        logger.error("Failed to start subprocess: %s", exc)
        return RunResult(return_code=-1, timed_out=False)

    poll_interval = 30  # check every 30 seconds
    while True:
        try:
            proc.wait(timeout=poll_interval)
            # Process ended on its own
            elapsed = time.time() - start
            logger.info("Subprocess finished in %.1fs with return code %d", elapsed, proc.returncode)
            if proc.returncode == -9:
                logger.warning(
                    "Subprocess exited with return code -9 before watchdog timeout; "
                    "treating this as an external SIGKILL/crash, not a watchdog timeout."
                )
            return RunResult(return_code=proc.returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            if elapsed >= timeout_sec:
                logger.warning(
                    "Subprocess hung for %.1fh (timeout=%.1fh) — killing...",
                    elapsed / 3600, timeout_sec / 3600,
                )
                try:
                    proc.send_signal(signal.SIGTERM)
                    time.sleep(5)
                    if proc.poll() is None:
                        proc.kill()
                        logger.warning("SIGTERM ignored, sent SIGKILL.")
                except Exception as kill_exc:
                    logger.error("Error killing subprocess: %s", kill_exc)
                proc.wait()
                return RunResult(return_code=-9, timed_out=True)
            # Still within timeout, keep waiting


# ---------------------------------------------------------------------------
# Rerun mode (in-process, no subprocess)
# ---------------------------------------------------------------------------

def _rerun_dataset_complete(output_dir: Path, dataset_name: str) -> bool:
    """A dataset is done when its output CSV exists."""
    return (output_dir / f"{dataset_name}.csv").exists()


def _resolve_rerun_targets(
    *,
    data_root: Path,
    artifact_root: Path,
    output_root_base: Path,
    type_names: list[str] | None,
) -> list[RerunTarget]:
    """Work out which datasets a rerun pass should cover, and why each qualifies.

    The reason is carried alongside the target so the rerun log states what
    prompted each one -- a dataset re-run for a stall and one re-run for an
    incomplete output are different situations.
    """
    from experiment.longmem.helpers.rerun_support import (
        retrieval_datasets_from_artifacts,
    )

    if type_names:
        return [
            RerunTarget(
                category=name,
                data_folder=append_type_subdir(data_root, name),
                artifact_dir=append_type_subdir(artifact_root, name),
                output_root=append_type_subdir(output_root_base, name),
            )
            for name in type_names
        ]

    category_dirs: list[Path] = []
    for subdir in sorted(path for path in artifact_root.iterdir() if path.is_dir()):
        try:
            if retrieval_datasets_from_artifacts(subdir, None):
                category_dirs.append(subdir)
        except Exception:
            continue

    if category_dirs:
        targets: list[RerunTarget] = []
        for category_dir in category_dirs:
            category = category_dir.name
            data_folder = data_root / category
            targets.append(
                RerunTarget(
                    category=category,
                    data_folder=data_folder if data_folder.exists() else data_root,
                    artifact_dir=category_dir,
                    output_root=output_root_base / category,
                )
            )
        return targets

    return [
        RerunTarget(
            category=None,
            data_folder=data_root,
            artifact_dir=artifact_root,
            output_root=output_root_base,
        )
    ]


def _run_rerun_mode(
    args,
    artifact_dir: Path,
    output_root: Path,
    logger: logging.Logger,
    status_path: Path,
    *,
    data_folder: Path | None = None,
) -> int:
    """Run the watchdog in rerun mode: re-do the datasets that need it, then stop."""
    from experiment.longmem.helpers.rerun_support import (
        cleanup_retrieval_loggers,
        rerun_accuracy,
        retrieval_datasets_from_artifacts,
    )
    from experiment.longmem.pipeline.aggregate import (
        update_all_answers_csv,
        update_progress_rows,
    )
    from experiment.longmem.pipeline.rerun import LongMemRerun

    if not artifact_dir.exists():
        logger.error("Artifact dir not found: %s", artifact_dir)
        return 2

    ensure_dir(output_root)
    if data_folder is None and args.data_folder:
        single_type = args.type[0] if args.type and len(args.type) == 1 else None
        data_folder = append_type_subdir(Path(args.data_folder), single_type).resolve()
    selected_stages = resolve_stages(
        args.stages,
        no_judge=args.no_judge,
        artifact_dir=str(artifact_dir),
    )
    dataset_selector = args.dataset_id or os.environ.get("MDQA_DATASET_ID", "").strip() or None

    logger.info("=" * 60)
    logger.info("Rerun mode (retrieval-only)")
    logger.info("artifact_dir : %s", artifact_dir)
    logger.info("output_root  : %s", output_root)
    logger.info("data_folder  : %s", data_folder or "(none)")
    logger.info("no_judge     : %s", args.no_judge)
    logger.info("num          : %s", args.num if args.num is not None else "all")
    logger.info("dataset_id   : %s", dataset_selector or "(all)")
    logger.info("stages       : %s", ",".join(selected_stages))
    logger.info("=" * 60)

    status = {
        "session_start": datetime.now().isoformat(),
        "state": "starting",
        "completed_datasets": 0,
        "total_datasets": 0,
    }
    write_status_file(status_path, status)

    # Discover all datasets from artifact_dir
    all_datasets = retrieval_datasets_from_artifacts(artifact_dir, args.datasets)
    all_datasets = select_dataset_names(
        all_datasets,
        dataset_selector,
        scope_label=f"artifact_dir '{artifact_dir.name}'",
    )
    if args.num is not None:
        all_datasets = all_datasets[: args.num]

    total = len(all_datasets)
    logger.info("Datasets found: %d", total)
    if not all_datasets:
        logger.info("Nothing to do.")
        status["state"] = "done"
        write_status_file(status_path, status)
        return 0

    # Skip already-completed datasets for the default full rerun flow,
    # and for qa_eval-only runs (--stage qa_eval [--no-judge]) -- so resuming after
    # a crash does not have to answer the whole batch again (2026-07-16).
    # judge-only flows are not skipped.
    if tuple(selected_stages) == tuple(DEFAULT_STAGES) or tuple(selected_stages) == ("qa_eval",):
        to_run = [name for name in all_datasets if not _rerun_dataset_complete(output_root, name)]
        already_done = total - len(to_run)
        if already_done:
            logger.info("Already complete: %d / %d — skipping.", already_done, total)
    else:
        to_run = list(all_datasets)
        already_done = 0

    status.update({"total_datasets": total, "completed_datasets": already_done, "state": "running"})
    write_status_file(status_path, status)

    results: list[dict] = []
    with LongMemRerun.from_env() as runner:
        for index, dataset_name in enumerate(to_run, 1):
            logger.info("[%d/%d] %s", index, len(to_run), dataset_name)
            dataset_log_dir = output_root / f"logs_{dataset_name}"
            ensure_dir(dataset_log_dir)
            try:
                result = runner.rerun_dataset(
                    dataset_name=dataset_name,
                    output_dir=output_root,
                    data_folder=data_folder,
                    log_dir=dataset_log_dir,
                    artifact_dir=artifact_dir,
                    no_judge=args.no_judge,
                    stages=set(selected_stages),
                )
                results.append(result)
                logger.info("%s | correctness=%s", dataset_name, result["correctness"])
            except Exception as exc:
                logger.exception("Dataset %s failed", dataset_name)
                error_result = {"dataset": dataset_name, "error": str(exc)}
                results.append(error_result)
                try:
                    runner.graph.clear_all()
                except Exception:
                    pass
            finally:
                cleanup_retrieval_loggers(dataset_log_dir)

            completed = already_done + index
            status.update({"completed_datasets": completed})
            write_status_file(status_path, status)

    success = [row for row in results if "error" not in row]
    errors = [row for row in results if "error" in row]

    if success:
        update_progress_rows(output_root, success, filename="progress.csv")
        update_all_answers_csv(output_root, success)
    if errors:
        logger.warning("Errors: %d", len(errors))
        for row in errors:
            logger.warning("  %s: %s", row["dataset"], row["error"])

    correct, judged = rerun_accuracy(success)
    if judged:
        logger.info("Accuracy: %d/%d = %.1f%%", correct, judged, correct / judged * 100)

    status["state"] = "done"
    write_status_file(status_path, status)
    return 0


# ---------------------------------------------------------------------------
# Main watchdog loop
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LongMem watchdog — batch or retrieval-rerun mode")
    # Shared arg groups
    add_data_args(parser)
    add_child_args(parser)
    add_run_args(parser)
    add_rerun_args(parser)
    # Watchdog-specific args
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sleep", type=int, default=30, help="Base seconds between restarts")
    parser.add_argument("--max-restarts", type=int, default=10, help="0 = infinite")
    parser.add_argument("--log-dir", default=None, help="Directory for watchdog logs")
    parser.add_argument(
        "--io-cooldown", type=int, default=10,
        help="Extra seconds to pause after a crash to let I/O settle (0 to disable)",
    )
    parser.add_argument(
        "--timeout", type=int, default=604800,
        help="Seconds before killing a hung subprocess (default: 604800 = 7 days)",
    )
    args = parser.parse_args(argv)
    args.raw_argv = list(argv) if argv is not None else sys.argv[1:]
    artifact_root = Path(args.artifact_dir).resolve() if args.artifact_dir else None
    selected_stages = resolve_stages(
        args.stages,
        no_judge=args.no_judge,
        artifact_dir=str(artifact_root) if artifact_root is not None else None,
    )
    dataset_selector = args.dataset_id or os.environ.get("MDQA_DATASET_ID", "").strip() or None

    rerun_mode = artifact_root is not None

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    data_folder_base = args.data_folder or "./experiment/longmem/script_data/"
    data_root = Path(data_folder_base).resolve()
    child_file = Path(args.child_file).resolve()
    single_type = args.type[0] if args.type and len(args.type) == 1 else None
    data_folder = append_type_subdir(data_root, single_type).resolve()
    output_root_base = Path(args.output_root) if args.output_root else default_output_root(run_tag)
    output_root = append_type_subdir(output_root_base, single_type).resolve() if rerun_mode else output_root_base.resolve()
    log_dir = Path(args.log_dir).resolve() if args.log_dir else default_log_dir(run_tag).resolve()

    logger = setup_logger(log_dir)
    status_path = log_dir / "watchdog_status.json"
    metadata_root = output_root if rerun_mode else output_root_base.resolve()
    _write_watchdog_metadata(
        metadata_root / "run_metadata.json",
        args=args,
        run_tag=run_tag,
        data_root=data_root,
        data_folder=data_folder,
        child_file=child_file,
        output_root=metadata_root,
        artifact_dir=artifact_root,
        log_dir=log_dir,
        batch_module=BATCH_MODULE,
        selected_stages=selected_stages,
        dataset_selector=dataset_selector,
        rerun_mode=rerun_mode,
    )

    # ── Rerun mode (--artifact-dir set): run retrieval in-process ────────────
    if rerun_mode:
        rerun_targets = _resolve_rerun_targets(
            data_root=data_root,
            artifact_root=artifact_root,
            output_root_base=output_root_base.resolve(),
            type_names=args.type,
        )
        if len(rerun_targets) > 1:
            logger.info("Rerun mode categories: %s", ", ".join(target.artifact_dir.name for target in rerun_targets))
        exit_code = 0
        for target in rerun_targets:
            result = _run_rerun_mode(
                args,
                target.artifact_dir.resolve(),
                target.output_root.resolve(),
                logger,
                status_path,
                data_folder=target.data_folder.resolve() if target.data_folder is not None else None,
            )
            if result != 0:
                exit_code = result
        return exit_code

    # ── Batch mode: existing subprocess watchdog loop ────────────────────────

    # Validate paths
    if args.child:
        if not data_root.exists():
            logger.error("Data root not found: %s", data_root)
            return 2
        if not child_file.exists():
            logger.error("Child manifest not found: %s", child_file)
            return 2
        try:
            child_groups = resolve_child_datasets(data_root, child_file, type_name=args.type)
            if dataset_selector:
                child_groups = {
                    category: select_datasets(
                        csv_paths,
                        dataset_selector,
                        scope_label=f"category '{category}'",
                    )
                    for category, csv_paths in child_groups.items()
                }
        except Exception as exc:
            logger.error("Failed to resolve child datasets: %s", exc)
            return 2
    else:
        if not data_folder.exists():
            logger.error("Data folder not found: %s", data_folder)
            return 2
        if dataset_selector:
            try:
                for folder in list_run_targets(data_folder):
                    select_datasets(
                        discover_csvs(folder, args.file_pattern),
                        dataset_selector,
                        scope_label=f"folder '{folder.name}'",
                    )
            except Exception as exc:
                logger.error("Failed to resolve dataset selector: %s", exc)
                return 2
    logger.info("data_root    : %s", data_root)
    logger.info("child        : %s", args.child)
    logger.info("child_file   : %s", child_file if args.child else "(disabled)")
    logger.info("type         : %s", ", ".join(args.type) if args.type else "(all)")
    logger.info("data_folder  : %s", data_folder if not args.child else "(manifest-driven)")
    logger.info("run_tag      : %s", run_tag)
    logger.info("output_root  : %s", output_root)
    logger.info("batch_module : %s", BATCH_MODULE)
    logger.info("no_judge     : %s", args.no_judge)
    logger.info("stages       : %s", ",".join(selected_stages))
    logger.info("dataset_id   : %s", dataset_selector or "(all)")
    logger.info("num          : %s", args.num if args.num is not None else "all")
    logger.info("max_restarts : %s", args.max_restarts if args.max_restarts > 0 else "infinite")
    logger.info("base_sleep   : %ds", args.sleep)
    logger.info("io_cooldown  : %ds", args.io_cooldown)
    logger.info("timeout      : %ds (%.1fh)", args.timeout, args.timeout / 3600)
    if args.child:
        logger.info("child groups : %s", {key: len(value) for key, value in child_groups.items()})

    restarts = 0
    backoff = args.sleep
    session_start = datetime.now().isoformat()

    status = {
        "session_start": session_start,
        "restarts": 0,
        "last_return_code": None,
        "state": "starting",
        "completed_datasets": 0,
        "total_datasets": 0,
    }
    write_status_file(status_path, status)

    if tuple(selected_stages) != tuple(DEFAULT_STAGES):
        logger.info("Non-default stage selection detected; watchdog will run a single batch pass without restart loop.")
        env = os.environ.copy()
        env["MDQA_DATA_FOLDER"] = str(data_folder)
        env["MDQA_OUTPUT_ROOT"] = str(resolve_batch_output_root(data_folder, output_root))
        env["MDQA_FILE_PATTERN"] = args.file_pattern
        env["MDQA_CHILD"] = "1" if args.child else "0"
        env["MDQA_CHILD_FILE"] = str(child_file)
        env["MDQA_CHILD_TYPE"] = (",".join(args.type) if args.type else "") if args.child else ""
        env["MDQA_RUN_TAG"] = run_tag
        env["MDQA_RUN_JUDGE"] = "0" if args.no_judge else "1"
        env["MDQA_STAGES"] = ",".join(selected_stages)
        env["MDQA_DATASET_ID"] = dataset_selector or ""
        if args.num is not None:
            env["MDQA_NUM_DATASETS"] = str(args.num)
        if args.child:
            env["MDQA_DATA_FOLDER"] = str(data_root)
            env["MDQA_OUTPUT_ROOT"] = str(output_root)

        status["state"] = "running"
        write_status_file(status_path, status)
        result = run_once(args.python, BATCH_MODULE, env, logger, args.timeout)
        status["last_return_code"] = result.return_code
        status["state"] = "done" if result.return_code == 0 else "failed"
        write_status_file(status_path, status)
        return 0 if result.return_code == 0 else 1

    while True:
        try:
            # ── Check completion ──────────────────────────────────────────
            if args.child:
                completed, total = count_completion_child(
                    data_root,
                    output_root,
                    child_file,
                    args.type,
                    dataset_selector,
                )
            else:
                completed, total = count_completion_selected(
                    data_folder,
                    output_root,
                    args.file_pattern,
                    dataset_selector,
                )
            logger.info("Progress: %d / %d datasets complete", completed, total)
            status.update({"completed_datasets": completed, "total_datasets": total})

            if total > 0 and completed >= total:
                logger.info("All datasets complete. Exiting.")
                status["state"] = "done"
                write_status_file(status_path, status)
                return 0

            # ── Build env ─────────────────────────────────────────────────
            env = os.environ.copy()
            env["MDQA_DATA_FOLDER"] = str(data_folder)
            env["MDQA_OUTPUT_ROOT"] = str(resolve_batch_output_root(data_folder, output_root))
            env["MDQA_FILE_PATTERN"] = args.file_pattern
            env["MDQA_CHILD"] = "1" if args.child else "0"
            env["MDQA_CHILD_FILE"] = str(child_file)
            env["MDQA_CHILD_TYPE"] = (",".join(args.type) if args.type else "") if args.child else ""
            env["MDQA_RUN_TAG"] = run_tag
            env["MDQA_RUN_JUDGE"] = "0" if args.no_judge else "1"
            env["MDQA_STAGES"] = ",".join(selected_stages)
            env["MDQA_DATASET_ID"] = dataset_selector or ""
            if args.num is not None:
                env["MDQA_NUM_DATASETS"] = str(args.num)
            if args.child:
                env["MDQA_DATA_FOLDER"] = str(data_root)
                env["MDQA_OUTPUT_ROOT"] = str(output_root)

            # ── Launch ───────────────────────────────────────────────────
            status["state"] = "running"
            write_status_file(status_path, status)

            result = run_once(args.python, BATCH_MODULE, env, logger, args.timeout)
            code = result.return_code
            status["last_return_code"] = code

            # ── Post-run completion check ─────────────────────────────────
            if args.child:
                completed, total = count_completion_child(
                    data_root,
                    output_root,
                    child_file,
                    args.type,
                    dataset_selector,
                )
            else:
                completed, total = count_completion_selected(
                    data_folder,
                    output_root,
                    args.file_pattern,
                    dataset_selector,
                )
            status.update({"completed_datasets": completed, "total_datasets": total})
            logger.info("Progress after run: %d / %d datasets complete", completed, total)

            if total > 0 and completed >= total:
                logger.info("All datasets complete after run. Exiting.")
                status["state"] = "done"
                write_status_file(status_path, status)
                return 0

            # ── Handle crash / non-zero exit ──────────────────────────────
            if result.timed_out:
                logger.warning("Run was killed due to timeout (%.1fh).", args.timeout / 3600)
                if args.child:
                    skipped = mark_stuck_child_datasets_as_skipped(
                        data_root, output_root, child_file, args.type, dataset_selector, logger
                    )
                else:
                    skipped = mark_stuck_datasets_as_skipped(
                        data_folder, output_root, args.file_pattern, dataset_selector, logger
                    )
                if skipped:
                    status["skipped_datasets"] = status.get("skipped_datasets", []) + skipped
                    logger.warning("Skipped datasets: %s", skipped)
            elif code == -9:
                logger.warning(
                    "Run exited with code -9 before watchdog timeout; skipping watchdog timeout recovery."
                )
            if code != 0:
                logger.warning("Run exited with code %d (crash or error).", code)
                if args.io_cooldown > 0:
                    logger.info(
                        "I/O cooldown: sleeping %ds to let filesystem settle...",
                        args.io_cooldown,
                    )
                    time.sleep(args.io_cooldown)
            else:
                # Exited cleanly but datasets still incomplete — reset backoff
                backoff = args.sleep
                logger.info("Run exited cleanly but datasets still incomplete.")

            # ── Refresh system before restart ────────────────────────────
            logger.info("Running refresh_system to clear stuck state...")
            status["state"] = "refreshing"
            write_status_file(status_path, status)
            try:
                from tools.refresh_system import refresh_system
                refresh_system()
                logger.info("refresh_system completed successfully.")
            except Exception as ref_exc:
                logger.error("refresh_system failed: %s", ref_exc)

            # ── Restart bookkeeping ───────────────────────────────────────
            restarts += 1
            status["restarts"] = restarts
            status["state"] = "waiting_restart"
            write_status_file(status_path, status)

            if args.max_restarts > 0 and restarts > args.max_restarts:
                logger.error("Max restarts (%d) reached. Giving up.", args.max_restarts)
                status["state"] = "failed_max_restarts"
                write_status_file(status_path, status)
                return 1

            logger.info("Restart #%d — sleeping %ds before next attempt...", restarts, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)  # exponential backoff, cap at 5 min

        except KeyboardInterrupt:
            logger.info("Watchdog interrupted by user (KeyboardInterrupt).")
            status["state"] = "interrupted"
            write_status_file(status_path, status)
            return 130

        except Exception:
            # Watchdog itself crashed — log and try to carry on
            logger.exception("Unexpected watchdog error")
            status["state"] = "watchdog_error"
            write_status_file(status_path, status)
            logger.info("Sleeping 60s after watchdog error before retrying...")
            time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
