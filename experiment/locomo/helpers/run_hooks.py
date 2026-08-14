"""Run-level hooks: what happens around each sample, not inside it.

The runner owns the loop; these own the bookkeeping at its edges -- collecting
a finished worker's outputs, updating the run summary, syncing logs, and
refreshing the backend between samples.

They are separated from the runner so that per-dataset differences live in
`DatasetStrategy` flags consulted here, rather than as branches in the loop
itself.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any

from experiment.locomo.models import DatasetStrategy, RunRuntime, SamplePlan, WorkerPaths
from experiment.locomo.utils.io import ensure_dir, sync_logs, write_summary_map
from experiment.locomo.utils.log import log_event


def _worker_paths_for_sample(
    runtime: RunRuntime,
    sample_index: int,
    strategy: DatasetStrategy,
) -> WorkerPaths:
    run_root = runtime.config.run_root
    sample_dir = ensure_dir(run_root / f"sample_{sample_index}")
    if not strategy.uses_run_dirs:
        eval_stem = f"sample{sample_index}_eval_{runtime.config.run_tag}"
        return WorkerPaths(
            sample_index=sample_index,
            sample_dir=sample_dir,
            eval_csv=sample_dir / f"{eval_stem}.csv",
            judge_csv=sample_dir / f"{eval_stem}_judge.csv",
            stats_json=sample_dir / "correctness_summary.json",
        )

    run_eval_dir = ensure_dir(run_root / "eval")
    run_judge_dir = ensure_dir(run_root / "judge") if not runtime.config.no_judge else None
    return WorkerPaths(
        sample_index=sample_index,
        sample_dir=None,
        eval_csv=run_eval_dir / f"{sample_index}.csv",
        judge_csv=(
            run_judge_dir / f"{sample_index}.csv"
            if run_judge_dir is not None
            else run_root / f"{sample_index}.judge.skip.csv"
        ),
        stats_json=run_root / f".sample_{sample_index:06d}_correctness_summary.json",
    )


def _record_sample_outputs(
    runtime: RunRuntime,
    args: Any,
    sample_paths: WorkerPaths,
    strategy: DatasetStrategy,
) -> None:
    if not strategy.uses_run_dirs:
        return

    if not args.no_judge and sample_paths.stats_json.exists():
        runtime.per_sample_stats[f"sample_{sample_paths.sample_index}"] = json.loads(
            sample_paths.stats_json.read_text(encoding="utf-8")
        )
        write_summary_map(runtime.run_summary_json, runtime.per_sample_stats)
    if sample_paths.stats_json.exists():
        sample_paths.stats_json.unlink()


def _after_worker(runtime: RunRuntime, strategy: DatasetStrategy) -> None:
    if strategy.sync_logs_after_worker:
        sync_logs(runtime.config.run_root)


def _log_success(
    runtime: RunRuntime,
    args: Any,
    plan: SamplePlan,
    strategy: DatasetStrategy,
) -> None:
    if not strategy.uses_run_dirs:
        return
    if args.no_judge:
        log_event("SAVE", "Wrote eval artifact", path=plan.worker_paths.eval_csv.relative_to(runtime.config.run_root))
        return
    log_event(
        "SAVE",
        "Wrote eval and judge artifacts",
        eval=plan.worker_paths.eval_csv.relative_to(runtime.config.run_root),
        judge=plan.worker_paths.judge_csv.relative_to(runtime.config.run_root),
    )


def _refresh_system(*, sleep_seconds: float) -> None:
    log_event("REFRESH", "Cleaning system for next sample")
    refresh_cmd = [
        sys.executable, "-c",
        "import sys; sys.path.append('.');"
        "from tools.refresh_system import refresh_system; refresh_system()"
    ]
    subprocess.run(refresh_cmd)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
