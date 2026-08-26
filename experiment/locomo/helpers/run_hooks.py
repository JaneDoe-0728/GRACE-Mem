"""Run-level path and backend-refresh helpers for LoCoMo samples."""

from __future__ import annotations

import subprocess
import sys
import time

from experiment.locomo.models import RunConfig, WorkerPaths
from experiment.locomo.utils.io import ensure_dir
from experiment.locomo.utils.log import log_event


def _worker_paths_for_sample(config: RunConfig, sample_index: int) -> WorkerPaths:
    """Compute the standard per-sample LoCoMo output paths."""
    run_root = config.run_root
    sample_dir = ensure_dir(run_root / f"sample_{sample_index}")
    eval_stem = f"sample{sample_index}_eval_{config.run_tag}"
    return WorkerPaths(
        sample_index=sample_index,
        sample_dir=sample_dir,
        eval_csv=sample_dir / f"{eval_stem}.csv",
        judge_csv=sample_dir / f"{eval_stem}_judge.csv",
        stats_json=sample_dir / "correctness_summary.json",
    )


def _refresh_system(*, sleep_seconds: float) -> None:
    """Restart the graph backend between samples and wait for it to settle.

    The sleep is not padding: the backend accepts connections before it can
    serve them, and a query issued in that window fails in a way that looks like
    a retrieval bug rather than a startup race.
    """
    log_event("REFRESH", "Cleaning system for next sample")
    refresh_cmd = [
        sys.executable, "-c",
        "import sys; sys.path.append('.');"
        "from tools.refresh_system import refresh_system; refresh_system()"
    ]
    subprocess.run(refresh_cmd)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
