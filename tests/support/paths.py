"""Where the repository root is, computed once.

Every test that reaches for a file in the repo used to derive the root itself,
each with its own `parents[N]`. That count is a function of how deep the test
file happens to sit, so grouping the suite into directories silently broke all
of them at once -- the same failure mode `test_ingestion_pipeline` already
records a comment about. Deriving it here means a test can move between groups
without its paths caring.
"""
from __future__ import annotations

from pathlib import Path

#: The repo root: tests/support/paths.py -> tests/support -> tests -> root.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Golden files for the characterization tests, one directory per module.
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
