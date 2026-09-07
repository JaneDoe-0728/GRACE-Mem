"""Canonical project paths shared by experiment entrypoints."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "experiment"
