"""Compatibility wrapper for :mod:`experiment.locomo.analysis.flips`."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.locomo.analysis import flips as _implementation  # noqa: E402
from experiment.locomo.analysis.flips import *  # noqa: F401,F403,E402
from experiment.locomo.analysis.flips import main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
