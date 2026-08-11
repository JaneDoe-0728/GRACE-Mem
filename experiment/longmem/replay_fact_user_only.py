"""Compatibility wrapper for user-only LongMem fact replay analysis."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.longmem.analysis import fact_replay as _implementation  # noqa: E402
from experiment.longmem.analysis.fact_replay import *  # noqa: F401,F403,E402


def __getattr__(name: str):
    return getattr(_implementation, name)


def main(argv: list[str] | None = None) -> None:
    _implementation.main(argv, default_source_roles="user")


if __name__ == "__main__":
    main()
