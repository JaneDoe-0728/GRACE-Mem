"""Compatibility wrapper for :mod:`experiment.locomo.analysis.gold_recall`."""

from experiment.locomo.analysis import gold_recall as _implementation
from experiment.locomo.analysis.gold_recall import *  # noqa: F401,F403
from experiment.locomo.analysis.gold_recall import main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
