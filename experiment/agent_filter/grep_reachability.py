"""Compatibility wrapper for the LongMem Agent Filter reachability analysis."""

from experiment.longmem.analysis import agent_filter_reachability as _implementation
from experiment.longmem.analysis.agent_filter_reachability import *  # noqa: F401,F403
from experiment.longmem.analysis.agent_filter_reachability import main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
