"""Benchmark orchestration and reproducibility adapters."""

from importlib import import_module
from typing import Any

_MOVED_MODULES = {
    "judge": "experiment.common.evaluation.judge",
    "oracle": "experiment.common.evaluation.oracle",
    "score": "experiment.common.evaluation.score",
    "reproducibility": "experiment.common.reproducibility",
    "run_metadata": "experiment.common.run_metadata",
}

__all__ = [*_MOVED_MODULES]


def __getattr__(name: str) -> Any:
    if name in _MOVED_MODULES:
        return import_module(_MOVED_MODULES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
