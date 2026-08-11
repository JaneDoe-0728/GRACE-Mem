from importlib import import_module
from typing import Any

_MOVED_MODULES = {
    "aggregate": "experiment.locomo.analysis.aggregate",
    "decision": "experiment.locomo.pipeline.decision",
    "pipeline": "experiment.locomo.pipeline.runner",
    "run_filter_sweep": "experiment.locomo.analysis.filter_sweep",
    "snapshot": "experiment.locomo.artifacts.snapshot",
    "stage_adapter": "experiment.locomo.pipeline.stage_adapter",
    "summary": "experiment.locomo.analysis.summary",
    "vote_merge": "experiment.locomo.analysis.vote_merge",
    "workers": "experiment.locomo.pipeline.worker",
}

__all__ = ["stages", *_MOVED_MODULES]


def __getattr__(name: str) -> Any:
    if name in _MOVED_MODULES:
        return import_module(_MOVED_MODULES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
