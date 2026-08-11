from importlib import import_module
from typing import Any

_MOVED_MODULES = {
    "aggregate": "experiment.longmem.pipeline.aggregate",
    "decision": "experiment.longmem.pipeline.decision",
    "processor": "experiment.longmem.pipeline.processor",
    "rebuild_split_summaries": "experiment.longmem.tools.rebuild_split_summaries",
    "rerun": "experiment.longmem.pipeline.rerun",
    "rerun_split_experiments": "experiment.longmem.tools.rerun_split_experiments",
    "run_batch": "experiment.longmem.pipeline.batch",
    "snapshot": "experiment.longmem.artifacts.snapshot",
    "stage_adapter": "experiment.longmem.pipeline.stage_adapter",
    "watchdog": "experiment.longmem.pipeline.watchdog",
}

__all__ = [*_MOVED_MODULES]


def __getattr__(name: str) -> Any:
    if name in _MOVED_MODULES:
        return import_module(_MOVED_MODULES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
