"""Experiment configuration adapter for the core reproducibility runtime."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from grace_mem.runtime import reproducibility as _runtime


ReproducibilityConfig = _runtime.ReproducibilityConfig
DEFAULT_SEED = _runtime.DEFAULT_SEED
DEFAULT_DETERMINISTIC = _runtime.DEFAULT_DETERMINISTIC
prepare_reproducibility_env = _runtime.prepare_reproducibility_env
set_global_seed = _runtime.set_global_seed
build_dataloader_seed_components = _runtime.build_dataloader_seed_components


def _load_experiment_reproducibility_params() -> dict[str, Any]:
    try:
        module = importlib.import_module("experiment.experiment_config")
    except Exception:
        return {}
    params = getattr(module, "REPRODUCIBILITY_PARAMS", None)
    return dict(params) if isinstance(params, dict) else {}


def _experiment_defaults() -> dict[str, Any]:
    defaults = _load_experiment_reproducibility_params()
    _runtime.configure_reproducibility_defaults(defaults)
    return defaults


def resolve_reproducibility_config(
    *,
    seed: int | None = None,
    deterministic: bool | None = None,
) -> ReproducibilityConfig:
    return _runtime.resolve_reproducibility_config(
        seed=seed,
        deterministic=deterministic,
        defaults=_experiment_defaults(),
    )


def activate_reproducibility(
    *,
    seed: int | None = None,
    deterministic: bool | None = None,
    log_prefix: str | None = None,
) -> ReproducibilityConfig:
    return _runtime.activate_reproducibility(
        seed=seed,
        deterministic=deterministic,
        defaults=_experiment_defaults(),
        log_prefix=log_prefix,
    )


def get_runtime_reproducibility() -> ReproducibilityConfig:
    _experiment_defaults()
    return _runtime.get_runtime_reproducibility()


def current_reproducibility_state(
    cfg: ReproducibilityConfig | None = None,
) -> dict[str, Any]:
    return _runtime.current_reproducibility_state(cfg or get_runtime_reproducibility())


def attach_reproducibility_metadata(
    payload: dict[str, Any],
    *,
    cfg: ReproducibilityConfig | None = None,
) -> dict[str, Any]:
    return _runtime.attach_reproducibility_metadata(
        payload,
        config=cfg or get_runtime_reproducibility(),
    )


def write_reproducibility_file(
    directory: str | Path,
    *,
    filename: str = "reproducibility.json",
) -> Path:
    _experiment_defaults()
    return _runtime.write_reproducibility_file(directory, filename=filename)


_experiment_defaults()
