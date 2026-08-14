"""Process-wide deterministic runtime state with no experiment-layer dependency."""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_DETERMINISTIC = True

_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_NCCL_ENV_DEFAULTS = {
    "NCCL_ASYNC_ERROR_HANDLING": "1",
    "NCCL_BLOCKING_WAIT": "1",
}

_RUNTIME_CONFIG: ReproducibilityConfig | None = None
_CONFIG_DEFAULTS: dict[str, Any] = {}


@dataclass(frozen=True)
class ReproducibilityConfig:
    """The seed and determinism settings a run was executed under.

    Recorded into run metadata, so a result can be reproduced from what is
    stored rather than from memory of how it was invoked.
    """
    seed: int = DEFAULT_SEED
    deterministic: bool = DEFAULT_DETERMINISTIC


def configure_reproducibility_defaults(defaults: Mapping[str, Any] | None) -> None:
    """Register outer-layer defaults without importing that layer from core code."""
    global _CONFIG_DEFAULTS
    _CONFIG_DEFAULTS = dict(defaults or {})


def _env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return value.strip().lower() not in {"0", "false", "no", "off"}


def resolve_reproducibility_config(
    *,
    seed: int | None = None,
    deterministic: bool | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> ReproducibilityConfig:
    """Resolve explicit values, inherited process state, then registered defaults."""
    resolved_defaults = {**_CONFIG_DEFAULTS, **dict(defaults or {})}
    env_seed = os.environ.get("EXPERIMENT_SEED")
    env_deterministic = _env_bool("EXPERIMENT_DETERMINISTIC")
    return ReproducibilityConfig(
        seed=int(
            seed
            if seed is not None
            else env_seed
            if env_seed not in (None, "")
            else resolved_defaults.get("seed", DEFAULT_SEED)
        ),
        deterministic=bool(
            deterministic
            if deterministic is not None
            else env_deterministic
            if env_deterministic is not None
            else resolved_defaults.get("deterministic", DEFAULT_DETERMINISTIC)
        ),
    )


def prepare_reproducibility_env(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Set the environment variables that must be in place before torch loads.

    CUBLAS_WORKSPACE_CONFIG is the reason this is separate from `set_global_seed`:
    cuBLAS reads it at initialization, so setting it after torch has touched the
    GPU has no effect and `use_deterministic_algorithms` then fails at the first
    matmul. Call this before importing torch.

    The NCCL defaults pin collective-operation ordering, which is otherwise a
    source of run-to-run variation on multi-GPU machines.

    Returns:
        The resulting reproducibility state, for the run metadata.
    """
    os.environ["EXPERIMENT_SEED"] = str(int(seed))
    os.environ["EXPERIMENT_DETERMINISTIC"] = "1" if deterministic else "0"
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
        for key, value in _NCCL_ENV_DEFAULTS.items():
            os.environ[key] = value
    return current_reproducibility_state(
        ReproducibilityConfig(seed=int(seed), deterministic=bool(deterministic))
    )


def set_global_seed(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed every RNG in the process and switch torch to deterministic kernels.

    Four independent sources of randomness have to be seeded -- Python's
    `random`, NumPy, torch CPU, and torch CUDA -- because each has its own
    generator and missing one leaves a channel through which a run can diverge.

    Failures raise rather than warn. A run that believes it is seeded but is not
    produces results that cannot be reproduced and gives no indication of it,
    which is worse than failing at startup.

    Args:
        deterministic: Also force deterministic algorithm selection. Costs
            throughput -- some fast kernels have no deterministic variant, and
            torch raises instead of silently substituting one.
    """
    prepare_reproducibility_env(seed, deterministic)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception as exc:
        raise RuntimeError(f"Failed to seed NumPy with seed={seed}: {exc}") from exc

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"Failed to import PyTorch while applying seed={seed}: {exc}") from exc

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("highest")
        else:
            torch.backends.cudnn.deterministic = False
    except Exception as exc:
        raise RuntimeError(
            "Failed to enable deterministic PyTorch execution. "
            "Fix unsupported downstream operations instead of silently falling back."
        ) from exc

    return current_reproducibility_state(
        ReproducibilityConfig(seed=int(seed), deterministic=bool(deterministic))
    )


def activate_reproducibility(
    *,
    seed: int | None = None,
    deterministic: bool | None = None,
    defaults: Mapping[str, Any] | None = None,
    log_prefix: str | None = None,
) -> ReproducibilityConfig:
    """Apply the reproducibility configuration for this run.

    The entry point every runner calls at startup, before any model is loaded.
    """
    global _RUNTIME_CONFIG
    config = resolve_reproducibility_config(
        seed=seed,
        deterministic=deterministic,
        defaults=defaults,
    )
    set_global_seed(config.seed, config.deterministic)
    _RUNTIME_CONFIG = config
    if log_prefix:
        logger.info(
            "%s %s",
            log_prefix,
            json.dumps(current_reproducibility_state(config), ensure_ascii=False),
        )
    return config


def get_runtime_reproducibility() -> ReproducibilityConfig:
    global _RUNTIME_CONFIG
    if _RUNTIME_CONFIG is None:
        _RUNTIME_CONFIG = resolve_reproducibility_config()
    return _RUNTIME_CONFIG


def current_reproducibility_state(
    config: ReproducibilityConfig | None = None,
) -> dict[str, Any]:
    """Snapshot the seed and determinism settings actually in effect.

    Read from the environment rather than from the config object, so what is
    recorded is what the process will really use -- including an override set
    externally that the config never saw.
    """
    resolved = config or get_runtime_reproducibility()
    payload = asdict(resolved)
    payload["env"] = {
        "EXPERIMENT_SEED": os.environ.get("EXPERIMENT_SEED"),
        "EXPERIMENT_DETERMINISTIC": os.environ.get("EXPERIMENT_DETERMINISTIC"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        **{key: os.environ.get(key) for key in _NCCL_ENV_DEFAULTS},
    }
    return payload


def attach_reproducibility_metadata(
    payload: dict[str, Any],
    *,
    config: ReproducibilityConfig | None = None,
) -> dict[str, Any]:
    """Stamp a metadata payload with the run's reproducibility state.

    Called on every artifact that gets written, so any result file carries the
    seed that produced it.
    """
    output = dict(payload)
    output["reproducibility"] = current_reproducibility_state(config)
    return output


def write_reproducibility_file(
    directory: str | Path,
    *,
    filename: str = "reproducibility.json",
) -> Path:
    """Write the reproducibility state as a standalone file in the run directory.

    Duplicated from the run metadata on purpose: it is the one file someone
    reads when asking "what seed was this?", and burying it inside a larger
    payload makes that question harder than it should be.
    """
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    path.write_text(
        json.dumps(current_reproducibility_state(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def build_dataloader_seed_components(
    seed: int,
) -> tuple[Callable[[int], None], Any]:
    """Derive the per-worker seed components for a torch DataLoader.

    DataLoader workers are separate processes that inherit no RNG state, so
    without an explicit per-worker seed each one draws from an unseeded
    generator and the loading order stops being reproducible.
    """
    import numpy as np
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        """DataLoader `worker_init_fn`: seed this worker deterministically.

        Derives from the base seed and the worker id, so workers are seeded
        differently from each other -- identical seeds would make them draw the same
        augmentations -- but identically across runs.
        """
        worker_seed = seed + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)

    return seed_worker, generator
