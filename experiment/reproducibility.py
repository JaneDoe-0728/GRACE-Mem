"""
python - <<'PY'
from experiment.reproducibility import activate_reproducibility
import random, json
import numpy as np
import torch

def sample_once():
    activate_reproducibility()
    return {
        "python": [random.random() for _ in range(3)],
        "numpy": np.random.rand(3).tolist(),
        "torch": torch.rand(3).tolist(),
    }

a = sample_once()
b = sample_once()

print(json.dumps({
    "match": a == b,
    "first": a,
    "second": b,
}, indent=2))
PY
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_DETERMINISTIC = True

_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_NCCL_ENV_DEFAULTS = {
    "NCCL_ASYNC_ERROR_HANDLING": "1",
    "NCCL_BLOCKING_WAIT": "1",
}

_RUNTIME_CONFIG: "ReproducibilityConfig | None" = None


@dataclass(frozen=True)
class ReproducibilityConfig:
    seed: int = DEFAULT_SEED
    deterministic: bool = DEFAULT_DETERMINISTIC


def _load_experiment_reproducibility_params() -> dict[str, Any]:
    try:
        module = importlib.import_module("experiment_config")
    except Exception:
        return {}
    params = getattr(module, "REPRODUCIBILITY_PARAMS", None)
    return dict(params) if isinstance(params, dict) else {}


def resolve_reproducibility_config(
    *,
    seed: int | None = None,
    deterministic: bool | None = None,
) -> ReproducibilityConfig:
    params = _load_experiment_reproducibility_params()
    resolved_seed = int(seed if seed is not None else params.get("seed", DEFAULT_SEED))
    resolved_deterministic = bool(
        deterministic if deterministic is not None else params.get("deterministic", DEFAULT_DETERMINISTIC)
    )
    return ReproducibilityConfig(
        seed=resolved_seed,
        deterministic=resolved_deterministic,
    )


def prepare_reproducibility_env(seed: int, deterministic: bool = True) -> dict[str, str]:
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
    prepare_reproducibility_env(seed, deterministic)

    import random

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
            "If a downstream op is unsupported under deterministic mode, keep "
            "`REPRODUCIBILITY_PARAMS['deterministic']=True` for visibility and "
            "fix the offending op instead of silently falling back."
        ) from exc

    return current_reproducibility_state(
        ReproducibilityConfig(seed=int(seed), deterministic=bool(deterministic))
    )


def activate_reproducibility(
    *,
    seed: int | None = None,
    deterministic: bool | None = None,
    log_prefix: str | None = None,
) -> ReproducibilityConfig:
    global _RUNTIME_CONFIG
    cfg = resolve_reproducibility_config(seed=seed, deterministic=deterministic)
    set_global_seed(cfg.seed, cfg.deterministic)
    _RUNTIME_CONFIG = cfg
    if log_prefix:
        logger.info("%s %s", log_prefix, json.dumps(current_reproducibility_state(cfg), ensure_ascii=False))
    return cfg


def get_runtime_reproducibility() -> ReproducibilityConfig:
    global _RUNTIME_CONFIG
    if _RUNTIME_CONFIG is None:
        _RUNTIME_CONFIG = resolve_reproducibility_config()
    return _RUNTIME_CONFIG


def current_reproducibility_state(cfg: ReproducibilityConfig | None = None) -> dict[str, Any]:
    resolved = cfg or get_runtime_reproducibility()
    payload = asdict(resolved)
    payload["env"] = {
        "EXPERIMENT_SEED": os.environ.get("EXPERIMENT_SEED"),
        "EXPERIMENT_DETERMINISTIC": os.environ.get("EXPERIMENT_DETERMINISTIC"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        **{key: os.environ.get(key) for key in _NCCL_ENV_DEFAULTS},
    }
    return payload


def attach_reproducibility_metadata(payload: dict[str, Any], *, cfg: ReproducibilityConfig | None = None) -> dict[str, Any]:
    out = dict(payload)
    out["reproducibility"] = current_reproducibility_state(cfg)
    return out


def write_reproducibility_file(directory: str | Path, *, filename: str = "reproducibility.json") -> Path:
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    path.write_text(
        json.dumps(current_reproducibility_state(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def build_dataloader_seed_components(seed: int) -> tuple["Callable[[int], None]", Any]:
    import random
    import numpy as np
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = seed + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)

    return _seed_worker, generator
