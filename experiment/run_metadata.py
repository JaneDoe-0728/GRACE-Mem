from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from experiment.reproducibility import attach_reproducibility_metadata


_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATHS = {
    ".env": _REPO_ROOT / ".env",
}
_SECRET_HINTS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "PWD",
    "CREDENTIAL",
    "AUTH",
)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return namespace_to_dict(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def namespace_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: to_jsonable(value)
        for key, value in vars(args).items()
        if not key.startswith("_")
    }


def _is_secret_like(name: str) -> bool:
    upper = str(name).upper()
    return any(hint in upper for hint in _SECRET_HINTS)


def _mask_env_value(name: str, value: str | None) -> str | None:
    if value in (None, ""):
        return value
    if not _is_secret_like(name):
        return value
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _collect_env_snapshot() -> dict[str, Any]:
    files: dict[str, Any] = {}
    resolved: dict[str, Any] = {}
    all_keys: set[str] = set()
    parsed_by_label: dict[str, dict[str, str | None]] = {}

    for label, path in _ENV_PATHS.items():
        if not path.exists():
            continue
        parsed_raw = {
            str(key): (None if value is None else str(value))
            for key, value in dotenv_values(path).items()
        }
        if not parsed_raw:
            continue
        parsed_by_label[label] = parsed_raw
        all_keys.update(parsed_raw.keys())
        files[label] = {
            "path": str(path),
            "values": {key: _mask_env_value(key, value) for key, value in parsed_raw.items()},
        }

    for key in sorted(all_keys):
        value = os.environ.get(key)
        if value in (None, ""):
            for parsed_raw in parsed_by_label.values():
                fallback = parsed_raw.get(key)
                if fallback not in (None, ""):
                    value = str(fallback)
                    break
        resolved[key] = _mask_env_value(key, value if value is None else str(value))

    return {
        "files": files,
        "resolved": resolved,
    }


def write_run_metadata(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    if target.exists():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                existing = data
        except Exception:
            existing = {}

    merged = dict(existing)
    for key, value in payload.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value

    env_snapshot = _collect_env_snapshot()
    merged["env"] = env_snapshot

    model_name = env_snapshot.get("resolved", {}).get("MODEL_NAME")
    if model_name and "model_name" not in merged:
        merged["model_name"] = model_name

    target.write_text(
        json.dumps(attach_reproducibility_metadata(to_jsonable(merged)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target
