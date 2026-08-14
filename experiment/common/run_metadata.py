"""Capture what a run was configured with, so results stay interpretable.

A results directory full of numbers is worthless six months later if nobody can
say which model, dataset, seed, and flags produced them. This module writes that
context alongside the results: parsed arguments, relevant environment, and the
reproducibility state.

Environment capture is the delicate part. A .env holds API keys next to the
configuration worth recording, and run metadata gets committed, shared, and
pasted into issues. Values whose names match `_SECRET_HINTS` are therefore
redacted -- name-based rather than value-based, because there is no reliable
way to recognise a credential by its contents, and a false positive costs one
redacted line while a false negative leaks a key.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from experiment.common.reproducibility import attach_reproducibility_metadata


_REPO_ROOT = Path(__file__).resolve().parents[2]
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
    """Coerce any value into something `json.dumps` will accept.

    Total by construction: anything unrecognised falls through to `str(value)`
    rather than raising. Metadata capture must never be the thing that fails a
    run -- a config value rendered as its repr is still more useful than a
    crash after the compute has been spent.
    """
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
    """Flatten parsed CLI arguments into a JSON-safe dict.

    Underscore-prefixed attributes are dropped: argparse and the run scripts
    stash internal handles there, and those are neither serializable nor part
    of the run's configuration.
    """
    return {
        key: to_jsonable(value)
        for key, value in vars(args).items()
        if not key.startswith("_")
    }


def _is_secret_like(name: str) -> bool:
    """Judge from the variable's name alone whether its value is a credential.

    Substring matching, so OPENAI_API_KEY, KEY_PATH, and MY_TOKEN_V2 all
    trigger. Over-matching is the intended bias: the cost of redacting a
    harmless variable is one unreadable line in the metadata, and the cost of
    missing a real one is a leaked credential in a committed file.
    """
    upper = str(name).upper()
    return any(hint in upper for hint in _SECRET_HINTS)


def _mask_env_value(name: str, value: str | None) -> str | None:
    """Redact a secret value, keeping enough to identify which one it was.

    Long values keep their first and last four characters, which is enough to
    confirm *which* key a run used without disclosing it. Values of 8
    characters or fewer are starred out entirely -- at that length the retained
    prefix and suffix would be most of the secret.
    """
    if value in (None, ""):
        return value
    if not _is_secret_like(name):
        return value
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _collect_env_snapshot() -> dict[str, Any]:
    """Record the environment two ways: as written, and as actually resolved.

    They routinely differ, and the difference is what makes runs hard to
    explain after the fact. A shell export overrides .env, so the file says one
    model and the run used another. `files` records what each .env declares;
    `resolved` records what the process would actually read -- os.environ
    first, falling back to the file.

    Only keys that appear in some .env are resolved. Snapshotting all of
    os.environ would dump the entire shell into the metadata, most of it
    irrelevant and some of it sensitive.

    Every value passes through `_mask_env_value` on both paths.
    """
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
    """Merge `payload` into the run's metadata file and rewrite it.

    Merges rather than overwrites because a run's stages each contribute their
    own facts as they finish, and a plain write would leave only the last
    one's. The merge is one level deep -- nested dicts are combined key by key,
    anything else is replaced -- which is enough for the per-stage sections
    this is used for and avoids the surprises of a deep merge on lists.

    A corrupt existing file is treated as absent rather than raising: a run
    that has finished its compute must still be able to record its results.
    That does discard whatever the unreadable file held.

    Returns:
        The path written.
    """
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

    # Promote MODEL_NAME to a top-level field. It is the first thing anyone
    # comparing two runs looks for, and buried in env.resolved it is easy to
    # miss. Only when the caller did not already set one -- an explicit value
    # is more authoritative than the ambient environment.
    model_name = env_snapshot.get("resolved", {}).get("MODEL_NAME")
    if model_name and "model_name" not in merged:
        merged["model_name"] = model_name

    target.write_text(
        json.dumps(attach_reproducibility_metadata(to_jsonable(merged)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target
