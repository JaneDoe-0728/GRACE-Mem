import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ENV_TARGETS_PATH_KEY = "NOCO_TARGETS_PATH"


@dataclass(frozen=True)
class NocoTarget:
    name: str
    datasets: tuple[str, ...]
    noco_url: str
    org: str
    project_id: str


def _load_targets_config(targets_path: str | Path | None = None) -> dict[str, Any]:
    path = _resolve_targets_path(targets_path)
    if not path.exists():
        return {"targets": {}}
    targets: dict[str, Any] = {}
    current_target: dict[str, Any] | None = None
    current_target_name: str | None = None
    current_list_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0:
            if stripped != "targets:":
                raise ValueError(f"Invalid target config in {path}: unexpected root entry {stripped!r}")
            continue
        if indent == 2 and stripped.endswith(":"):
            current_target_name = stripped[:-1].strip()
            current_target = {}
            targets[current_target_name] = current_target
            current_list_key = None
            continue
        if current_target is None:
            raise ValueError(f"Invalid target config in {path}: field declared before target name")
        if indent == 4 and stripped.endswith(":"):
            key = stripped[:-1].strip()
            current_target[key] = []
            current_list_key = key
            continue
        if indent == 4 and ":" in stripped:
            key, value = stripped.split(":", 1)
            current_target[key.strip()] = value.strip()
            current_list_key = None
            continue
        if indent == 6 and stripped.startswith("- "):
            if current_list_key is None or not isinstance(current_target.get(current_list_key), list):
                raise ValueError(f"Invalid target config in {path}: list item without list key")
            current_target[current_list_key].append(stripped[2:].strip())
            continue
        raise ValueError(f"Invalid target config in {path}: unsupported line {raw_line!r}")

    return {"targets": targets}


def _resolve_targets_path(targets_path: str | Path | None = None) -> Path:
    if targets_path:
        return Path(targets_path)
    env_path = os.getenv(ENV_TARGETS_PATH_KEY, "").strip()
    if env_path:
        return Path(env_path)
    # Keep the package generic: dataset routing lives with the caller unless
    # they explicitly point to a targets file.
    return Path(__file__).resolve().parent / "noco_targets.yaml"


def _normalize_dataset_key(dataset: str) -> str:
    value = str(dataset or "").strip()
    lowered = value.lower()
    aliases = {
        "locomo10": "locomo",
        "locomo_failed_cases": "locomo",
        "locomo-plus": "locomo-plus",
        "locomo_plus": "locomo-plus",
        "locomo": "locomo",
        "longmem": "longmem",
        "longmem_failed_cases": "longmem",
    }
    return aliases.get(lowered, lowered)


def _build_target(name: str, payload: dict[str, Any]) -> NocoTarget:
    datasets = tuple(str(item).strip() for item in payload.get("datasets", []) if str(item).strip())
    return NocoTarget(
        name=name,
        datasets=datasets,
        noco_url=_normalize_scalar(payload.get("noco_url", "")),
        org=_normalize_scalar(payload.get("org", "noco")) or "noco",
        project_id=_normalize_scalar(payload.get("project_id", "")),
    )


def _normalize_scalar(value: Any) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def resolve_noco_target(dataset: str, targets_path: str | Path | None = None) -> NocoTarget:
    normalized = _normalize_dataset_key(dataset)
    targets = _load_targets_config(targets_path)["targets"]
    for name, payload in targets.items():
        if not isinstance(payload, dict):
            continue
        target = _build_target(name, payload)
        if normalized in {item.lower() for item in target.datasets}:
            if not target.noco_url or not target.project_id:
                raise ValueError(
                    f"Noco target '{name}' matched dataset '{dataset}' but is missing noco_url/project_id"
                )
            return target
    raise KeyError(f"No Noco target mapping found for dataset '{dataset}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve dataset -> NocoDB target from YAML")
    parser.add_argument("--dataset", required=True, help="Dataset name or family, e.g. locomo-plus")
    parser.add_argument("--targets-path", default=None, help="Optional custom YAML path")
    args = parser.parse_args()

    target = resolve_noco_target(args.dataset, args.targets_path)
    print(
        json.dumps(
            {
                "name": target.name,
                "datasets": list(target.datasets),
                "noco_url": target.noco_url,
                "org": target.org,
                "project_id": target.project_id,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
