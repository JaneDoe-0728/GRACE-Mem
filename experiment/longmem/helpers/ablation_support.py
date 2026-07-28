from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any

from experiment.longmem.helpers.analysis_cases import analysis_dir_for, data_folder_for, output_dir_for, scenario_alias
from experiment.longmem.helpers.rerun_support import read_summary_accuracy, upsert_result_csv
from experiment.longmem.utils.io import ensure_dir, latest_glob_match, write_text_file


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiment"
LONGMEM_DIR = EXP_DIR / "longmem"
CONFIG_PATH = EXP_DIR / "experiment_config.py"
ABLATION_RESULTS_DIR = LONGMEM_DIR / "output" / "ablation"
ABLATION_OUT = ABLATION_RESULTS_DIR / "ablation_results.csv"

SCENARIO_TO_TYPE = {
    "temporal": "temporal_reasoning",
    "multi_session": "multi_session",
}
PREFILLED: dict[str, dict[str, tuple[int, int]]] = {
    "baseline": {
        "temporal": (0, 46),
        "multi_session": (0, 65),
    },
    "reranker_topk_5": {
        "temporal": (17, 46),
        "multi_session": (8, 65),
    },
}
VARIANTS: dict[str, dict] = {
    "baseline": {},
    "reranker_topk_5": {"reranker_topk": 5},
    "filter_ent_thresh_03": {"filter_ent_threshold": 0.3},
    "filter_rel_thresh_03": {"filter_rel_threshold": 0.3},
    "filter_thresh_03": {"filter_ent_threshold": 0.3, "filter_rel_threshold": 0.3},
    "filter_ent_topk_15": {"filter_ent_topk": 15},
    "filter_rel_topk_15": {"filter_rel_topk": 15},
    "filter_topk_15": {"filter_ent_topk": 15, "filter_rel_topk": 15},
    "filter_thresh_03_topk_15": {
        "filter_ent_threshold": 0.3,
        "filter_rel_threshold": 0.3,
        "filter_ent_topk": 15,
        "filter_rel_topk": 15,
    },
    "reranker_thresh_m5": {"reranker_threshold": -5.0},
    "summary_topk_8": {"summary_topk_per_item": 8},
    "all_relaxed": {
        "filter_ent_threshold": 0.3,
        "filter_rel_threshold": 0.3,
        "filter_ent_topk": 15,
        "filter_rel_topk": 15,
        "reranker_threshold": -5.0,
        "summary_topk_per_item": 8,
    },
}

NOCO_TABLE = "ablation_results"
_noco_client = None
_noco_project_id = None
_noco_table_id = None


def scenario_output_root(run_tag: str, scenario: str) -> Path:
    return output_dir_for(run_tag, SCENARIO_TO_TYPE[scenario])


def scenario_data_folder(scenario: str) -> Path:
    return data_folder_for(SCENARIO_TO_TYPE[scenario])


def scenario_analysis_dir(run_tag: str, scenario: str) -> Path:
    return analysis_dir_for(run_tag, SCENARIO_TO_TYPE[scenario])


def load_experiment_config() -> tuple[dict, dict, dict]:
    spec = importlib.util.spec_from_file_location("experiment_config_runtime", CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load experiment config from {CONFIG_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        dict(getattr(module, "INGEST_PARAMS", {})),
        dict(getattr(module, "RETRIEVAL_PARAMS", {})),
        dict(getattr(module, "RERANKER_PARAMS", {})),
    )


def build_config(overrides: dict) -> tuple[dict, dict]:
    _, retrieval, reranker = load_experiment_config()
    for key, value in overrides.items():
        if key in retrieval:
            retrieval[key] = value
        elif key in reranker:
            reranker[key] = value
        else:
            raise ValueError(f"Unknown param: {key}")
    return retrieval, reranker


def write_config(retrieval: dict, reranker: dict) -> None:
    ingest, _, _ = load_experiment_config()
    lines = [
        '"""',
        "Single source of truth for ALL experiment parameters (ingestion + retrieval + reranker).",
        'Change values here — all experiment scripts will use them automatically.',
        '"""',
        "",
        "# ── Ingestion parameters ───────────────────────────────────────────────────",
        "INGEST_PARAMS = dict(",
    ]
    for key, value in ingest.items():
        lines.append(f"    {key}={repr(value)},")
    lines += [
        ")",
        "",
        "# ── Parameters passed to build_kg_context() per call ──────────────────────",
        "RETRIEVAL_PARAMS = dict(",
    ]
    for key, value in retrieval.items():
        lines.append(f"    {key}={repr(value)},")
    lines += [
        ")",
        "",
        "# ── Parameters set at Retriever init time (reranker) ──────────────────────",
        "RERANKER_PARAMS = dict(",
    ]
    for key, value in reranker.items():
        lines.append(f"    {key}={repr(value)},")
    lines += [")", ""]
    write_text_file(CONFIG_PATH, "\n".join(lines))


def latest_summary(output_dir: Path) -> Path | None:
    return latest_glob_match(output_dir, "rerun_summary_*.json")


def ensure_noco_columns(client, table_id: str, row: dict[str, Any]) -> None:
    schema = client.get_table_schema(table_id)
    existing_cols = {column["column_name"].lower() for column in schema["columns"]}
    for column in row:
        if column.lower() not in existing_cols:
            uidt = "LongText" if column in ("changed_params",) else "SingleLineText"
            client.create_column(table_id, column_name=column, column_title=column, uidt=uidt)


def get_noco_client():
    global _noco_client, _noco_project_id
    if _noco_client is not None:
        return _noco_client

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "noco-db-uploader" / ".env")
    os.environ.setdefault("NOCO_TARGETS_PATH", str(ROOT / "experiment" / "noco" / "noco_targets.yaml"))
    noco_url = os.getenv("NOCO_URL")
    api_token = os.getenv("API_TOKEN")
    _noco_project_id = os.getenv("PROJECT_ID")
    if not all([noco_url, api_token, _noco_project_id]):
        raise EnvironmentError("NOCO_URL, API_TOKEN, PROJECT_ID must be set in .env")

    sys.path.insert(0, str(ROOT / "noco-db-uploader"))
    from src.noco_client import NocoDBClient

    _noco_client = NocoDBClient(noco_url, api_token)
    return _noco_client


def get_noco_table_id(row: dict[str, Any]) -> str:
    global _noco_table_id
    if _noco_table_id is not None:
        return _noco_table_id
    client = get_noco_client()
    try:
        _noco_table_id = client.get_table_id_by_name(_noco_project_id, NOCO_TABLE)
    except ValueError:
        source_id = client.get_first_source_id(_noco_project_id)
        _noco_table_id = client.create_table(_noco_project_id, source_id, NOCO_TABLE, NOCO_TABLE)
        ensure_noco_columns(client, _noco_table_id, row)
    return _noco_table_id


def upsert_noco(row: dict[str, Any]) -> None:
    try:
        client = get_noco_client()
        table_id = get_noco_table_id(row)
        ensure_noco_columns(client, table_id, row)
        record = {key: str(value) for key, value in row.items()}
        variant = record.get("variant", "")
        existing = client.list_records(_noco_project_id, table_id, where=f"(variant,eq,{variant})")
        if existing:
            client.update_record(_noco_project_id, table_id, existing[0]["Id"], record)
        else:
            client.insert_record(_noco_project_id, table_id, record)
        print(f"  [NOCO] upserted variant={variant}")
    except Exception as exc:
        logger.warning("NocoDB upsert failed (non-fatal): %s", exc)


def upsert_csv(row: dict[str, Any]) -> None:
    ensure_dir(ABLATION_RESULTS_DIR)
    upsert_result_csv(ABLATION_OUT, row)
    print(f"  [CSV] updated → {ABLATION_OUT.name}")


def read_accuracy(summary_path: Path) -> tuple[int, int]:
    return read_summary_accuracy(summary_path)
