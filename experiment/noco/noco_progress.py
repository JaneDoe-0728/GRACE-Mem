"""
NocoDB progress row upsert helper.

Called from processor.py after each _save_progress_row to keep
NocoDB in sync. Each output subfolder maps to its own NocoDB table.

Requires in .env (root):
  NOCO_URL, API_TOKEN, PROJECT_ID
  ORG  (optional, defaults to "noco")

Failures are non-fatal — a warning is logged and the main pipeline continues.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any

from experiment.noco.client_loader import load_noco_client_class

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_client = None
_project_id = None
_org = None
_table_id_cache: Dict[str, str] = {}  # table_name -> table_id

EXCLUDE_COLS = {"stuck_history"}


def _get_client():
    global _client, _project_id, _org
    if _client is not None:
        return _client

    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(REPO_ROOT / "noco-db-uploader" / ".env")
    os.environ.setdefault(
        "NOCO_TARGETS_PATH",
        str(REPO_ROOT / "experiment" / "noco" / "noco_targets.yaml"),
    )

    noco_url = os.getenv("NOCO_URL")
    api_token = os.getenv("API_TOKEN")
    _project_id = os.getenv("PROJECT_ID")
    _org = os.getenv("ORG", "noco")

    if not all([noco_url, api_token, _project_id]):
        raise EnvironmentError("NOCO_URL, API_TOKEN, PROJECT_ID must be set in .env")

    NocoDBClient = load_noco_client_class()
    _client = NocoDBClient(noco_url, api_token)
    return _client


# Columns for the progress table (order matters for display; stuck_history is local-only)
_PROGRESS_COLUMNS = [
    ("dataset",            "SingleLineText"),
    ("status",             "SingleLineText"),
    ("correctness",        "SingleLineText"),
    ("question",           "LongText"),
    ("gold_answer",        "LongText"),
    ("generated_answer",   "LongText"),
    ("updated_at",         "SingleLineText"),
]


def _create_progress_table(table_name: str) -> str:
    """Create a new progress table with the standard columns. Returns the new table_id."""
    client = _get_client()
    source_id = client.get_first_source_id(_project_id)
    table_id = client.create_table(_project_id, source_id, table_name, table_name)

    schema = client.get_table_schema(table_id)
    existing_cols = {c["column_name"].lower() for c in schema["columns"]}
    for col_name, uidt in _PROGRESS_COLUMNS:
        if col_name.lower() not in existing_cols:
            client.create_column(table_id, column_name=col_name, column_title=col_name, uidt=uidt)

    logger.info("NocoDB: created table '%s' (id=%s) with %d columns.", table_name, table_id, len(_PROGRESS_COLUMNS))
    return table_id


def _get_table_id(table_name: str) -> str:
    if table_name in _table_id_cache:
        return _table_id_cache[table_name]

    client = _get_client()
    try:
        table_id = client.get_table_id_by_name(_project_id, table_name)
    except ValueError:
        logger.info("NocoDB: table '%s' not found, creating it automatically...", table_name)
        table_id = _create_progress_table(table_name)

    _table_id_cache[table_name] = table_id
    return table_id


def upsert_row(table_name: str, row: Dict[str, Any]) -> None:
    """
    Upsert a progress row into the NocoDB table for the given subfolder.

    Args:
        table_name: NocoDB table name (= output subfolder name, e.g. "multi_session")
        row: dict with progress fields (stuck_history is automatically excluded)
    """
    record = {k: v for k, v in row.items() if k not in EXCLUDE_COLS}

    client = _get_client()
    table_id = _get_table_id(table_name)

    dataset = record.get("dataset")
    existing = client.list_records(
        _project_id,
        table_id,
        where=f"(dataset,eq,{dataset})",
    )

    if existing:
        row_id = existing[0]["Id"]
        client.update_record(_project_id, table_id, row_id, record)
    else:
        client.insert_record(_project_id, table_id, record)
