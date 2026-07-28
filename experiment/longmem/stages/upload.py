from __future__ import annotations

import importlib
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from experiment.longmem.utils.io import read_csv_frame, remove_file, write_csv_frame


logger = logging.getLogger(__name__)


class UploadStage:
    """LongMem upload stage for NocoDB progress sync."""

    DATASET_FAMILY = "longmem"
    EXCLUDE_COLS = {"stuck_history"}
    SUMMARY_DATASET = "SAMPLE_ACCURACY"
    PROGRESS_COLUMNS = [
        ("dataset", "SingleLineText"),
        ("status", "SingleLineText"),
        ("correctness", "SingleLineText"),
        ("question", "LongText"),
        ("gold_answer", "LongText"),
        ("generated_answer", "LongText"),
        ("updated_at", "SingleLineText"),
    ]

    def __init__(self) -> None:
        self._client_cache: dict[str, Any] = {}
        self._project_id_cache: dict[str, str] = {}
        self._table_id_cache: dict[str, str] = {}

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[3]

    @staticmethod
    def _uploader_root() -> Path:
        return UploadStage._repo_root() / "noco-db-uploader"

    def _ensure_noco_import_path(self) -> None:
        uploader_root = self._uploader_root()
        if str(uploader_root) not in sys.path:
            sys.path.insert(0, str(uploader_root))
            logger.info("Added Noco uploader path: %s", uploader_root)

    def _load_noco_env(self) -> None:
        from dotenv import load_dotenv

        repo_root = self._repo_root()
        load_dotenv(repo_root / ".env")
        load_dotenv(self._uploader_root() / ".env")
        os.environ.setdefault(
            "NOCO_TARGETS_PATH",
            str(repo_root / "experiment" / "noco" / "noco_targets.yaml"),
        )
        logger.info("Loaded NocoDB environment from repo and uploader .env files")

    def _load_noco_modules(self):
        self._ensure_noco_import_path()
        config_mod = importlib.import_module("config")
        noco_client_mod = importlib.import_module("src.noco_client")
        src_mod = importlib.import_module("src")
        logger.info("Imported NocoDB uploader modules")
        return config_mod.NocoDBConfig, noco_client_mod.NocoDBClient, src_mod

    def _get_client(self, *, dataset: str | None = None):
        dataset_key = dataset or "env"
        if dataset_key in self._client_cache:
            logger.info("Reusing cached NocoDB client for dataset key '%s'", dataset_key)
            return self._client_cache[dataset_key]

        self._load_noco_env()
        api_token = os.getenv("API_TOKEN")
        if not api_token:
            raise EnvironmentError("API_TOKEN must be set in .env")

        NocoDBConfig, NocoDBClient, _ = self._load_noco_modules()
        noco_config = NocoDBConfig.from_dataset(dataset) if dataset else NocoDBConfig.from_env()
        if not noco_config.noco_url or not noco_config.project_id:
            raise EnvironmentError("NOCO_URL and PROJECT_ID must be set for NocoDB upload")

        self._project_id_cache[dataset_key] = noco_config.project_id
        self._client_cache[dataset_key] = NocoDBClient(noco_config.noco_url, api_token)
        logger.info(
            "Created NocoDB client for dataset key '%s' with project '%s'",
            dataset_key,
            noco_config.project_id,
        )
        return self._client_cache[dataset_key]

    def _create_progress_table(self, *, table_name: str, dataset: str | None = None) -> str:
        dataset_key = dataset or "env"
        client = self._get_client(dataset=dataset)
        project_id = self._project_id_cache[dataset_key]
        source_id = client.get_first_source_id(project_id)
        logger.info(
            "Creating NocoDB progress table '%s' for dataset key '%s' in project '%s'",
            table_name,
            dataset_key,
            project_id,
        )
        table_id = client.create_table(project_id, source_id, table_name, table_name)

        schema = client.get_table_schema(table_id)
        existing_cols = {column["column_name"].lower() for column in schema["columns"]}
        for column_name, uidt in self.PROGRESS_COLUMNS:
            if column_name.lower() not in existing_cols:
                logger.info("Adding missing NocoDB column '%s' to table '%s'", column_name, table_name)
                client.create_column(
                    table_id,
                    column_name=column_name,
                    column_title=column_name,
                    uidt=uidt,
                )
        logger.info("Created NocoDB progress table '%s' with table id '%s'", table_name, table_id)
        return table_id

    def _get_table_id(self, *, table_name: str, dataset: str | None = None) -> str:
        dataset_key = dataset or "env"
        cache_key = f"{dataset_key}::{table_name}"
        if cache_key in self._table_id_cache:
            logger.info("Reusing cached table id for '%s'", cache_key)
            return self._table_id_cache[cache_key]

        client = self._get_client(dataset=dataset)
        project_id = self._project_id_cache[dataset_key]
        try:
            table_id = client.get_table_id_by_name(project_id, table_name)
            logger.info("Resolved existing NocoDB table '%s' to id '%s'", table_name, table_id)
        except ValueError:
            logger.warning("NocoDB table '%s' not found; creating it now", table_name)
            table_id = self._create_progress_table(table_name=table_name, dataset=dataset)

        self._table_id_cache[cache_key] = table_id
        return table_id

    def upsert_progress_row(self, *, table_name: str, row: dict) -> None:
        record = {key: value for key, value in row.items() if key not in self.EXCLUDE_COLS}
        dataset_key = self.DATASET_FAMILY
        client = self._get_client(dataset=dataset_key)
        project_id = self._project_id_cache[dataset_key]
        table_id = self._get_table_id(table_name=table_name, dataset=dataset_key)

        dataset_name = record.get("dataset")
        logger.info(
            "Upserting progress row for dataset '%s' into table '%s' (%d fields)",
            dataset_name,
            table_name,
            len(record),
        )
        existing = client.list_records(
            project_id,
            table_id,
            where=f"(dataset,eq,{dataset_name})",
        )
        if existing:
            row_id = existing[0]["Id"]
            logger.info(
                "Updating existing NocoDB progress row for dataset '%s' with row id '%s'",
                dataset_name,
                row_id,
            )
            client.update_record(project_id, table_id, row_id, record)
        else:
            logger.info("Inserting new NocoDB progress row for dataset '%s'", dataset_name)
            client.insert_record(project_id, table_id, record)

    @classmethod
    def _build_accuracy_summary_row(cls, df: pd.DataFrame) -> dict[str, str]:
        if "dataset" not in df.columns or "correctness" not in df.columns:
            correct = 0
            total = 0
        else:
            work_df = df.copy()
            dataset_series = work_df["dataset"].fillna("").astype(str).str.strip()
            correctness_series = work_df["correctness"].fillna("").astype(str).str.strip()
            valid_mask = dataset_series.ne(cls.SUMMARY_DATASET)
            judged_mask = correctness_series.isin(("0", "1"))
            total = int((valid_mask & judged_mask).sum())
            correct = int((valid_mask & correctness_series.eq("1")).sum())

        percent = (correct / total * 100.0) if total else 0.0
        return {
            "dataset": cls.SUMMARY_DATASET,
            "status": "summary",
            "correctness": f"{correct}/{total} ({percent:.1f}%)",
        }

    @classmethod
    def append_accuracy_summary_row(cls, df: pd.DataFrame) -> pd.DataFrame:
        summary_row = cls._build_accuracy_summary_row(df)
        filtered_df = df
        if "dataset" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["dataset"].fillna("").astype(str).str.strip().ne(cls.SUMMARY_DATASET)
            ].copy()
        return pd.concat([filtered_df, pd.DataFrame([summary_row])], ignore_index=True)

    def upsert_accuracy_summary(self, *, table_name: str, progress_df: pd.DataFrame) -> None:
        self.upsert_progress_row(
            table_name=table_name,
            row=self._build_accuracy_summary_row(progress_df),
        )

    def upload_progress_file(self, *, csv_path: str | Path, table_name: str) -> str:
        self._load_noco_env()
        _, _, src_mod = self._load_noco_modules()
        upload_file = src_mod.upload_file

        csv_path = Path(csv_path)
        logger.info("Uploading progress CSV '%s' to table '%s'", csv_path, table_name)
        df = read_csv_frame(csv_path, dtype=str, encoding="utf-8-sig")
        logger.info("Loaded progress CSV with %d rows and %d columns", len(df), len(df.columns))
        cols_to_drop = [column for column in df.columns if column.lower() in self.EXCLUDE_COLS]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info("Dropped excluded columns before upload: %s", cols_to_drop)
        df = self.append_accuracy_summary_row(df)
        logger.info("Appended category accuracy summary row for table '%s'", table_name)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        write_csv_frame(df, Path(tmp_path))
        logger.info("Wrote temporary upload CSV: %s", tmp_path)

        try:
            upload_id = upload_file(
                tmp_path,
                table_title=table_name,
                table_name=table_name,
                dataset=self.DATASET_FAMILY,
            )
            logger.info(
                "Completed NocoDB CSV upload for table '%s' with result '%s'",
                table_name,
                upload_id,
            )
            return upload_id
        finally:
            remove_file(Path(tmp_path))
            logger.info("Removed temporary upload CSV: %s", tmp_path)
