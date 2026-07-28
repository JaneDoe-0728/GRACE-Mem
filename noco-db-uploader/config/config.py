import os
from dataclasses import dataclass
from typing import Optional

from .targets import resolve_noco_target


@dataclass
class NocoDBConfig:
    """NocoDB configuration"""
    noco_url: str
    api_token: str
    org: str
    project_id: str
    source_id: Optional[str] = None

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        source_id = os.getenv("SOURCE_ID", "")
        return cls(
            noco_url=os.getenv("NOCO_URL", "http://localhost:8080/"),
            api_token=os.getenv("API_TOKEN", ""),
            org=os.getenv("ORG", "noco"),
            project_id=os.getenv("PROJECT_ID", ""),
            source_id=source_id if source_id else None,
        )

    @classmethod
    def from_dataset(cls, dataset: str):
        """Load configuration from YAML target routing for a dataset family."""
        source_id = os.getenv("SOURCE_ID", "")
        target = resolve_noco_target(dataset)
        return cls(
            noco_url=target.noco_url,
            api_token=os.getenv("API_TOKEN", ""),
            org=target.org,
            project_id=target.project_id,
            source_id=source_id if source_id else None,
        )


@dataclass
class UploadConfig:
    """Upload configuration"""
    table_title: str
    table_name: str
    xlsx_path: str

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            table_title=os.getenv("TABLE_TITLE", "table_from_xlsx"),
            table_name=os.getenv("TABLE_NAME", "table_from_xlsx"),
            xlsx_path=os.getenv("XLSX_PATH", ""),
        )
