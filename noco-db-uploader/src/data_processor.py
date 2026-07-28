import json
import pandas as pd
from typing import Dict, List, Any, Optional, Set


def normalize_name(s: str) -> str:
    """Normalize column names to lowercase with underscores"""
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_record(row: Dict[str, Any], str_columns: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Normalize a record dict for safe NocoDB insertion.

    - None → ""
    - bool → "true" / "false"
    - dict / list → JSON string
    - columns in str_columns → str(value)  (use for numeric fields stored as SingleLineText)
    - everything else → unchanged
    """
    normalized: Dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            normalized[key] = ""
        elif isinstance(value, bool):
            normalized[key] = "true" if value else "false"
        elif isinstance(value, (dict, list)):
            normalized[key] = json.dumps(value, ensure_ascii=False)
        elif str_columns and key in str_columns:
            normalized[key] = str(value)
        else:
            normalized[key] = value
    return normalized


class ExcelProcessor:
    """Process Excel files for NocoDB upload"""

    def __init__(self, xlsx_path: str):
        self.xlsx_path = xlsx_path
        self.df = None

    def load_excel(self) -> pd.DataFrame:
        """Load and normalize Excel/CSV file (auto-detects actual format)"""
        path = self.xlsx_path

        # Detect actual file type by sniffing the first bytes
        with open(path, "rb") as f:
            magic = f.read(4)

        # ZIP magic bytes = real xlsx/xlsm; otherwise treat as CSV
        if magic[:2] == b"PK":
            self.df = pd.read_excel(path, engine="openpyxl")
        else:
            # Try common CSV encodings
            for enc in ("utf-8-sig", "utf-8", "cp950", "latin-1"):
                try:
                    self.df = pd.read_csv(path, encoding=enc)
                    print(f"Loaded as CSV (encoding={enc})")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Cannot read file as CSV with common encodings: {path}")

        self.df.columns = [normalize_name(c) for c in self.df.columns]
        print("Columns:", list(self.df.columns))
        return self.df

    def get_columns(self) -> List[str]:
        """Get list of normalized column names"""
        if self.df is None:
            raise ValueError("Excel file not loaded. Call load_excel() first.")
        return list(self.df.columns)

    def prepare_records(self, valid_columns: Set[str]) -> List[Dict[str, Any]]:
        """
        Convert DataFrame to list of records for NocoDB

        Args:
            valid_columns: Set of valid column names from table schema

        Returns:
            List of record dictionaries
        """
        if self.df is None:
            raise ValueError("Excel file not loaded. Call load_excel() first.")

        payload = []

        for _, row in self.df.iterrows():
            record = {}

            # Map all columns directly
            for col in self.df.columns:
                if col in valid_columns:
                    v = row[col]
                    record[col] = None if pd.isna(v) else str(v)

            payload.append(record)

        print(f"Prepared {len(payload)} rows")
        return payload