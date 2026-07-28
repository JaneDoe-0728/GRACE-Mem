import requests
from typing import Callable, Dict, List, Any, Optional, Set
from datetime import datetime


def _default_uidt(column: str, value: Any) -> str:
    """Generic column-type inference: numeric values → Number, everything else → SingleLineText."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "Number"
    return "SingleLineText"


def table_needs_recreation(
    schema: Dict[str, Any],
    row: Dict[str, Any],
    uidt_fn: Callable[[str, Any], str] = None,
) -> bool:
    """Return True if any existing column has a different uidt than desired."""
    if uidt_fn is None:
        uidt_fn = _default_uidt
    schema_columns = {c["column_name"].lower(): c for c in schema["columns"]}
    for col in row:
        if col.lower() not in schema_columns:
            continue
        if schema_columns[col.lower()].get("uidt") != uidt_fn(col, row[col]):
            return True
    return False


class NocoDBClient:
    """Client for interacting with NocoDB API"""

    def __init__(self, noco_url: str, api_token: str, timeout: int = 10):
        self.noco_url = noco_url.rstrip("/")
        self.api_headers = {
            "xc-token": api_token.strip(),
            "Content-Type": "application/json",
        }
        self.timeout = timeout

    def get_sources(self, base_id: str) -> List[Dict[str, Any]]:
        """
        Get all sources for a given base/project

        Args:
            base_id: The project/base ID

        Returns:
            List of source dictionaries
        """
        sources_url = f"{self.noco_url}/api/v2/meta/bases/{base_id}/sources"
        resp = requests.get(sources_url, headers=self.api_headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["list"]

    def get_first_source_id(self, base_id: str) -> str:
        """
        Get the first source ID for a given base/project

        Args:
            base_id: The project/base ID

        Returns:
            The first source ID
        """
        sources = self.get_sources(base_id)
        if not sources:
            raise ValueError(f"No sources found for base_id: {base_id}")

        source_id = sources[0]["id"]
        print(f"📌 Auto-detected SOURCE_ID: {source_id}")
        return source_id

    def create_table(
        self,
        project_id: str,
        source_id: str,
        table_title: str,
        table_name: str,
    ) -> str:
        """
        Create an empty table in NocoDB
        If table name exists (422 error), automatically append timestamp

        Returns:
            str: The created table ID
        """
        create_table_url = (
            f"{self.noco_url}/api/v1/db/meta/projects/"
            f"{project_id}/{source_id}/tables"
        )

        original_title = table_title
        original_name = table_name
        max_retries = 5

        for attempt in range(max_retries):
            create_table_payload = {
                "title": table_title,
                "table_name": table_name,
                "description": "",
                "is_hybrid": True,
                "columns": [],
            }

            resp = requests.post(
                create_table_url,
                headers=self.api_headers,
                json=create_table_payload,
                timeout=self.timeout,
            )

            if resp.status_code == 422:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                table_title = f"{original_title}_{timestamp}"
                table_name = f"{original_name}_{timestamp}"
                print(f"⚠️  Table '{original_name}' already exists. Retrying with: {table_name}")
                continue

            resp.raise_for_status()
            table_id = resp.json()["id"]

            if table_title != original_title:
                print(f"✅ Created table with updated name: {table_title} (ID: {table_id})")
            else:
                print(f"✅ Created table: {table_id}")

            return table_id

        raise Exception(f"Failed to create table after {max_retries} attempts")

    def get_table_schema(self, table_id: str) -> Dict[str, Any]:
        """Get table schema"""
        schema_url = f"{self.noco_url}/api/v1/db/meta/tables/{table_id}"
        resp = requests.get(schema_url, headers=self.api_headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def create_column(
        self,
        table_id: str,
        column_name: str,
        column_title: str,
        uidt: str = "LongText",
    ):
        """Create a new column in the table"""
        create_col_url = f"{self.noco_url}/api/v1/db/meta/tables/{table_id}/columns"

        payload = {
            "title": column_title,
            "column_name": column_name,
            "uidt": uidt,
        }

        resp = requests.post(create_col_url, headers=self.api_headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        print(f"Created column: {column_name}")

    def bulk_insert(
        self,
        org: str,
        project_id: str,
        table_id: str,
        data: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> None:
        """Bulk insert data into table in batches accepted by NocoDB."""
        bulk_url = f"{self.noco_url}/api/v1/db/data/bulk/{org}/{project_id}/{table_id}"
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        total = len(data)
        for start in range(0, total, batch_size):
            batch = data[start:start + batch_size]
            batch_num = start // batch_size + 1
            batch_total = (total + batch_size - 1) // batch_size

            resp = requests.post(
                bulk_url,
                headers=self.api_headers,
                json=batch,
                timeout=self.timeout,
            )

            print(f"Batch {batch_num}/{batch_total} status:", resp.status_code)
            if not resp.ok:
                print("Response:", resp.text)
                resp.raise_for_status()

    def get_tables(self, project_id: str) -> List[Dict[str, Any]]:
        """List all tables in a project"""
        url = f"{self.noco_url}/api/v1/db/meta/projects/{project_id}/tables"
        resp = requests.get(url, headers=self.api_headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["list"]

    def get_table_id_by_name(self, project_id: str, table_name: str) -> str:
        """Find a table ID by its title or name (case-insensitive)"""
        tables = self.get_tables(project_id)
        for t in tables:
            if t["title"].lower() == table_name.lower() or t["table_name"].lower() == table_name.lower():
                return t["id"]
        available = [t["title"] for t in tables]
        raise ValueError(f"Table '{table_name}' not found. Available tables: {available}")

    def delete_table(self, table_id: str) -> requests.Response:
        """Delete a table by its table ID."""
        url = f"{self.noco_url}/api/v1/db/meta/tables/{table_id}"
        resp = requests.delete(url, headers=self.api_headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp

    def insert_record(
        self,
        project_id: str,
        table_id: str,
        record: Dict[str, Any],
    ) -> requests.Response:
        """Insert a single record into an existing table"""
        url = f"{self.noco_url}/api/v1/db/data/noco/{project_id}/{table_id}"
        resp = requests.post(url, headers=self.api_headers, json=record, timeout=self.timeout)
        print("Status:", resp.status_code)
        print("Response:", resp.text)
        resp.raise_for_status()
        return resp

    def list_records(
        self,
        project_id: str,
        table_id: str,
        where: str = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """List records from a table, optionally filtered by a where clause."""
        url = f"{self.noco_url}/api/v1/db/data/noco/{project_id}/{table_id}"
        params = {"limit": limit}
        if where:
            params["where"] = where
        resp = requests.get(url, headers=self.api_headers, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("list", [])

    def update_record(
        self,
        project_id: str,
        table_id: str,
        row_id: int,
        record: Dict[str, Any],
    ) -> requests.Response:
        """Update an existing record by its row ID."""
        url = f"{self.noco_url}/api/v1/db/data/noco/{project_id}/{table_id}/{row_id}"
        resp = requests.patch(url, headers=self.api_headers, json=record, timeout=self.timeout)
        resp.raise_for_status()
        return resp

    def delete_record(
        self,
        project_id: str,
        table_id: str,
        row_id: int,
    ) -> requests.Response:
        """Delete an existing record by its row ID."""
        url = f"{self.noco_url}/api/v1/db/data/noco/{project_id}/{table_id}/{row_id}"
        resp = requests.delete(url, headers=self.api_headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp

    def upload_file(
        self,
        file_path: str,
        project_id: str,
        source_id: str,
        org: str,
        table_title: str = None,
        table_name: str = None,
    ) -> str:
        """
        Upload an Excel or CSV file to NocoDB as a new table.

        Auto-detects file format, creates the table and columns,
        then bulk inserts all rows.

        Args:
            file_path:    Path to .xlsx or .csv file.
            project_id:   NocoDB project/base ID.
            source_id:    NocoDB source ID.
            org:          Organisation name (default "noco").
            table_title:  Display name for the new table (defaults to filename).
            table_name:   DB name for the new table (defaults to filename).

        Returns:
            The created table ID.
        """
        import os
        from .data_processor import ExcelProcessor, normalize_name
        base = os.path.splitext(os.path.basename(file_path))[0]
        if table_title is None:
            table_title = base
        if table_name is None:
            table_name = normalize_name(base)
        else:
            table_name = normalize_name(table_name)

        # Step 1: Load file
        print(f"Loading file: {file_path}")
        processor = ExcelProcessor(file_path)
        processor.load_excel()

        # Step 2: Create table
        print(f"Creating table: {table_title}")
        table_id = self.create_table(project_id, source_id, table_title, table_name)

        # Step 3: Create columns
        schema = self.get_table_schema(table_id)
        existing_cols = {normalize_name(c["column_name"]) for c in schema["columns"]}
        for col in processor.get_columns():
            if col not in existing_cols:
                self.create_column(table_id, column_name=col, column_title=col, uidt="LongText")

        # Step 4: Get updated schema and prepare records
        schema = self.get_table_schema(table_id)
        valid_cols = {c["column_name"] for c in schema["columns"]}
        payload = processor.prepare_records(valid_cols)

        # Step 5: Bulk insert
        print(f"Inserting {len(payload)} rows...")
        self.bulk_insert(org, project_id, table_id, payload)
        print(f"Upload complete. Table ID: {table_id}")
        return table_id

    def clear_table(self, project_id: str, table_id: str) -> None:
        """Delete every row in a table."""
        while True:
            rows = self.list_records(project_id, table_id, limit=1000)
            if not rows:
                break
            for row in rows:
                self.delete_record(project_id, table_id, row["Id"])

    def ensure_table(
        self,
        project_id: str,
        table_name: str,
        row: Dict[str, Any],
        source_id: str = None,
        uidt_fn: Callable[[str, Any], str] = None,
    ) -> str:
        """
        Ensure a table exists with schema compatible with *row*.

        Creates the table if missing, recreates it if column types diverge,
        and adds any columns present in *row* but absent from the schema.
        Returns the table ID.
        """
        if uidt_fn is None:
            uidt_fn = _default_uidt
        try:
            table_id = self.get_table_id_by_name(project_id, table_name)
        except ValueError:
            sid = source_id or self.get_first_source_id(project_id)
            table_id = self.create_table(project_id, sid, table_name, table_name)
        else:
            schema = self.get_table_schema(table_id)
            if table_needs_recreation(schema, row, uidt_fn):
                self.delete_table(table_id)
                sid = source_id or self.get_first_source_id(project_id)
                table_id = self.create_table(project_id, sid, table_name, table_name)

        schema = self.get_table_schema(table_id)
        existing_cols = {c["column_name"].lower() for c in schema["columns"]}
        for col, value in row.items():
            if col.lower() not in existing_cols:
                self.create_column(table_id, column_name=col, column_title=col, uidt=uidt_fn(col, value))
        return table_id

    def replace_table_rows(
        self,
        project_id: str,
        org: str,
        *,
        table_name: str,
        rows: List[Dict[str, Any]],
        source_id: str = None,
        uidt_fn: Callable[[str, Any], str] = None,
        str_columns: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Replace all rows in a NocoDB table.

        Normalizes records, ensures the table and schema exist, clears existing
        rows, then bulk-inserts the new ones.  Returns a status dict.
        """
        from .data_processor import normalize_record

        if not rows:
            raise ValueError(f"No rows available for table {table_name}")

        normalized = [normalize_record(r, str_columns=str_columns) for r in rows]
        table_id = self.ensure_table(project_id, table_name, normalized[0], source_id=source_id, uidt_fn=uidt_fn)
        self.clear_table(project_id, table_id)
        self.bulk_insert(org, project_id, table_id, normalized)
        return {"status": "replaced", "table": table_name, "table_id": table_id, "row_count": len(normalized)}
