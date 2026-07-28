from .noco_client import NocoDBClient, table_needs_recreation
from .data_processor import normalize_record


def __getattr__(name: str):
    if name in {"ExcelProcessor", "normalize_name"}:
        from .data_processor import ExcelProcessor, normalize_name

        return {"ExcelProcessor": ExcelProcessor, "normalize_name": normalize_name}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def insert_record(table: str, record: dict, table_is_id: bool = False, dataset: str | None = None) -> dict:
    """
    Insert a single record into an existing NocoDB table.

    Reads NOCO_URL, API_TOKEN, PROJECT_ID from environment variables (or .env file).
    Auto-creates any missing columns before inserting.

    Args:
        table:        Table name (title) or table ID.
        record:       Dict of {column: value} to insert.
        table_is_id:  Set True if `table` is a table ID, not a name.

    Returns:
        The inserted row as a dict.

    Example:
        from src import insert_record
        insert_record("MyTable", {"name": "Alice", "age": "30", "city": "Bangkok"})
    """
    import os
    from dotenv import load_dotenv
    from config import NocoDBConfig

    load_dotenv()
    noco_config = NocoDBConfig.from_dataset(dataset) if dataset else NocoDBConfig.from_env()
    client = NocoDBClient(noco_config.noco_url, noco_config.api_token)

    table_id = table if table_is_id else client.get_table_id_by_name(noco_config.project_id, table)

    # Auto-create missing columns
    schema = client.get_table_schema(table_id)
    existing_cols = {c["column_name"].lower() for c in schema["columns"]}
    for key in record:
        if key.lower() not in existing_cols:
            client.create_column(table_id, column_name=key, column_title=key, uidt="LongText")

    resp = client.insert_record(noco_config.project_id, table_id, record)
    return resp.json()


def upload_file(
    file_path: str,
    table_title: str = None,
    table_name: str = None,
    dataset: str | None = None,
) -> str:
    """
    Upload an Excel or CSV file to NocoDB as a new table.

    Reads NOCO_URL, API_TOKEN, PROJECT_ID, ORG from environment variables (or .env file).
    Auto-fetches SOURCE_ID from PROJECT_ID.

    Args:
        file_path:    Path to .xlsx or .csv file.
        table_title:  Display name for the new table (defaults to filename).
        table_name:   DB name for the new table (defaults to filename).

    Returns:
        The created table ID.

    Example:
        from src import upload_file
        upload_file("data.xlsx", table_title="Sales 2024")
    """
    import os
    from dotenv import load_dotenv
    from config import NocoDBConfig

    load_dotenv()
    noco_config = NocoDBConfig.from_dataset(dataset) if dataset else NocoDBConfig.from_env()
    client = NocoDBClient(noco_config.noco_url, noco_config.api_token)
    source_id = noco_config.source_id or client.get_first_source_id(noco_config.project_id)

    return client.upload_file(
        file_path=file_path,
        project_id=noco_config.project_id,
        source_id=source_id,
        org=noco_config.org,
        table_title=table_title,
        table_name=table_name,
    )


__all__ = [
    "NocoDBClient",
    "ExcelProcessor",
    "normalize_name",
    "normalize_record",
    "table_needs_recreation",
    "insert_record",
    "upload_file",
]
