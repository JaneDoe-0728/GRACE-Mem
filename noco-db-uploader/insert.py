#!/usr/bin/env python3
"""
Insert a single record into an existing NocoDB table.

Usage examples:
  # Pass data as key=value pairs
  python insert.py --table "MyTable" key1=value1 key2=value2

  # Pass data as a JSON string
  python insert.py --table "MyTable" --json '{"name": "Alice", "age": "30"}'

  # Use table ID directly instead of table name
  python insert.py --table-id "md_xxxxx" key1=value1
"""
import argparse
import json
import os
import sys
from dotenv import load_dotenv

from config import NocoDBConfig
from src import NocoDBClient


def parse_args():
    parser = argparse.ArgumentParser(description="Insert a single record into a NocoDB table")

    table_group = parser.add_mutually_exclusive_group(required=True)
    table_group.add_argument("--table", metavar="TABLE_NAME", help="Table title/name to insert into")
    table_group.add_argument("--table-id", metavar="TABLE_ID", help="Table ID to insert into directly")

    parser.add_argument("--json", metavar="JSON_STRING", help='Record as JSON, e.g. \'{"col": "value"}\'')
    parser.add_argument("--dataset", metavar="DATASET", help="Optional dataset family for YAML-based target routing")
    parser.add_argument("fields", nargs="*", metavar="key=value", help="Record fields as key=value pairs")

    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    noco_config = NocoDBConfig.from_dataset(args.dataset) if args.dataset else NocoDBConfig.from_env()
    if not noco_config.api_token:
        sys.exit("Error: API_TOKEN is required")
    if not noco_config.project_id:
        sys.exit("Error: PROJECT_ID is required")

    # Build record dict
    if args.json:
        try:
            record = json.loads(args.json)
        except json.JSONDecodeError as e:
            sys.exit(f"Error: Invalid JSON -- {e}")
    else:
        record = {}
        for field in (args.fields or []):
            if "=" not in field:
                sys.exit(f"Error: Expected key=value format, got: {field!r}")
            k, _, v = field.partition("=")
            record[k] = v

    if not record:
        sys.exit("Error: No data provided. Use key=value pairs or --json '{...}'")

    client = NocoDBClient(noco_config.noco_url, noco_config.api_token)

    # Resolve table ID
    if args.table_id:
        table_id = args.table_id
    else:
        print(f"Looking up table '{args.table}'...")
        table_id = client.get_table_id_by_name(noco_config.project_id, args.table)
        print(f"Found table ID: {table_id}")

    # Ensure all columns exist in the table, create any that are missing
    schema = client.get_table_schema(table_id)
    existing_cols = {c["column_name"].lower() for c in schema["columns"]}
    for key in record:
        if key.lower() not in existing_cols:
            print(f"Column '{key}' not found — creating it...")
            client.create_column(table_id, column_name=key, column_title=key, uidt="LongText")

    print(f"Inserting record: {record}")
    client.insert_record(noco_config.project_id, table_id, record)
    print("Record inserted successfully.")


if __name__ == "__main__":
    main()
