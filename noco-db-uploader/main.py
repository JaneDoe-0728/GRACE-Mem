#!/usr/bin/env python3
"""
NocoDB Uploader — Example Usage

Demonstrates three ways to push data into NocoDB:
  1. Insert a single record into an existing table
  2. Upload an Excel file as a new table
  3. Upload a CSV file as a new table
"""
from src import insert_record, upload_file


def example_insert_record():
    """Insert a single record into an existing NocoDB table."""
    print("=" * 50)
    print("Example 1: Insert a single record")
    print("=" * 50)
    row = insert_record(
        "Table-test",
        {"name": "Charlie", "age": "28", "city": "Chiang Mai"},
    )
    print(f"Inserted row: {row}\n")


def example_upload_excel():
    """Upload the AORUS MASTER 16 Excel file as a new NocoDB table."""
    print("=" * 50)
    print("Example 2: Upload Excel file")
    print("=" * 50)
    table_id = upload_file(
        "test_data/AORUS MASTER 16 BXH new .xlsx",
        table_title="AORUS MASTER 16",
    )
    print(f"Created table ID: {table_id}\n")


def example_upload_csv():
    """Upload the motherboard comparison CSV file as a new NocoDB table."""
    print("=" * 50)
    print("Example 3: Upload CSV file")
    print("=" * 50)
    table_id = upload_file(
        "test_data/motherboard_comparison_extended.csv",
        table_title="Motherboard Comparison",
    )
    print(f"Created table ID: {table_id}\n")


if __name__ == "__main__":
    example_insert_record()
    example_upload_excel()
    example_upload_csv()
