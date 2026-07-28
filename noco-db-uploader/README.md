# NocoDB Uploader

Upload Excel/CSV files or insert individual records into NocoDB via API — no GUI required.

## Features

- Insert a single record directly into any existing table
- Upload a full Excel/CSV file as a new table (auto-creates table and columns)
- Auto-creates missing columns before inserting
- Auto-fetches `SOURCE_ID` from `PROJECT_ID`
- Usable as a CLI tool or imported as a Python function

## Project Structure

```
noco-db-uploader/
├── config/
│   ├── __init__.py
│   ├── config.py          # Configuration classes
│   └── targets.py         # Optional dataset->target resolver
├── src/
│   ├── __init__.py        # insert_record() and upload_file() convenience functions
│   ├── noco_client.py     # NocoDB API client
│   └── data_processor.py  # Excel/CSV processing logic
├── main.py                # Bulk upload entry point (Excel/CSV → new table)
├── insert.py              # CLI for inserting a single record
├── pyproject.toml
└── .env
```

## Installation

This project uses [UV](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Configuration

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

```env
NOCO_URL=http://localhost:8081
API_TOKEN=your_api_token_here
ORG=noco
PROJECT_ID=your_project_id

# Only needed for bulk Excel upload via main.py:
TABLE_TITLE=my_table
TABLE_NAME=my_table
XLSX_PATH=/path/to/file.xlsx
```

**Getting your API token:**
- Login to NocoDB → Account Settings → Generate API Token

**Getting PROJECT_ID:**
- Open your NocoDB project → copy the ID from the browser URL

---

## Usage

### 1. Insert a single record (CLI)

```bash
# By column=value pairs
uv run insert.py --table "MyTable" name=Alice age=30 city=Bangkok

# By JSON
uv run insert.py --table "MyTable" --json '{"name": "Alice", "age": "30"}'

# By table ID directly (skips name lookup)
uv run insert.py --table-id "m9msnf6e4qq819e" name=Alice age=30
```

Missing columns are created automatically.

---

### 2. Insert a single record (Python function)

```python
from src import insert_record

# Insert into a table by name
row = insert_record("MyTable", {"name": "Alice", "age": "30", "city": "Bangkok"})
print(row)
# {"Id": 1, "name": "Alice", "age": "30", "city": "Bangkok", ...}

# Insert using a table ID directly
row = insert_record("m9msnf6e4qq819e", {"name": "Bob"}, table_is_id=True)
```

Reads `NOCO_URL`, `API_TOKEN`, and `PROJECT_ID` from `.env` automatically.
Missing columns are created automatically.

If you use dataset-based routing, point `NOCO_TARGETS_PATH` at a YAML file owned by
your application/repository.

---

### 3. Upload Excel / CSV (Python function)

```python
from src import upload_file

# Upload Excel — table name defaults to filename
upload_file("sales.xlsx")

# Upload CSV with a custom table name
upload_file("customers.csv", table_title="Customers 2024")
```

Reads `NOCO_URL`, `API_TOKEN`, `PROJECT_ID`, and `ORG` from `.env` automatically.
Creates a new table, adds all columns, and bulk inserts all rows.

---

### 4. Upload Excel / CSV (CLI)

```bash
# Set XLSX_PATH, TABLE_TITLE, TABLE_NAME in .env then run:
uv run main.py
```

---

## Troubleshooting

**Columns not showing in NocoDB UI after insert**
- Click **Fields** in the toolbar → enable the new columns (they may be hidden by default)

**Connection timeout**
- Verify `NOCO_URL` is correct and reachable (check hostname and port)
- Make sure you are on the correct network

**API token error**
- Ensure there is no leading/trailing space around the token in `.env`
