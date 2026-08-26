#!/usr/bin/env python3
"""
System Refresh Script
=====================
Performs a complete reset of the KG system:
1. Clears local VDB files and cache
2. Clears experiment/logs folder
3. Clears the FalkorDB graph database
4. Reinitializes the graph schema

Usage:
    python tools/refresh_system.py
"""

import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load environment variables from repo root .env file
ENV_PATH = REPO_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Allow `python tools/refresh_system.py` from any working directory.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grace_mem.storage import MGR
from grace_mem.storage.paths import resolve_artifacts_dir
from grace_mem.graph.falkordb import graph_from_env


def refresh_system():
    """Perform a complete system refresh."""
    print("\n" + "="*60)
    print("SYSTEM REFRESH - Complete Reset")
    print("="*60)

    artifacts_path = resolve_artifacts_dir()
    logs_path = Path("./logs")

    print(f"\nArtifacts directory: {artifacts_path}")
    print(f"Logs directory: {logs_path}")

    # 1) Clear local VDB files and cache
    print("\n[1/4] Clearing local VDB files and cache...")
    try:
        # Use the global manager to ensure in-process references are reset.
        MGR.reset_all(delete_files=True)
        print("Local VDB and cache cleared successfully")
    except Exception as e:
        print(f"Error clearing local files: {e}")
        raise

    # 2) Clear experiment/logs folder
    print("\n[2/4] Clearing experiment/logs folder...")
    try:
        if logs_path.exists():
            shutil.rmtree(logs_path)
            logs_path.mkdir(parents=True, exist_ok=True)
            print("Logs folder cleared successfully")
        else:
            print("Logs folder doesn't exist, skipping")
    except Exception as e:
        print(f"Error clearing logs: {e}")
        raise

    # 3) Clear FalkorDB graph database
    print("\n[3/4] Clearing FalkorDB graph database...")
    try:
        graph = graph_from_env().open()
        graph.clear_all()
        print("FalkorDB graph cleared successfully")
    except Exception as e:
        print(f"Error clearing FalkorDB: {e}")
        raise

    # 4) Reinitialize graph schema
    print("\n[4/4] Reinitializing graph schema...")
    try:
        graph.init_schema()
        print("Graph schema initialized successfully")
    except Exception as e:
        print(f"Error initializing schema: {e}")
        raise
    finally:
        graph.close()

    print("\n" + "="*60)
    print("System refresh completed successfully!")
    print("="*60 + "\n")


def main():
    """Main entry point with confirmation prompt."""
    print("\nWARNING: This will permanently delete:")
    print("   - All VDB files and cache (./grace_mem/storage/artifacts)")
    print("   - All log files (./experiment/logs)")
    print("   - All data in FalkorDB graph database")

    response = input("\nAre you sure you want to continue? (yes/no): ").strip().lower()
    if response not in ("yes", "y"):
        print("Operation cancelled")
        sys.exit(0)

    try:
        refresh_system()
    except Exception as e:
        print(f"\nRefresh failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
