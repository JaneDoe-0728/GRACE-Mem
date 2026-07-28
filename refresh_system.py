#!/usr/bin/env python3
"""
System Refresh Script
=====================
Performs a complete reset of the KG system:
1. Clears local VDB files and cache
2. Clears experiment/logs folder
3. Clears Neo4j graph database
4. Reinitializes Neo4j schema

Usage:
    python refresh_system.py
"""

import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from repo root .env file
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

from KG.storage import MGR
from KG.storage.paths import resolve_artifacts_dir
# from KG.graph.neo4j import graph_from_env
from KG.graph.falkordb import graph_from_env


def refresh_system():
    """Perform a complete system refresh."""
    print("\n" + "="*60)
    print("⚠️  SYSTEM REFRESH - Complete Reset")
    print("="*60)

    artifacts_path = resolve_artifacts_dir()
    logs_path = Path("./logs")

    print(f"\n📁 Artifacts directory: {artifacts_path}")
    print(f"📁 Logs directory: {logs_path}")

    # 1) Clear local VDB files and cache
    print("\n[1/4] Clearing local VDB files and cache...")
    try:
        # Use the global manager to ensure in-process references are reset.
        MGR.reset_all(delete_files=True)
        print("✅ Local VDB and cache cleared successfully")
    except Exception as e:
        print(f"❌ Error clearing local files: {e}")
        raise

    # 2) Clear experiment/logs folder
    print("\n[2/4] Clearing experiment/logs folder...")
    try:
        if logs_path.exists():
            shutil.rmtree(logs_path)
            logs_path.mkdir(parents=True, exist_ok=True)
            print("✅ Logs folder cleared successfully")
        else:
            print("ℹ️  Logs folder doesn't exist, skipping")
    except Exception as e:
        print(f"❌ Error clearing logs: {e}")
        raise

    # 3) Clear Neo4j graph database
    print("\n[3/4] Clearing Neo4j graph database...")
    try:
        graph = graph_from_env().open()
        graph.clear_all()
        print("✅ Neo4j graph cleared successfully")
    except Exception as e:
        print(f"❌ Error clearing Neo4j: {e}")
        raise

    # 4) Reinitialize Neo4j schema
    print("\n[4/4] Reinitializing Neo4j schema...")
    try:
        graph.init_schema()
        print("✅ Neo4j schema initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing schema: {e}")
        raise
    finally:
        graph.close()

    print("\n" + "="*60)
    print("✅ System refresh completed successfully!")
    print("="*60 + "\n")


def main():
    """Main entry point with confirmation prompt."""
    print("\n⚠️  WARNING: This will permanently delete:")
    print("   - All VDB files and cache (./KG/storage/artifacts)")
    print("   - All log files (./experiment/logs)")
    print("   - All data in Neo4j graph database")

    response = input("\nAre you sure you want to continue? (yes/no): ").strip().lower()
    if response not in ("yes", "y"):
        print("❌ Operation cancelled")
        sys.exit(0)

    try:
        refresh_system()
    except Exception as e:
        print(f"\n❌ Refresh failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
