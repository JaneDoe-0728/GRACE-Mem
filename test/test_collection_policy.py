from pathlib import Path

from conftest import MANUAL_SCRIPT_NAMES


TEST_ROOT = Path(__file__).resolve().parent


def test_manual_collection_exclusions_are_explicit_existing_scripts():
    ignored_paths = [TEST_ROOT / name for name in MANUAL_SCRIPT_NAMES]

    assert len(MANUAL_SCRIPT_NAMES) == len(set(MANUAL_SCRIPT_NAMES))
    assert all(path.is_file() for path in ignored_paths)
    assert all(path.name.startswith("test_") for path in ignored_paths)
