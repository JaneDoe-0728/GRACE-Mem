import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from KG.storage.chroma_manager import VDBManager


def test_vdb_manager_creates_nested_artifacts_dir(tmp_path):
    artifacts_dir = tmp_path / "nested" / "sample_0" / "artifacts"

    mgr = VDBManager(artifacts_dir)

    assert mgr.ART == artifacts_dir
    assert artifacts_dir.exists()
    assert artifacts_dir.is_dir()
