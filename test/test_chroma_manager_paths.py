import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from KG.storage.chroma_manager import VDBManager


def test_vdb_manager_creates_nested_artifacts_dir(tmp_path):
    artifacts_dir = tmp_path / "nested" / "sample_0" / "artifacts"

    mgr = VDBManager(artifacts_dir)

    assert mgr.ART == artifacts_dir
    assert artifacts_dir.exists()
    assert artifacts_dir.is_dir()


def test_vdb_manager_close_releases_clients_and_is_idempotent(tmp_path):
    mgr = VDBManager(tmp_path / "artifacts")
    clients = [Mock(), Mock(), Mock()]
    mgr._entities_vdb, mgr._relationships_vdb, mgr._summaries_vdb = clients
    mgr._entities_bm25 = object()

    mgr.close(clear_cache=True)
    mgr.close(clear_cache=True)

    for client in clients:
        client.close.assert_called_once_with()
    assert mgr._entities_vdb is None
    assert mgr._relationships_vdb is None
    assert mgr._summaries_vdb is None
    assert mgr._entities_bm25 is None
    assert all(not values for values in mgr.cache.values())
