# storage/manager.py
from pathlib import Path
import logging
import os, shutil, threading
from typing import Optional

from grace_mem.storage.chroma_vdb import EntitiesVDB, RelationshipsVDB, SummariesVDB
from grace_mem.storage.bm25 import EntitiesBM25
from grace_mem.storage.cache import CacheStore
from grace_mem.storage.paths import resolve_artifacts_dir

logger = logging.getLogger(__name__)

class VDBManager:
    '''
    The single entry point for the vector stores and the cache -- the ChromaDB
    indexes for entities, relationships and summaries -- offering one shared
    asynchronous persist and reset flow.

    All files (ChromaDB directories, cache, BM25) are stored in the artifacts directory.
    '''
    def __init__(self, artifacts_dir: Path) -> None:
        """Set up artifact locations and lazy handles for all vector stores."""
        self.ART = artifacts_dir
        # parents=True: per-sample artifact paths are nested (e.g. <run>/sample_0/artifacts),
        # and callers should not have to pre-create the intermediate directories.
        self.ART.mkdir(parents=True, exist_ok=True)

        # ChromaDB directories
        self.ENT_CHROMA_DIR = self.ART / "entities_chroma"
        self.REL_CHROMA_DIR = self.ART / "relationships_chroma"
        self.SUM_CHROMA_DIR = self.ART / "summaries_chroma"

        # BM25 file
        self.ENT_BM25  = self.ART / "entities_bm25.pkl"

        # Cache files (now in artifacts directory)
        self.ENT_CACHE = self.ART / "entities_cache.pkl"
        self.REL_CACHE = self.ART / "relationships_cache.pkl"
        self.ENT_META = self.ART / "entities_meta.jsonl"
        self.REL_META = self.ART / "relationships_meta.jsonl"
        self.SUM_META = self.ART / "summaries_meta.jsonl"

        # Load cache from artifacts directory
        self.cache = CacheStore.load(cache_dir=self.ART)

        self._entities_vdb: Optional[EntitiesVDB] = None
        self._relationships_vdb: Optional[RelationshipsVDB] = None
        self._summaries_vdb: Optional[SummariesVDB] = None
        self._entities_bm25: Optional[EntitiesBM25] = None
        self._persist_lock = threading.Lock()
        self._persist_thread: Optional[threading.Thread] = None
        self._persist_error: Optional[Exception] = None

    def initialize(self) -> bool:
        """Report whether this was a fresh initialization (true when no files existed)."""
        ent_ok = self.ENT_CHROMA_DIR.exists()
        rel_ok = self.REL_CHROMA_DIR.exists()
        if ent_ok or rel_ok:
            # the vdb loads itself
            return False
        # Fresh start: clear the cache
        CacheStore.clear(self.cache)
        return True

    # ========== Entities ==========
    def get_entities_vdb(self, dim: int) -> EntitiesVDB:
        """Return the shared entities vector database, creating it if needed."""
        if self._entities_vdb is None:
            self._entities_vdb = EntitiesVDB(dim, str(self.ENT_CHROMA_DIR), "entities")
        return self._entities_vdb

    #NEW: Entities BM25
    def get_entities_bm25(self, load_if_empty: bool = False) -> EntitiesBM25:
        """Return the shared entities BM25 index, optionally loading it from disk."""
        if self._entities_bm25 is None:
            self._entities_bm25 = EntitiesBM25()
            if load_if_empty and self.ENT_BM25.exists():
                try:
                    self._entities_bm25.load(str(self.ENT_BM25))
                except Exception:
                    pass
        return self._entities_bm25

    # ========== Relationships ==========
    def get_relationships_vdb(self, dim: int) -> RelationshipsVDB:
        """Return the shared relationships vector database, creating it if needed."""
        if self._relationships_vdb is None:
            self._relationships_vdb = RelationshipsVDB(dim, str(self.REL_CHROMA_DIR), "relationships")
        return self._relationships_vdb

    def get_summaries_vdb(self, dim: int) -> SummariesVDB:
        """Return the shared summaries vector database, creating it if needed."""
        if self._summaries_vdb is None:
            self._summaries_vdb = SummariesVDB(dim, str(self.SUM_CHROMA_DIR), "summaries")
        return self._summaries_vdb

    # ========== Persist / Reset ==========
    def persist_async(self) -> None:
        """Start a background thread that persists the entity/relationship VDBs and the cache together."""
        def _task() -> None:
            """Persist every initialized index, metadata dump, and cache file."""
            try:
                if self._entities_vdb:
                    self._entities_vdb.save()
                    self._entities_vdb.export_metadatas_jsonl(str(self.ENT_META))
                if self._entities_bm25:
                    self._entities_bm25.save(str(self.ENT_BM25))
                if self._relationships_vdb:
                    self._relationships_vdb.save()
                    self._relationships_vdb.export_metadatas_jsonl(str(self.REL_META))
                if self._summaries_vdb:
                    self._summaries_vdb.save()
                    self._summaries_vdb.export_metadatas_jsonl(str(self.SUM_META))
                CacheStore.save(self.cache, cache_dir=self.ART)
            except Exception as exc:
                self._persist_error = exc
        with self._persist_lock:
            self._persist_error = None
            t = threading.Thread(target=_task, daemon=True)
            t.start()
            self._persist_thread = t

    def _wait_for_persist(self) -> None:
        """Wait for an in-flight persist and surface its failure."""
        with self._persist_lock:
            thread = self._persist_thread
        if thread is not None:
            thread.join(timeout=30)
            if thread.is_alive():
                raise RuntimeError(
                    "persist_async thread timed out - VDB state unknown"
                )
            self._persist_thread = None
        if self._persist_error:
            error = self._persist_error
            self._persist_error = None
            raise RuntimeError(f"Background persist failed: {error}") from error

    def flush_persist(self) -> None:
        """Wait for any in-flight persist thread, then persist synchronously."""
        self._wait_for_persist()
        # Synchronous persist to guarantee all files are on disk
        if self._entities_vdb:
            self._entities_vdb.save()
            self._entities_vdb.export_metadatas_jsonl(str(self.ENT_META))
        if self._entities_bm25:
            self._entities_bm25.save(str(self.ENT_BM25))
        if self._relationships_vdb:
            self._relationships_vdb.save()
            self._relationships_vdb.export_metadatas_jsonl(str(self.REL_META))
        if self._summaries_vdb:
            self._summaries_vdb.save()
            self._summaries_vdb.export_metadatas_jsonl(str(self.SUM_META))
        CacheStore.save(self.cache, cache_dir=self.ART)
        self.validate_artifacts()

    def validate_artifacts(self) -> None:
        """Raise RuntimeError if any persisted artifact file is missing or empty."""
        if (
            self._entities_vdb is None
            and self._relationships_vdb is None
            and self._summaries_vdb is None
            and self._entities_bm25 is None
        ):
            return

        errors: list[str] = []

        def _check_chroma(label: str, chroma_dir: Path) -> None:
            sqlite = chroma_dir / "chroma.sqlite3"
            if not sqlite.exists() or sqlite.stat().st_size == 0:
                errors.append(f"{label}/chroma.sqlite3 missing or empty")
                return
            uuid_dirs = [p for p in chroma_dir.iterdir() if p.is_dir()]
            if uuid_dirs:
                data_bin = uuid_dirs[0] / "data_level0.bin"
                if not data_bin.exists() or data_bin.stat().st_size == 0:
                    errors.append(f"{label}/<uuid>/data_level0.bin missing or empty")

        def _check_file(label: str, path: Path) -> None:
            if not path.exists() or path.stat().st_size == 0:
                errors.append(f"{label} missing or empty ({path.name})")

        if self._entities_vdb is not None:
            _check_chroma("entities_chroma", self.ENT_CHROMA_DIR)
            _check_file("entities_meta.jsonl", self.ENT_META)
        if self._relationships_vdb is not None:
            _check_chroma("relationships_chroma", self.REL_CHROMA_DIR)
            _check_file("relationships_meta.jsonl", self.REL_META)
        if self._summaries_vdb is not None:
            _check_chroma("summaries_chroma", self.SUM_CHROMA_DIR)
        if self._entities_bm25 is not None:
            _check_file("entities_bm25.pkl", self.ENT_BM25)

        _check_file("entities_cache.pkl", self.ENT_CACHE)
        _check_file("relationships_cache.pkl", self.REL_CACHE)

        if errors:
            raise RuntimeError(
                "Artifact validation failed after flush_persist:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def close(self, *, persist: bool = False, clear_cache: bool = False) -> None:
        """Release initialized vector-store clients and optional in-memory state."""
        first_error: Exception | None = None
        try:
            if persist:
                self.flush_persist()
            else:
                self._wait_for_persist()
        except Exception as exc:
            first_error = exc

        for vdb in (
            self._entities_vdb,
            self._relationships_vdb,
            self._summaries_vdb,
        ):
            if vdb is None:
                continue
            try:
                vdb.close()
            except Exception as exc:
                logger.warning("Failed to close vector store: %s", exc)
                if first_error is None:
                    first_error = exc

        self._entities_vdb = None
        self._relationships_vdb = None
        self._summaries_vdb = None
        self._entities_bm25 = None
        if clear_cache:
            try:
                CacheStore.clear(self.cache)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def reset_all(self, delete_files: bool = True) -> None:
        """Reset every VDB and the cache, optionally deleting the files as well."""
        # Close clients before deleting files on disk.
        self.close()

        # Delete files on disk (now safe - no open SQLite connections).
        if delete_files:
            import shutil
            # Delete ChromaDB directories
            if self.ENT_CHROMA_DIR.exists(): shutil.rmtree(self.ENT_CHROMA_DIR)
            if self.REL_CHROMA_DIR.exists(): shutil.rmtree(self.REL_CHROMA_DIR)
            if self.SUM_CHROMA_DIR.exists(): shutil.rmtree(self.SUM_CHROMA_DIR)

            # Delete other artifact files
            files_to_delete = [
                self.ENT_CACHE,
                self.ENT_BM25,
                self.REL_CACHE,
                self.ENT_META,
                self.REL_META,
                self.SUM_META,
            ]
            for p in files_to_delete:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

        # Clear cache in memory.
        CacheStore.clear(self.cache)
def _resolve_art_dir() -> Path:
    """Resolve and create the artifacts directory used by storage singletons.

    Honors KG_ARTIFACTS_DIR so parallel processes can be isolated onto separate
    working dirs (see grace_mem.storage.paths.resolve_artifacts_dir)."""
    return resolve_artifacts_dir(create=True)

# Singleton: every call site imports this one instance
ART_DIR = _resolve_art_dir()
MGR = VDBManager(ART_DIR)
