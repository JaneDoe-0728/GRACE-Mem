"""Single owner of every persistent store for one run.

Four stores hold a run's state -- entity, relationship, and summary vector
collections plus the BM25 index -- alongside the extraction cache. They are
gathered behind one manager because they must stay mutually consistent: an
entity written to the vector store but missing from BM25 is retrievable by one
half of hybrid search and invisible to the other, and the two would silently
disagree about what the corpus contains.

Everything is rooted at one `artifacts_dir`. That is what isolates concurrent
experiment samples from each other: each gets its own directory, so nothing
they write can collide, and a run's artifacts can be inspected or discarded as
a unit.

Stores are constructed lazily. A pipeline configured for summary-only
retrieval never touches the entity collection, and opening a Chroma index costs
real time at startup.
"""

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
    """Owns the vector stores, the BM25 index, and the cache for one run.

    Every path is derived from the artifacts directory passed in, so two
    managers with different directories share no state at all -- which is what
    lets the harness run samples in parallel within one process.

    Persistence is asynchronous: `persist_async` hands the write to a
    background thread so ingestion is not blocked by disk. The consequence is
    that a failure surfaces later, on the next call, via `_persist_error`
    rather than at the point of the write.
    """

    def __init__(self, artifacts_dir: Path) -> None:
        """Derive every store path from `artifacts_dir` and load the cache.

        Only the cache is read eagerly -- it is small, and ingestion needs it
        immediately to decide what can be skipped. The vector stores stay
        unopened until first use.
        """
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
        """Prepare the stores, reporting whether this is a cold start.

        Existence of a Chroma directory is the resume signal -- Chroma reloads
        its own contents, so nothing needs doing here.

        On a genuine cold start the cache is cleared, which looks redundant
        against an empty directory but is not: the cache is loaded from disk in
        `__init__` and may hold extractions from a previous run whose vector
        stores were since deleted. Keeping it would mean ingestion skips
        extraction for entities that no longer exist in any index, leaving the
        graph permanently short of them.

        Returns:
            True on a cold start, False when resuming existing artifacts.
        """
        ent_ok = self.ENT_CHROMA_DIR.exists()
        rel_ok = self.REL_CHROMA_DIR.exists()
        if ent_ok or rel_ok:
            return False
        CacheStore.clear(self.cache)
        return True

    # ========== Entities ==========
    def get_entities_vdb(self, dim: int) -> EntitiesVDB:
        """Return the shared entities vector database, creating it if needed."""
        if self._entities_vdb is None:
            self._entities_vdb = EntitiesVDB(dim, str(self.ENT_CHROMA_DIR), "entities")
        return self._entities_vdb

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
        """Persist every initialized store on a background thread.

        Ingestion calls this after each turn, and a synchronous write there
        would put disk latency on the critical path of thousands of turns.

        Two consequences the caller has to respect. The thread is a daemon, so
        an interpreter exit will not wait for it -- `flush_persist` before
        shutdown or lose the tail of a run. And an exception is captured into
        `_persist_error` rather than raised, surfacing only at the next
        `_wait_for_persist`; a run that never waits never learns that its
        writes failed.
        """
        def _task() -> None:
            """Save each initialized store, then the cache last.

            Cache last on purpose: it records what has been extracted, so if
            the process dies mid-persist a cache that lags the indexes causes
            re-extraction (wasteful but correct), whereas a cache ahead of them
            would skip work whose output was never written.
            """
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
        """Join any in-flight persist thread and re-raise what it swallowed.

        The 30-second join is bounded rather than indefinite because a hung
        Chroma write would otherwise hang the run with no diagnostic. Timing
        out is reported as an error in its own right: the on-disk state is
        genuinely unknown at that point, and continuing would build on it.

        Raises:
            RuntimeError: If the persist thread timed out, or if the background
                write failed. The stored error is cleared either way, so it is
                reported once.
        """
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
        """Block until everything is on disk. Call before reading artifacts.

        Waits for the in-flight async persist and then writes again
        synchronously. The second write is not redundant: the background thread
        may have started before the most recent mutations, so joining it alone
        does not establish that current state was saved.
        """
        self._wait_for_persist()
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
            """Verify a Chroma directory is present and non-empty.

            An empty directory is the signature of a failed or interrupted persist: it
            exists, so the resume path treats it as valid, and the run then evaluates
            against an index holding nothing.
            """
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
