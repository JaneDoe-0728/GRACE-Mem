"""On-disk cache of extracted entities and relationships.

Extraction is the expensive half of ingestion: every turn costs an LLM call,
and re-running a sample to tweak a retrieval parameter would pay for all of
them again. This cache makes ingestion resumable, so retrieval experiments can
iterate against a corpus that was extracted once.

Entities and relationships are pickled to separate files, which lets a partial
write fail without corrupting both halves. Each is stored twice -- a compact
`*` map for lookup and a `*_full` map retaining the complete record -- because
retrieval only needs the former and loading the latter on the hot path was
measurably slower.

`cache_dir` should always be passed. The module-level CACHE_DIR default is a
single shared directory, so two datasets run without it silently overwrite each
other's extractions.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

from grace_mem.utils.atomic_write import atomic_write

logger = logging.getLogger(__name__)

# Deprecated process-wide fallback for the legacy no-argument path; pass an
# explicit cache_dir instead. The directory is created when that path is
# actually taken, not at import: importing a module must not write to whatever
# happens to be the caller's working directory, and it used to fail outright
# under a read-only cwd.
CACHE_DIR = Path("vdb_cache")
ENTITY_CACHE_FILE = CACHE_DIR / "entities_cache.pkl"
RELATIONSHIP_CACHE_FILE = CACHE_DIR / "relationships_cache.pkl"


def _resolve_cache_files(cache_dir: Path | str | None) -> tuple[Path, Path]:
    """Return (entity_file, relationship_file), creating the directory."""
    directory = CACHE_DIR if cache_dir is None else Path(cache_dir)
    directory.mkdir(exist_ok=True, parents=True)
    return (directory / "entities_cache.pkl", directory / "relationships_cache.pkl")

class CacheStore:
    """Load, save, and clear the entity/relationship extraction cache.

    Stateless: the cache dict is owned by the caller and passed in. That keeps
    a per-sample cache genuinely isolated, which matters because the experiment
    harness runs samples concurrently in one process.
    """

    @staticmethod
    def load(cache_dir: Path | None = None) -> dict[str, dict]:
        """Read the cache from disk, returning empty maps if absent.

        A missing or unreadable cache is not an error -- it is a cold start, so
        the caller gets the four expected keys either way and ingestion simply
        re-extracts. A corrupt shard is logged and treated the same, on the
        grounds that re-extraction is always correct and a half-decoded pickle
        is not.

        Args:
            cache_dir: Per-dataset cache directory, created if absent. None
                falls back to the shared vdb_cache/ (deprecated).

        Returns:
            A dict with keys entities, entities_full, relationships,
            relationships_full -- always all four, even on a cold start.
        """
        entity_cache_file, relationship_cache_file = _resolve_cache_files(cache_dir)

        def _load(p: Path) -> dict[str, dict]:
            """Load a cached shard from disk and fall back to an empty mapping on error."""
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except FileNotFoundError:
                return {}
            except Exception as e:
                logger.error("Cache load failed: %s -> %s", p, e)
                return {}

        ent = _load(entity_cache_file)
        rel = _load(relationship_cache_file)
        cache: dict[str, dict] = {
            "entities": {},
            "entities_full": {},
            "relationships": {},
            "relationships_full": {},
        }
        cache.update(ent)
        cache.update(rel)
        return cache

    @staticmethod
    def save(cache: dict[str, dict], cache_dir: Path | None = None) -> None:
        """Pickle the cache to disk as two shards.

        Unlike `load`, a write failure re-raises after logging: a cache that
        silently failed to persist would look like a cold start on the next
        run, and the whole extraction would be paid for twice with no
        indication why.

        Args:
            cache: The four-key cache dict. Missing keys are written as empty.
            cache_dir: Per-dataset cache directory, created if absent. None
                falls back to the shared vdb_cache/ (deprecated).

        Raises:
            Exception: Whatever pickling or the filesystem raised.
        """
        entity_cache_file, relationship_cache_file = _resolve_cache_files(cache_dir)

        def _dump(p: Path, obj: dict[str, dict]) -> None:
            """Write one cache shard to disk, atomically.

            `load` treats an unreadable shard as a cold start, so a truncated
            pickle costs a full re-extraction rather than raising -- expensive
            and silent, which is the worst combination.
            """
            try:
                with atomic_write(p, "wb") as f:
                    pickle.dump(obj, f)
            except Exception as e:
                logger.error("Cache dump failed: %s -> %s", p, e)
                raise

        _dump(entity_cache_file, {"entities": cache.get("entities", {}), "entities_full": cache.get("entities_full", {})})
        _dump(relationship_cache_file, {"relationships": cache.get("relationships", {}), "relationships_full": cache.get("relationships_full", {})})

    @staticmethod
    def clear(cache: dict[str, dict]) -> None:
        """Empty the cache in memory, leaving the files on disk.

        Clears each map in place rather than rebinding, because callers hold
        references to the same dict -- reassigning would leave them looking at
        the old contents. The next `load` restores from disk; use `reset` to
        discard the files too.
        """
        cache.get("entities", {}).clear()
        cache.get("entities_full", {}).clear()
        cache.get("relationships", {}).clear()
        cache.get("relationships_full", {}).clear()

def build_id_to_meta_maps(cache: dict[str, dict]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Invert the cache into id -> metadata lookups.

    The cache is keyed by normalized name, but retrieval works in ids: graph
    traversal and vector search both return ids, and the evidence stage needs
    the metadata behind each. Building both maps once per query beats a linear
    scan per lookup.

    Records without an `id` are skipped rather than raising -- an entity that
    failed to sync to the graph has no id yet, and it should drop out of
    id-based lookup rather than abort the query.

    Returns:
        (entity_id -> meta, relationship_id -> meta). Values alias the cached
        dicts; mutating one mutates the cache.
    """
    ent_id2meta, rel_id2meta = {}, {}
    for m in cache.get("entities", {}).values():
        if isinstance(m, dict) and m.get("id"): ent_id2meta[m["id"]] = m
    for m in cache.get("relationships", {}).values():
        if isinstance(m, dict) and m.get("id"): rel_id2meta[m["id"]] = m
    return ent_id2meta, rel_id2meta
