# storage/cache.py
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import pickle, logging,os
logger = logging.getLogger(__name__)

# Backward compatibility: global cache directory (deprecated)
CACHE_DIR = Path("vdb_cache")
CACHE_DIR.mkdir(exist_ok=True)
ENT_FILE = CACHE_DIR / "entities_cache.pkl"
REL_FILE = CACHE_DIR / "relationships_cache.pkl"

class CacheStore:
    """
    提供實體/關係的快取管理：
    - load()  ：從檔案載入快取內容（entities / relationships）
    - save()  ：將目前快取寫入檔案（pickle 格式）
    - clear() ：清空記憶體中的快取

    Updated to support custom cache directories (per-dataset).
    """
    @staticmethod
    def load(cache_dir: Optional[Path] = None) -> Dict[str, Dict]:
        """
        Load cache from disk.

        Args:
            cache_dir: Custom cache directory. If None, uses global vdb_cache/ (deprecated).
        """
        if cache_dir is None:
            # Backward compatibility: use global cache
            ent_file = ENT_FILE
            rel_file = REL_FILE
        else:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(exist_ok=True, parents=True)
            ent_file = cache_dir / "entities_cache.pkl"
            rel_file = cache_dir / "relationships_cache.pkl"

        def _load(p: Path) -> Dict[str, Dict]:
            """Load a cached shard from disk and fall back to an empty mapping on error."""
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except FileNotFoundError:
                return {}
            except Exception as e:
                logger.error("Cache load failed: %s -> %s", p, e)
                return {}

        ent = _load(ent_file)
        rel = _load(rel_file)
        cache = {"entities":{}, "entities_full":{}, "relationships":{}, "relationships_full":{}}
        cache.update(ent)
        cache.update(rel)
        return cache

    @staticmethod
    def save(cache: Dict[str, Dict], cache_dir: Optional[Path] = None) -> None:
        """
        Save cache to disk.

        Args:
            cache: Cache dictionary to save
            cache_dir: Custom cache directory. If None, uses global vdb_cache/ (deprecated).
        """
        if cache_dir is None:
            # Backward compatibility: use global cache
            ent_file = ENT_FILE
            rel_file = REL_FILE
        else:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(exist_ok=True, parents=True)
            ent_file = cache_dir / "entities_cache.pkl"
            rel_file = cache_dir / "relationships_cache.pkl"

        def _dump(p: Path, obj: Dict[str, Dict]) -> None:
            """Write one cache shard to disk."""
            try:
                with open(p, "wb") as f:
                    pickle.dump(obj, f)
            except Exception as e:
                logger.error("Cache dump failed: %s -> %s", p, e)
                raise

        _dump(ent_file, {"entities": cache.get("entities", {}), "entities_full": cache.get("entities_full", {})})
        _dump(rel_file, {"relationships": cache.get("relationships", {}), "relationships_full": cache.get("relationships_full", {})})

    @staticmethod
    def clear(cache: Dict[str, Dict]) -> None:
        """Clear cache in memory (does not delete files)"""
        cache.get("entities", {}).clear()
        cache.get("entities_full", {}).clear()
        cache.get("relationships", {}).clear()
        cache.get("relationships_full", {}).clear()

    @staticmethod
    def reset(cache: Dict[str, Dict], cache_dir: Optional[Path] = None) -> None:
        """
        Clear cache in memory and delete cache files from disk.

        Args:
            cache: Cache dictionary to clear
            cache_dir: Custom cache directory. If None, uses global vdb_cache/ (deprecated).
        """
        # Clear memory
        CacheStore.clear(cache)

        # Delete files
        if cache_dir is None:
            # Backward compatibility: use global cache
            files_to_delete = (ENT_FILE, REL_FILE)
        else:
            cache_dir = Path(cache_dir)
            files_to_delete = (
                cache_dir / "entities_cache.pkl",
                cache_dir / "relationships_cache.pkl"
            )

        for p in files_to_delete:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

def build_id_to_meta_maps(cache: Dict[str, Dict]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Build entity and relationship ID lookup tables from the shared cache."""
    ent_id2meta, rel_id2meta = {}, {}
    for m in cache.get("entities", {}).values():
        if isinstance(m, dict) and m.get("id"): ent_id2meta[m["id"]] = m
    for m in cache.get("relationships", {}).values():
        if isinstance(m, dict) and m.get("id"): rel_id2meta[m["id"]] = m
    return ent_id2meta, rel_id2meta
