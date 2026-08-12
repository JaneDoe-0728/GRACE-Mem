import chromadb
import uuid
import os
import json
import threading
import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from grace_mem.embeddings import embedder
from grace_mem.utils.logger_config import make_module_jlog

_debug_jlog = make_module_jlog(
    name="grace_mem.Storage.ChromaVDB",
    filename="kg_chroma_debug.jsonl",
)

def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    """
    Normalizes a batch of vectors to have unit L2 norm.
    """
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12  # Avoid division by zero
    return vectors / norm


def _serialize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize nested metadata values to JSON strings for ChromaDB compatibility.
    ChromaDB only accepts str, int, float, bool, or None as metadata values.
    """
    result = {}
    for k, v in meta.items():
        if isinstance(v, (dict, list)):
            result[k] = json.dumps(v)
        else:
            result[k] = v
    return result


def _deserialize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deserialize JSON string metadata values back to their original types.
    """
    if meta is None:
        return None
    result = {}
    for k, v in meta.items():
        if isinstance(v, str) and v.startswith('{') or isinstance(v, str) and v.startswith('['):
            try:
                result[k] = json.loads(v)
            except json.JSONDecodeError:
                result[k] = v
        else:
            result[k] = v
    return result

class SimpleChromaVDB:
    """
    A simple vector database using ChromaDB with an interface similar to SimpleFAISSVDB.
    It supports add, search, compare_by_id, delete, and update operations.
    """
    def __init__(self, dim: int, path: str, collection_name: str) -> None:
        """Open or create the Chroma collection backing this vector store."""
        self.dim = dim
        self.path = path
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=self.path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "ip",
                "hnsw:num_threads": 1,  # single-threaded search for deterministic results
            },
        )
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        """Return the current number of stored vectors in the collection."""
        with self._lock:
            return self._collection.count()

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        """Insert deduplicated vectors and metadata into the collection."""
        if not metadatas:
            return

        assert vectors.shape[0] == len(metadatas)
        assert vectors.shape[1] == self.dim

        # Deduplicate entries by ID, preferring longer provenance (same logic as search dedup)
        dedup_map = {}  # id -> (index, prov_len)
        for i, meta in enumerate(metadatas):
            mid = meta.get("id") or str(uuid.uuid4())
            prov_len = len((meta.get("prov") or {}).get("events", []))
            prev = dedup_map.get(mid)
            if prev is None or prov_len > prev[1]:
                dedup_map[mid] = (i, prov_len)

        # Collect deduplicated entries
        dedup_indices = [idx for idx, _ in dedup_map.values()]
        dedup_vectors = vectors[dedup_indices]
        dedup_metadatas = [metadatas[i] for i in dedup_indices]
        dedup_ids = [m.get("id") or str(uuid.uuid4()) for m in dedup_metadatas]

        # Serialize nested metadata values for ChromaDB compatibility
        serialized_metadatas = [_serialize_metadata(m) for m in dedup_metadatas]

        normalized_vectors = normalize_l2(dedup_vectors.astype(np.float32))

        with self._lock:
            # Using upsert to handle both add and update scenarios for simplicity
            self._collection.upsert(
                ids=dedup_ids,
                embeddings=normalized_vectors.tolist(),
                metadatas=serialized_metadatas
            )

    def upsert(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        """Alias add() so callers can use an upsert-style API."""
        self.add(vectors, metadatas)

    def search(self, query_vec: np.ndarray, top_k: int = 5, threshold: Optional[float] = None
               ) -> List[Tuple[Dict[str, Any], float]]:
        """Search the collection by cosine similarity and deduplicate repeated IDs."""
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        
        normalized_query = normalize_l2(query_vec.astype(np.float32))

        with self._lock:
            results = self._collection.query(
                query_embeddings=normalized_query.tolist(),
                n_results=top_k,
            )

        hits = []
        if results and results['ids']:
            for i in range(len(results['ids'][0])):
                # Chroma with 'ip' space returns 1 - inner_product. We convert it back.
                score = 1.0 - results['distances'][0][i]
                if threshold is None or score >= threshold:
                    # Deserialize metadata back to original structure
                    meta = _deserialize_metadata(results['metadatas'][0][i])
                    hits.append((meta, float(score)))

        # Deduplication logic from SimpleFAISSVDB
        filtered = {}
        for meta, score in hits:
            mid = meta.get("id")
            if not mid:
                continue
            prov_len = len((meta.get("prov") or {}).get("events", []))
            prev = filtered.get(mid)
            if not prev:
                filtered[mid] = (meta, score, prov_len)
            else:
                _, prev_score, prev_len = prev
                if (prov_len, score) > (prev_len, prev_score):
                    filtered[mid] = (meta, score, prov_len)
        return sorted([(m, s) for (m, s, _) in filtered.values()], key=lambda x: x[1], reverse=True)

    def batch_search(
        self, query_vecs: np.ndarray, top_k: int = 5, threshold: Optional[float] = None
    ) -> List[List[Tuple[Dict[str, Any], float]]]:
        """Batch vector search: one ChromaDB query for all N entity vectors.

        Returns a list (length N) where each element is the per-entity result
        in the same format as ``search()``.
        """
        if query_vecs.ndim == 1:
            query_vecs = query_vecs[None, :]

        n = query_vecs.shape[0]
        normalized = normalize_l2(query_vecs.astype(np.float32))

        with self._lock:
            results = self._collection.query(
                query_embeddings=normalized.tolist(),
                n_results=top_k,
            )

        out: List[List[Tuple[Dict[str, Any], float]]] = []
        for qi in range(n):
            hits = []
            if results and results.get("ids") and qi < len(results["ids"]):
                for j in range(len(results["ids"][qi])):
                    score = 1.0 - results["distances"][qi][j]
                    if threshold is None or score >= threshold:
                        meta = _deserialize_metadata(results["metadatas"][qi][j])
                        hits.append((meta, float(score)))

            filtered: Dict[str, Tuple[Dict[str, Any], float, int]] = {}
            for meta, score in hits:
                mid = meta.get("id")
                if not mid:
                    continue
                prov_len = len((meta.get("prov") or {}).get("events", []))
                prev = filtered.get(mid)
                if not prev:
                    filtered[mid] = (meta, score, prov_len)
                else:
                    _, prev_score, prev_len = prev
                    if (prov_len, score) > (prev_len, prev_score):
                        filtered[mid] = (meta, score, prov_len)
            out.append(sorted([(m, s) for (m, s, _) in filtered.values()], key=lambda x: x[1], reverse=True))
        return out

    def compare_by_id(
        self,
        mid: str,
        query_vec: np.ndarray,
        threshold: float = 0.0
    ) -> Optional[Tuple[Dict[str, Any], float]]:
        """Fetch one stored vector by ID and compare it to the query vector."""
        
        if query_vec.ndim == 1:
            qv = query_vec[None, :]
        else:
            qv = query_vec
            
        normalized_qv = normalize_l2(qv.astype(np.float32))

        with self._lock:
            res = self._collection.get(ids=[mid], include=['embeddings', 'metadatas'])
        
        if not res or not res['ids']:
            return None

        stored_vec = np.array(res['embeddings'][0])
        target_meta = _deserialize_metadata(res['metadatas'][0])
        
        # Stored vectors are already normalized
        score = float(np.dot(stored_vec, normalized_qv[0]))

        return (target_meta, score) if score >= threshold else None

    def compare_by_id_raw(
        self,
        mid: str,
        query_vec: np.ndarray,
        request_id: Optional[str] = None,
        debug_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """Fetch one stored vector by ID and return raw cosine score without threshold."""
        if query_vec.ndim == 1:
            qv = query_vec[None, :]
        else:
            qv = query_vec
        normalized_qv = normalize_l2(qv.astype(np.float32))
        with self._lock:
            res = self._collection.get(ids=[mid], include=['embeddings'])
        if not res or not res['ids']:
            with self._lock:
                collection_count = self._collection.count()
                retry_res = self._collection.get(ids=[mid], include=['embeddings'])

            retry_hit = bool(retry_res and retry_res.get('ids'))
            debug_payload = {
                "lookup_id": mid,
                "collection_name": self.collection_name,
                "vdb_path": self.path,
                "collection_count": collection_count,
                "retry_hit": retry_hit,
                "query_dim": int(normalized_qv.shape[1]),
            }
            if debug_context:
                debug_payload.update(debug_context)
            _debug_jlog("compare_by_id_raw_miss", request_id, **debug_payload)

            if not retry_hit:
                return None

            stored_vec = np.array(retry_res['embeddings'][0])
            retry_score = float(np.dot(stored_vec, normalized_qv[0]))
            _debug_jlog(
                "compare_by_id_raw_retry_hit",
                request_id,
                lookup_id=mid,
                collection_name=self.collection_name,
                vdb_path=self.path,
                score=retry_score,
                **(debug_context or {}),
            )
            return retry_score
        stored_vec = np.array(res['embeddings'][0])
        return float(np.dot(stored_vec, normalized_qv[0]))

    def delete(self, ids: List[str]) -> None:
        """
        Deletes entries by their IDs.
        """
        if not ids:
            return
        with self._lock:
            self._collection.delete(ids=ids)

    def update(self, ids: List[str], vectors: Optional[np.ndarray] = None, metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Updates entries by their IDs. Can update vectors, metadatas, or both.
        """
        if not ids:
            return

        embeddings_list = None
        if vectors is not None:
            assert vectors.shape[0] == len(ids)
            assert vectors.shape[1] == self.dim
            normalized_vectors = normalize_l2(vectors.astype(np.float32))
            embeddings_list = normalized_vectors.tolist()

        # Serialize nested metadata values for ChromaDB compatibility
        serialized_metadatas = None
        if metadatas is not None:
            serialized_metadatas = [_serialize_metadata(m) for m in metadatas]

        with self._lock:
            self._collection.update(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=serialized_metadatas
            )

    def rebuild(self, all_vectors: np.ndarray, all_metadatas: List[Dict[str, Any]]) -> None:
        """Replace the collection contents with a complete new snapshot of vectors."""
        assert all_vectors.shape[0] == len(all_metadatas)
        assert all_vectors.shape[1] == self.dim
        
        with self._lock:
            # Clear the collection
            self._client.delete_collection(name=self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "ip", "hnsw:num_threads": 1},
            )
            # Add all data
            if len(all_metadatas) > 0:
                self.add(all_vectors, all_metadatas)

    def save(self) -> None:
        """Provide a compatibility no-op because PersistentClient saves automatically."""
        # ChromaDB with PersistentClient persists automatically. This method is for API compatibility.
        pass

    def close(self) -> None:
        """Release the underlying ChromaDB PersistentClient and its SQLite connections."""
        with self._lock:
            if self._client is not None:
                # 1) Evict from ChromaDB's SharedSystemClient class-level cache
                #    so the next PersistentClient(path=...) creates a fresh system.
                try:
                    ident = getattr(self._client, '_identifier', None)
                    cache = getattr(type(self._client), '_identifier_to_system', None)
                    if ident and isinstance(cache, dict):
                        cache.pop(ident, None)
                except Exception:
                    pass
                # 2) Stop the system to release SQLite connections
                try:
                    sys_obj = getattr(self._client, '_system', None)
                    if sys_obj is not None and hasattr(sys_obj, 'stop'):
                        sys_obj.stop()
                except Exception:
                    pass
            self._collection = None
            self._client = None

    def load(self) -> None:
        """Provide a compatibility no-op because PersistentClient loads automatically."""
        # ChromaDB with PersistentClient loads automatically. This method is for API compatibility.
        pass

    def export_metadatas_jsonl(self, output_path: str) -> int:
        """
        Export all metadatas to a jsonl file (FAISS-compatible meta dump).
        Returns the number of rows written.
        """
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        tmp_path = output_path + ".tmp"

        with self._lock:
            results = self._collection.get(include=["metadatas"])

        ids = results.get("ids") or []
        metas = results.get("metadatas") or []

        with open(tmp_path, "w", encoding="utf-8") as f:
            for mid, meta in zip(ids, metas):
                meta = _deserialize_metadata(meta) if meta is not None else {}
                if "id" not in meta:
                    meta["id"] = mid
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        os.replace(tmp_path, output_path)
        return len(ids)


class EntitiesVDB(SimpleChromaVDB):
    pass

class RelationshipsVDB(SimpleChromaVDB):
    pass

class SummariesVDB(SimpleChromaVDB):
    """
    Stores and retrieves dialogue summaries using a vector database.
    """
    def add_summary(self, session_id: int, message_id: int, summary_text: str, dialogue_datetime: Optional[str] = None, raw_text: Optional[str] = None) -> str:
        """Embed and store one dialogue summary with its turn metadata."""
        vec = embedder.embed([summary_text])
        summary_id = f"{session_id}:{message_id}"
        meta = {
            "id": summary_id,
            "summary_id": summary_id,
            "session_id": session_id,
            "message_id": message_id,
            "summary_text": summary_text,
            "ts": datetime.datetime.utcnow().isoformat()
        }
        if dialogue_datetime is not None:
            meta["dialogue_datetime"] = dialogue_datetime
        if raw_text is not None:
            meta["raw_text"] = raw_text

        self.add(vec, [meta])
        return meta["summary_id"]

    def add_split_turns(
        self,
        session_id: int | str,
        message_id: int,
        user_text: str,
        assistant_summary: str,
        dialogue_datetime: Optional[str] = None,
    ) -> None:
        """Store two VDB entries per turn: {id}:u (user raw) and {id}:a (assistant compressed)."""
        base_id = f"{session_id}:{message_id}"
        ts = datetime.datetime.utcnow().isoformat()
        for suffix, text, role in (("u", user_text, "user"), ("a", assistant_summary, "assistant")):
            entry_id = f"{base_id}:{suffix}"
            vec = embedder.embed([text])
            meta: dict = {
                "id": entry_id,
                "summary_id": base_id,
                "session_id": session_id,
                "message_id": message_id,
                "text": text,
                "role": role,
                "ts": ts,
            }
            if dialogue_datetime is not None:
                meta["dialogue_datetime"] = dialogue_datetime
            self.add(vec, [meta])

    def get_text_by_entry_id(self, entry_id: str) -> Optional[str]:
        """Return the text stored for a split entry (e.g. session:msg:u or :a)."""
        eid = str(entry_id).strip()
        if not eid:
            return None
        with self._lock:
            results = self._collection.get(ids=[eid], include=["metadatas"])
        if results and results["metadatas"]:
            return (results["metadatas"][0].get("text") or "").strip() or None
        return None

    def get_raw_turn_text_by_id(self, summary_id: str) -> Optional[str]:
        """Return the raw (pre-compression) turn text for a summary ID, or None if not stored."""
        sid = str(summary_id).strip()
        if not sid:
            return None
        with self._lock:
            results = self._collection.get(ids=[sid], include=["metadatas"])
        if results and results["metadatas"]:
            return (results["metadatas"][0].get("raw_text") or "").strip() or None
        return None

    def get_summary_text_by_id(self, summary_id: str) -> Optional[str]:
        """Return the summary_text metadata for an entry (legacy artifacts that
        predate the text/raw_text keys store only summary_text)."""
        sid = str(summary_id).strip()
        if not sid:
            return None
        with self._lock:
            results = self._collection.get(ids=[sid], include=["metadatas"])
        if results and results["metadatas"]:
            return (results["metadatas"][0].get("summary_text") or "").strip() or None
        return None

    def get_recent_summaries(
        self, 
        session_id: int, 
        k: int = 2, 
        text_only: bool = True
    ) -> List[str] | List[Dict[str, Any]]:
        """Return the most recent summaries for a session as text or metadata rows."""
        
        with self._lock:
            results = self._collection.get(
                where={"session_id": session_id},
                include=["metadatas"]
            )
        
        if not results or not results['ids']:
            return []

        # Sort by timestamp (descending) to find the most recent
        sorted_metas = sorted(results['metadatas'], key=lambda m: m.get('ts', ''), reverse=True)
        
        metas = sorted_metas[:k]

        if not metas:
            return []

        if text_only:
            return [m.get("summary_text", "").strip() for m in metas if m.get("summary_text")]
        
        return metas

    def get_summaries_by_ids(self, summary_ids: list[str], max_len: int = 3000, top_n: int = 10) -> list[str]:
        """Fetch summary texts by ID, truncating each result and capping the count."""
        ids = [str(sid).strip() for sid in summary_ids if sid is not None]
        if not ids:
            return []

        with self._lock:
            results = self._collection.get(ids=ids, include=['metadatas'])

        out = []
        if results and results['metadatas']:
            for m in results['metadatas']:
                t = (m.get("summary_text") or "").strip()
                if not t:
                    continue
                out.append(t if len(t) <= max_len else t[:max_len] + "…")
                if len(out) >= top_n:
                    break
        return out
