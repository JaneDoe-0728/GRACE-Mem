"""ChromaDB vector stores for entities, relationships, and summaries.

`SimpleChromaVDB` wraps one Chroma collection; the three subclasses at the
bottom specialise it per content type. They are separate collections rather
than one with a type field so a search can be scoped without a metadata filter
-- entity search must never surface a summary, and enforcing that structurally
beats enforcing it in every query.

Two Chroma constraints shape everything here.

Metadata must be scalar. Chroma stores only str/int/float/bool/None, but the
records carry nested provenance and temporal blobs, so `_serialize_metadata`
JSON-encodes them on write and `_deserialize_metadata` restores them on read.
Callers see nested dicts and never handle the encoding.

Similarity is cosine over unit vectors. Vectors are L2-normalized before every
write and query, which lets Chroma's distance be read directly as cosine
similarity. Skipping normalization on either side does not error -- it just
returns quietly wrong rankings.
"""

import datetime
import json
import threading
import uuid
from collections.abc import Mapping
from typing import Any, cast

import chromadb
import numpy as np

from grace_mem.services.embedding.embeddings import embedder
from grace_mem.utils.atomic_write import atomic_write
from grace_mem.utils.logger_config import make_module_jlog

_debug_jlog = make_module_jlog(
    name="grace_mem.Storage.ChromaVDB",
    filename="kg_chroma_debug.jsonl",
)

ChromaScalar = str | int | float | bool | None
ChromaMetadata = dict[str, ChromaScalar]

def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit L2 norm.

    Unit vectors are what make Chroma's inner product equal cosine similarity,
    so this is applied on both the write and query paths.

    A 1-D input is promoted to a single row, letting callers pass one vector
    without reshaping. Zero-norm rows -- an empty string embeds to one -- would
    divide by zero, so their norm is floored at 1e-12; the row stays zero and
    scores 0 against everything, which is the right answer for empty text.
    """
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    return vectors / norm


def _serialize_metadata(meta: dict[str, Any]) -> ChromaMetadata:
    """
    Serialize nested metadata values to JSON strings for ChromaDB compatibility.
    ChromaDB only accepts str, int, float, bool, or None as metadata values.
    """
    result: ChromaMetadata = {}
    for k, v in meta.items():
        if isinstance(v, (dict, list)):
            result[k] = json.dumps(v)
        else:
            result[k] = v
    return result


def _deserialize_metadata(meta: Mapping[str, ChromaScalar] | None) -> dict[str, Any]:
    """Restore JSON-encoded metadata values written by `_serialize_metadata`.

    Which values were encoded is not recorded, so this infers it: a string
    opening with "{" or "[" is attempted as JSON and left alone if it does not
    parse. The inference is imperfect by construction -- a genuine description
    that happens to begin with a brace and parse as JSON would come back as a
    dict. That has not occurred in practice on this corpus, and the alternative
    (a parallel map of encoded keys) would have to stay in sync with the data
    forever.
    """
    if meta is None:
        return {}
    result: dict[str, Any] = {}
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
    """One persistent Chroma collection behind a small, FAISS-shaped API.

    The interface deliberately mirrors the FAISS store this replaced, so the
    two remain swappable and the retrieval code stays backend-agnostic.

    Deduplication on `id` runs on both write and read. It is needed on both
    because an id can enter the collection more than once through different
    paths, and when it does, the copy with the longer provenance wins -- that
    one has accumulated more source turns and is strictly more informative.

    Every mutation and query holds `_lock`: Chroma's client is not safe to
    share across the ingestion worker threads that reach it.
    """

    def __init__(self, dim: int, path: str, collection_name: str) -> None:
        """Open or create the backing collection, configured for determinism.

        The collection is created with `hnsw:space="ip"` -- inner product,
        which equals cosine on the unit vectors this class writes -- and
        `hnsw:num_threads=1`. Single-threaded search costs latency and buys
        reproducibility: multi-threaded HNSW traversal can return different
        neighbours for the same query between runs, which would make an
        experiment's results unrepeatable for reasons unrelated to the change
        being measured.
        """
        self.dim = dim
        self.path = path
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=self.path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "ip",
                "hnsw:num_threads": 1,
            },
        )
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        """Return the current number of stored vectors in the collection."""
        with self._lock:
            return self._collection.count()

    def add(self, vectors: np.ndarray, metadatas: list[dict[str, Any]]) -> None:
        """Write vectors and metadata, deduplicating by id within the batch.

        Implemented as a Chroma upsert, so this both inserts and overwrites --
        re-adding an existing id replaces it rather than raising.

        Args:
            vectors: Shape (n, dim). Normalized here, so callers need not.
            metadatas: One per row. An entry without an `id` gets a generated
                UUID, which makes it unaddressable afterwards -- always supply
                ids for anything that will be updated later.

        Raises:
            AssertionError: If the vector count or width does not match.
                Asserted rather than validated because a width mismatch means
                the wrong embedding model is configured, and the run is not
                salvageable.
        """
        if not metadatas:
            return

        assert vectors.shape[0] == len(metadatas)
        assert vectors.shape[1] == self.dim

        # Within-batch dedup on id, keeping the copy with the most provenance
        # events. The same rule as the read path in `search`, so a duplicate
        # resolves the same way whichever side it is caught on.
        dedup_map: dict[str, tuple[int, int]] = {}  # id -> (index, prov_len)
        for i, meta in enumerate(metadatas):
            mid = str(meta.get("id") or uuid.uuid4())
            prov_len = len((meta.get("prov") or {}).get("events", []))
            prev = dedup_map.get(mid)
            if prev is None or prov_len > prev[1]:
                dedup_map[mid] = (i, prov_len)

        dedup_indices = [idx for idx, _ in dedup_map.values()]
        dedup_vectors = vectors[dedup_indices]
        dedup_metadatas = [metadatas[i] for i in dedup_indices]
        dedup_ids = [str(m.get("id") or uuid.uuid4()) for m in dedup_metadatas]

        serialized_metadatas: list[Mapping[str, ChromaScalar]] = [
            _serialize_metadata(m) for m in dedup_metadatas
        ]

        normalized_vectors = normalize_l2(dedup_vectors.astype(np.float32))

        with self._lock:
            self._collection.upsert(
                ids=dedup_ids,
                embeddings=normalized_vectors.tolist(),
                metadatas=serialized_metadatas
            )

    def upsert(self, vectors: np.ndarray, metadatas: list[dict[str, Any]]) -> None:
        """Alias add() so callers can use an upsert-style API."""
        self.add(vectors, metadatas)

    def search(self, query_vec: np.ndarray, top_k: int = 5, threshold: float | None = None
               ) -> list[tuple[dict[str, Any], float]]:
        """Return the top_k nearest entries by cosine similarity, best first.

        Args:
            top_k: Neighbours requested from Chroma. Note this is applied
                *before* deduplication, so a result set containing duplicate
                ids returns fewer than top_k rows.
            threshold: Minimum similarity to keep. None keeps everything.

        Returns:
            (metadata, score) pairs, descending by score. Metadata is
            deserialized, so nested provenance and temporal blobs arrive as
            dicts.
        """
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        
        normalized_query = normalize_l2(query_vec.astype(np.float32))

        with self._lock:
            results = self._collection.query(
                query_embeddings=normalized_query.tolist(),
                n_results=top_k,
            )

        hits: list[tuple[dict[str, Any], float]] = []
        ids_batches = results.get("ids") or []
        distance_batches = results.get("distances") or []
        metadata_batches = results.get("metadatas") or []
        if ids_batches and distance_batches and metadata_batches:
            for _, distance, raw_meta in zip(
                ids_batches[0], distance_batches[0], metadata_batches[0]
            ):
                # Chroma reports the 'ip' space as a distance, 1 - inner_product.
                # Inverting recovers the similarity, which on unit vectors is
                # cosine -- the scale every caller and threshold assumes.
                score = 1.0 - distance
                if threshold is None or score >= threshold:
                    meta = _deserialize_metadata(raw_meta)
                    hits.append((meta, float(score)))

        # Same id can appear more than once; keep the richest copy. Ranked on
        # (provenance length, score) so provenance dominates -- between two
        # copies of one entity, the one citing more source turns is preferred
        # even at a slightly lower score, since the score reflects the shared
        # embedding while provenance reflects real evidence.
        filtered: dict[str, tuple[dict[str, Any], float, int]] = {}
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
        self, query_vecs: np.ndarray, top_k: int = 5, threshold: float | None = None
    ) -> list[list[tuple[dict[str, Any], float]]]:
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

        ids_batches = results.get("ids") or []
        distance_batches = results.get("distances") or []
        metadata_batches = results.get("metadatas") or []
        out: list[list[tuple[dict[str, Any], float]]] = []
        for qi in range(n):
            hits: list[tuple[dict[str, Any], float]] = []
            if qi < min(len(ids_batches), len(distance_batches), len(metadata_batches)):
                for _, distance, raw_meta in zip(
                    ids_batches[qi], distance_batches[qi], metadata_batches[qi]
                ):
                    score = 1.0 - distance
                    if threshold is None or score >= threshold:
                        meta = _deserialize_metadata(raw_meta)
                        hits.append((meta, float(score)))

            filtered: dict[str, tuple[dict[str, Any], float, int]] = {}
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
    ) -> tuple[dict[str, Any], float] | None:
        """Fetch one stored vector by ID and compare it to the query vector."""
        
        if query_vec.ndim == 1:
            qv = query_vec[None, :]
        else:
            qv = query_vec
            
        normalized_qv = normalize_l2(qv.astype(np.float32))

        with self._lock:
            res = self._collection.get(ids=[mid], include=['embeddings', 'metadatas'])
        
        embeddings = cast(Any, res.get("embeddings"))
        metadatas = res.get("metadatas")
        if not res or not res['ids'] or embeddings is None or metadatas is None:
            return None

        stored_vec = np.array(embeddings[0])
        target_meta = _deserialize_metadata(metadatas[0])
        
        # Stored vectors are already normalized
        score = float(np.dot(stored_vec, normalized_qv[0]))

        return (target_meta, score) if score >= threshold else None

    def compare_by_id_raw(
        self,
        mid: str,
        query_vec: np.ndarray,
        request_id: str | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> float | None:
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

            retry_embeddings = cast(Any, retry_res.get("embeddings"))
            if retry_embeddings is None:
                return None
            stored_vec = np.array(retry_embeddings[0])
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
        embeddings = cast(Any, res.get("embeddings"))
        if embeddings is None:
            return None
        stored_vec = np.array(embeddings[0])
        return float(np.dot(stored_vec, normalized_qv[0]))

    def delete(self, ids: list[str]) -> None:
        """
        Deletes entries by their IDs.
        """
        if not ids:
            return
        with self._lock:
            self._collection.delete(ids=ids)

    def update(self, ids: list[str], vectors: np.ndarray | None = None, metadatas: list[dict[str, Any]] | None = None) -> None:
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
        serialized_metadatas: list[Mapping[str, ChromaScalar]] | None = None
        if metadatas is not None:
            serialized_metadatas = [_serialize_metadata(m) for m in metadatas]

        with self._lock:
            self._collection.update(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=serialized_metadatas
            )

    def rebuild(self, all_vectors: np.ndarray, all_metadatas: list[dict[str, Any]]) -> None:
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
            self._collection = cast(Any, None)
            self._client = cast(Any, None)

    def load(self) -> None:
        """No-op: PersistentClient loads on open. Kept for FAISS API parity."""

    def export_metadatas_jsonl(self, output_path: str) -> int:
        """Dump every record's metadata to JSONL, one object per line.

        The analysis tooling reads this rather than the Chroma collection --
        it is inspectable without a Chroma dependency, and cheap to load into a
        dataframe.

        Written to a temp file and moved into place, so a reader never observes
        a half-written export, and flushed to disk before the move so a crash
        cannot promote a file the kernel had not finished writing.

        Records whose metadata lacks an `id` get the collection's id, so the
        export is self-contained and joinable.

        Returns:
            Rows written.
        """
        with self._lock:
            results = self._collection.get(include=["metadatas"])

        ids = results.get("ids") or []
        metas = results.get("metadatas") or []

        with atomic_write(output_path, "w", encoding="utf-8") as f:
            for mid, meta in zip(ids, metas):
                meta = _deserialize_metadata(meta) if meta is not None else {}
                if "id" not in meta:
                    meta["id"] = mid
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        return len(ids)


class EntitiesVDB(SimpleChromaVDB):
    """Vector store for entity descriptions.

    Adds nothing to the base class -- it exists as a distinct type so entity
    search cannot be pointed at the wrong collection, and so a future
    entity-specific behaviour has somewhere to live.
    """


class RelationshipsVDB(SimpleChromaVDB):
    """Vector store for relationship descriptions. See `EntitiesVDB`."""


class SummariesVDB(SimpleChromaVDB):
    """Vector store for per-turn dialogue summaries.

    The only subclass that specialises behaviour, because summaries are keyed
    by conversation position rather than by content. Ids are composed as
    "session:message" (and "session:message:u"/":a" when a turn is split), so a
    retrieved summary locates itself in the conversation without a lookup --
    which is what lets the evidence stage fetch neighbouring turns.
    """

    def add_summary(self, session_id: int, message_id: int, summary_text: str, dialogue_datetime: str | None = None, raw_text: str | None = None) -> str:
        """Embed and store one turn's summary.

        Only `summary_text` is embedded. `raw_text` rides along as metadata so
        the original wording stays available for evidence rendering without
        competing with the summary in vector search.

        Args:
            dialogue_datetime: The turn's wall-clock time, kept for temporal
                filtering. `ts` is separately recorded as ingest time -- the
                two differ and are not interchangeable.

        Returns:
            The summary id, "session_id:message_id".
        """
        vec = embedder.embed([summary_text])
        summary_id = f"{session_id}:{message_id}"
        meta: dict[str, Any] = {
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
        return summary_id

    def add_split_turns(
        self,
        session_id: int | str,
        message_id: int,
        user_text: str,
        assistant_summary: str,
        dialogue_datetime: str | None = None,
    ) -> None:
        """Store a turn as two entries: ":u" user raw, ":a" assistant summary.

        Split because the halves suit different queries. The user turn is
        usually the question or the fact being stated and is kept verbatim,
        while the assistant turn is long and is stored compressed. Embedding
        them together produced a vector dominated by assistant phrasing, which
        pushed the user's own words out of reach of retrieval.

        The suffixed ids extend the "session:message" scheme, so both halves
        still resolve back to one turn.
        """
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

    def get_text_by_entry_id(self, entry_id: str) -> str | None:
        """Return the text stored for a split entry (e.g. session:msg:u or :a)."""
        eid = str(entry_id).strip()
        if not eid:
            return None
        with self._lock:
            results = self._collection.get(ids=[eid], include=["metadatas"])
        if results and results["metadatas"]:
            return str(results["metadatas"][0].get("text") or "").strip() or None
        return None

    def get_raw_turn_text_by_id(self, summary_id: str) -> str | None:
        """Return the raw (pre-compression) turn text for a summary ID, or None if not stored."""
        sid = str(summary_id).strip()
        if not sid:
            return None
        with self._lock:
            results = self._collection.get(ids=[sid], include=["metadatas"])
        if results and results["metadatas"]:
            return str(results["metadatas"][0].get("raw_text") or "").strip() or None
        return None

    def get_summary_text_by_id(self, summary_id: str) -> str | None:
        """Return the summary_text metadata for an entry (legacy artifacts that
        predate the text/raw_text keys store only summary_text)."""
        sid = str(summary_id).strip()
        if not sid:
            return None
        with self._lock:
            results = self._collection.get(ids=[sid], include=["metadatas"])
        if results and results["metadatas"]:
            return str(results["metadatas"][0].get("summary_text") or "").strip() or None
        return None

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
                t = str(m.get("summary_text") or "").strip()
                if not t:
                    continue
                out.append(t if len(t) <= max_len else t[:max_len] + "…")
                if len(out) >= top_n:
                    break
        return out
