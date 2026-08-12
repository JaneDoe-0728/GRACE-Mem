# graph/neo4j.py
# Drop-in replacement for your Neo4j wrapper, backed by FalkorDB (OpenCypher over RESP/Redis protocol).
#
# Requires:
#   pip install FalkorDB
#
# Env (kept for backward compatibility with your existing project):
#   NEO4J_URI        -> treated as Redis URI for FalkorDB, e.g. redis://:pass@localhost:6379/0
#   NEO4J_USERNAME   -> Redis ACL username (optional; can be omitted if your Redis uses default user)
#   NEO4J_PASSWORD   -> Redis password (optional if no auth)
# Optional:
#   GRAPH_NAME       -> graph key/name in FalkorDB (default: "memory")
#
# Notes:
# - FalkorDB supports parameterized queries via graph.query(cypher, params_dict)
# - Unique constraints in FalkorDB are created via GRAPH.CONSTRAINT CREATE ... (requires index first)

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import os
from urllib.parse import urlparse
import json
import time

from grace_mem.utils.logger_config import make_module_jlog

_jlog = make_module_jlog(name="grace_mem.Graph", filename="kg_ingestor.jsonl")
logger = logging.getLogger(__name__)

# Max rows per UNWIND batch query.  Keeps individual query strings manageable.
_BATCH_SIZE = 200

try:
    from falkordb import FalkorDB
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: FalkorDB python client.\n"
        "Install with: pip install FalkorDB"
    ) from e

try:
    import redis
    from redis.exceptions import ConnectionError as RedisConnectionError
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: redis-py.\n"
        "It should be installed transitively, but if not: pip install redis"
    ) from e


@dataclass
class GraphConfig:
    # For compatibility with your original code, we keep these names.
    # uri can be a redis/rediss URI: redis://[:password]@host:port/db
    uri: str
    user: str
    password: str

    # FalkorDB graph key/name
    graph_name: str = "memory"

    # Your KG schema knobs (same defaults as before)
    entity_label: str = "Entity"
    rel_type: str = "KG_REL"

    # Redis client settings
    decode_responses: bool = True
    socket_connect_timeout: float = 5.0
    socket_timeout: float = 30.0


class Graph:
    """
    Encapsulates FalkorDB operations using OpenCypher.
    Designed to be a drop-in replacement for the original Neo4j wrapper.
    """

    def __init__(self, cfg: GraphConfig) -> None:
        """Store FalkorDB settings and lazy connection handles."""
        self.cfg = cfg
        self._db: Optional[FalkorDB] = None
        self._graph = None  # FalkorDB Graph object

    # ---------- lifecycle ----------
    def open(self) -> "Graph":
        """Connect to FalkorDB and select the configured graph if needed."""
        if self._db is None or self._graph is None:
            self._db = self._connect()
            self._graph = self._db.select_graph(self.cfg.graph_name)
        return self

    def close(self) -> None:
        """Close the FalkorDB client and drop cached connection objects."""
        # FalkorDB client is redis-py based; close connection pool if present
        if self._db is not None:
            try:
                # redis.Redis has .close() in newer versions; connection_pool.disconnect() is a fallback
                if hasattr(self._db, "close"):
                    self._db.close()
                elif hasattr(self._db, "connection_pool") and self._db.connection_pool is not None:
                    self._db.connection_pool.disconnect()
            except Exception:
                pass
        self._db = None
        self._graph = None

    def reconnect(self) -> "Graph":
        """
        Force drop existing client and reconnect.
        """
        try:
            self.close()
        except Exception:
            pass
        return self.open()

    def __enter__(self) -> "Graph":
        """Enter a context-managed FalkorDB wrapper."""
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close the FalkorDB connection when leaving a context manager block."""
        self.close()

    # ---------- schema ----------
    def init_schema(self) -> None:
        """
        Best-effort schema init:
        - Create exact-match indices on (Entity.id) and (REL.id)
        - Create UNIQUE constraints for those properties (requires indices)
        If constraint creation fails (e.g. already exists / permissions), we continue.
        """
        self._ensure_open()

        # Create indices via Cypher (supported)
        index_stmts = [
            f"CREATE INDEX FOR (e:{self.cfg.entity_label}) ON (e.id)",
            f"CREATE INDEX FOR ()-[r:{self.cfg.rel_type}]-() ON (r.id)",
        ]
        for s in index_stmts:
            try:
                self._run_write(s, {})
                logger.debug("[Schema] Index created: %s", s)
            except Exception as e:
                # Index might already exist or server might reject; continue
                logger.debug("[Schema] Index skipped: %s | %r", s, e)

        # Create UNIQUE constraints via Redis command GRAPH.CONSTRAINT CREATE
        # NOTE: FalkorDB requires an exact-match index before creating a unique constraint.
        try:
            self._create_unique_constraint_node(self.cfg.entity_label, ["id"])
            logger.debug("[Schema] UNIQUE node constraint ok: %s(id)", self.cfg.entity_label)
        except Exception as e:
            logger.debug("[Schema] UNIQUE node constraint skipped: %r", e)

        try:
            self._create_unique_constraint_rel(self.cfg.rel_type, ["id"])
            logger.debug("[Schema] UNIQUE rel constraint ok: %s(id)", self.cfg.rel_type)
        except Exception as e:
            logger.debug("[Schema] UNIQUE rel constraint skipped: %r", e)

    # ---------- upsert helpers ----------
    @staticmethod
    def _cypher_map_literal(d: Dict[str, Any]) -> str:
        """Render a flat dict of scalars as a Cypher map literal: {k: val, ...}"""
        parts = [f"{k}: {Graph._cypher_literal(v)}" for k, v in d.items()]
        return "{" + ", ".join(parts) + "}"

    def _build_unwind_query(self, rows: List[Dict[str, Any]], body: str) -> str:
        """Build: UNWIND [{...}, ...] AS row <body>  with all values embedded as literals."""
        list_lit = "[" + ", ".join(self._cypher_map_literal(r) for r in rows) + "]"
        return f"UNWIND {list_lit} AS row\n{body}"

    # ---------- upsert ----------
    def sync_entities(self, entity_idx: Dict) -> int:
        """Batch-upsert entities via a single UNWIND query per chunk (N+1 → O(N/batch) round-trips)."""
        if not entity_idx:
            logger.debug("[sync_entities] empty input")
            return 0

        t0 = time.monotonic()
        now_iso = datetime.utcnow().isoformat()

        # Deduplicate: keep last entry per id
        by_id: Dict[str, Dict[str, Any]] = {}
        dropped = 0
        for meta in entity_idx.values():
            if not meta or "id" not in meta:
                dropped += 1
                continue
            eid = str(meta["id"])
            by_id[eid] = {
                "id": eid,
                "name": meta.get("name"),
                "type": meta.get("type"),
                "description": meta.get("description"),
                "now": now_iso,
            }

        logger.debug("[sync_entities] total_in=%d unique=%d dropped=%d", len(entity_idx), len(by_id), dropped)

        if not by_id:
            return 0

        # Batch UNWIND body — row fields accessed as row.id, row.name, etc.
        batch_body = f"""MERGE (e:{self.cfg.entity_label} {{id: row.id}})
ON CREATE SET
    e.created_at  = row.now,
    e.updated_at  = row.now,
    e.name        = row.name,
    e.type        = row.type,
    e.description = row.description
ON MATCH SET
    e.name        = coalesce(row.name, e.name),
    e.type        = coalesce(row.type, e.type),
    e.description = CASE
        WHEN row.description IS NOT NULL AND row.description <> e.description
        THEN row.description ELSE e.description END,
    e.updated_at  = row.now
RETURN count(*) AS ct"""

        # Single-row fallback cypher (used when a batch chunk fails)
        single_body = f"""MERGE (e:{self.cfg.entity_label} {{id: $id}})
ON CREATE SET
    e.created_at  = $now,
    e.updated_at  = $now,
    e.name        = $name,
    e.type        = $type,
    e.description = $description
ON MATCH SET
    e.name        = coalesce($name, e.name),
    e.type        = coalesce($type, e.type),
    e.description = CASE
        WHEN $description IS NOT NULL AND $description <> e.description
        THEN $description ELSE e.description END,
    e.updated_at  = $now
RETURN 1 AS ct"""

        rows = list(by_id.values())
        ok = 0

        for chunk_start in range(0, len(rows), _BATCH_SIZE):
            chunk = rows[chunk_start : chunk_start + _BATCH_SIZE]
            try:
                cypher = self._build_unwind_query(chunk, batch_body)
                result = self._run_write(cypher, {})
                ct = (result or {}).get("ct")
                ok += int(ct) if ct is not None else len(chunk)
            except Exception as batch_exc:
                _jlog("graph_entity_batch_failed", None,
                      chunk_offset=chunk_start,
                      chunk_size=len(chunk),
                      error=str(batch_exc),
                      error_type=type(batch_exc).__name__)
                for row in chunk:
                    try:
                        self._run_write(single_body, row)
                        ok += 1
                    except Exception as e:
                        _jlog("graph_entity_row_failed", None,
                              entity_id=row.get("id"),
                              error=str(e),
                              error_type=type(e).__name__)

        elapsed = time.monotonic() - t0
        logger.debug("[sync_entities] ok=%d/%d elapsed=%.3fs", ok, len(by_id), elapsed)
        return ok

    def sync_relationships(self, relationship_metas: List[Dict]) -> int:
        """Batch-upsert relationships via a single UNWIND query per chunk.

        The per-row endpoint existence check (which doubled query count) is removed:
        MATCH naturally skips rows where source or target nodes don't exist, and we
        log a count mismatch to catch missing-node situations at the batch level.
        """
        if not relationship_metas:
            logger.debug("[sync_relationships] empty input")
            return 0

        t0 = time.monotonic()
        now_iso = datetime.utcnow().isoformat()

        rows = []
        dropped = 0
        for m in relationship_metas:
            sid = m.get("source_id")
            tid = m.get("target_id")
            rid = m.get("id")
            if not sid or not tid or not rid:
                dropped += 1
                continue

            sid, tid, rid = str(sid), str(tid), str(rid)

            kw = m.get("keywords")
            if isinstance(kw, (list, tuple, set)):
                kw = json.dumps(list(kw), ensure_ascii=False)

            strength = m.get("strength")
            if strength is not None:
                try:
                    strength = int(strength)
                except Exception:
                    strength = None

            rows.append({
                "rid": rid,
                "sid": sid,
                "tid": tid,
                "description": m.get("description"),
                "keywords": kw,
                "strength": strength,
                "source_type": m.get("source_type"),
                "target_type": m.get("target_type"),
                "now": now_iso,
            })

        logger.debug("[sync_relationships] total_in=%d kept=%d dropped=%d", len(relationship_metas), len(rows), dropped)

        if not rows:
            return 0

        # Batch UNWIND body — MATCH skips rows where endpoints are absent
        batch_body = f"""MATCH (s:{self.cfg.entity_label} {{id: row.sid}})
MATCH (t:{self.cfg.entity_label} {{id: row.tid}})
MERGE (s)-[r:{self.cfg.rel_type} {{id: row.rid}}]->(t)
ON CREATE SET
    r.created_at  = row.now,
    r.updated_at  = row.now
SET
    r.description = row.description,
    r.keywords    = row.keywords,
    r.strength    = row.strength,
    r.source_type = row.source_type,
    r.target_type = row.target_type,
    r.updated_at  = row.now
RETURN count(*) AS ct"""

        # Single-row fallback (no pre-check; MATCH handles missing endpoints)
        single_body = f"""MATCH (s:{self.cfg.entity_label} {{id: $sid}})
MATCH (t:{self.cfg.entity_label} {{id: $tid}})
MERGE (s)-[r:{self.cfg.rel_type} {{id: $rid}}]->(t)
ON CREATE SET
    r.created_at  = $now,
    r.updated_at  = $now
SET
    r.description = $description,
    r.keywords    = $keywords,
    r.strength    = $strength,
    r.source_type = $source_type,
    r.target_type = $target_type,
    r.updated_at  = $now
RETURN 1 AS ct"""

        ok = 0
        for chunk_start in range(0, len(rows), _BATCH_SIZE):
            chunk = rows[chunk_start : chunk_start + _BATCH_SIZE]
            try:
                cypher = self._build_unwind_query(chunk, batch_body)
                result = self._run_write(cypher, {})
                ct = (result or {}).get("ct")
                written = int(ct) if ct is not None else len(chunk)
                if written < len(chunk):
                    _jlog("graph_rel_chunk_partial", None,
                          chunk_offset=chunk_start,
                          written=written,
                          chunk_size=len(chunk),
                          note="some source/target nodes likely absent")
                ok += written
            except Exception as batch_exc:
                _jlog("graph_rel_batch_failed", None,
                      chunk_offset=chunk_start,
                      chunk_size=len(chunk),
                      error=str(batch_exc),
                      error_type=type(batch_exc).__name__)
                for row in chunk:
                    try:
                        self._run_write(single_body, row)
                        ok += 1
                    except Exception as e:
                        _jlog("graph_rel_row_failed", None,
                              relationship_id=row.get("rid"),
                              source_id=row.get("sid"),
                              target_id=row.get("tid"),
                              error=str(e),
                              error_type=type(e).__name__)

        elapsed = time.monotonic() - t0
        logger.debug("[sync_relationships] ok=%d/%d elapsed=%.3fs", ok, len(rows), elapsed)
        return ok

    # ---------- queries ----------
    def get_node_subgraph(self, entity_ids: List[str]) -> Dict[str, Dict]:
        """Fetch matching nodes and their adjacent edges and neighbors."""
        if not entity_ids:
            return {}

        # IMPORTANT: build ids_lit with proper escaping
        ids = [str(x) for x in entity_ids if x is not None and str(x).strip()]
        if not ids:
            return {}
        ids_lit = "[" + ",".join(self._cypher_literal(i) for i in ids) + "]"

        query = f"""
        MATCH (e:{self.cfg.entity_label})
        WHERE e.id IN {ids_lit}
        OPTIONAL MATCH (e)-[r:{self.cfg.rel_type}]-(n:{self.cfg.entity_label})
        RETURN
        e.id AS source_id,
        e.name AS source_name,
        e.type AS source_type,
        e.description AS source_desc,

        r.id AS rel_id,
        r.description AS rel_desc,
        r.keywords AS rel_keywords,
        r.strength AS rel_strength,

        n.id AS neighbor_id,
        n.name AS neighbor_name,
        n.type AS neighbor_type,
        n.description AS neighbor_desc
        """

        rows = self._run_read(query, {})
        out: Dict[str, Dict] = {}

        for rec in rows:
            sid = rec.get("source_id")
            if sid is None:
                continue

            item = out.setdefault(str(sid), {
                "self": {
                    "id": sid,
                    "name": rec.get("source_name"),
                    "type": rec.get("source_type"),
                    "desc": rec.get("source_desc"),
                },
                "neighbors": [],
            })

            # OPTIONAL MATCH miss: neighbor_id will be None
            if rec.get("neighbor_id") is None:
                continue

            item["neighbors"].append({
                "rel_id": rec.get("rel_id"),
                "rel_desc": rec.get("rel_desc"),
                "rel_keywords": rec.get("rel_keywords"),
                "rel_strength": rec.get("rel_strength"),
                "neighbor_id": rec.get("neighbor_id"),
                "neighbor_name": rec.get("neighbor_name"),
                "neighbor_type": rec.get("neighbor_type"),
                "neighbor_desc": rec.get("neighbor_desc"),
            })

        return out


    def check_entity_ids(self, ids: List[str]) -> List[str]:
        """Return the subset of entity IDs that actually exist in FalkorDB."""
        if not ids:
            return []
        safe_ids = [str(x) for x in ids if x]
        if not safe_ids:
            return []
        ids_lit = "[" + ",".join(self._cypher_literal(i) for i in safe_ids) + "]"
        query = f"MATCH (e:{self.cfg.entity_label}) WHERE e.id IN {ids_lit} RETURN e.id AS id"
        rows = self._run_read(query, {})
        return [r["id"] for r in rows if r.get("id") is not None]

    def check_relationship_ids(self, ids: List[str]) -> List[str]:
        """Return the subset of relationship IDs that actually exist in FalkorDB."""
        if not ids:
            return []
        safe_ids = [str(x) for x in ids if x]
        if not safe_ids:
            return []
        ids_lit = "[" + ",".join(self._cypher_literal(i) for i in safe_ids) + "]"
        query = f"MATCH ()-[r:{self.cfg.rel_type}]->() WHERE r.id IN {ids_lit} RETURN r.id AS id"
        rows = self._run_read(query, {})
        return [r["id"] for r in rows if r.get("id") is not None]

    def get_edge_subgraph(self, rel_ids: List[str]) -> List[Dict]:
        """Fetch matching relationships together with source and target nodes."""
        if not rel_ids:
            return []

        rids = [str(x).strip() for x in rel_ids if x is not None and str(x).strip()]
        if not rids:
            return []

        rids_lit = "[" + ",".join(self._cypher_literal(x) for x in rids) + "]"

        query = f"""
        MATCH (a:{self.cfg.entity_label})-[r:{self.cfg.rel_type}]->(b:{self.cfg.entity_label})
        WHERE r.id IN {rids_lit}
        RETURN
        r.id AS rel_id,
        r.description AS rel_desc,
        r.keywords AS rel_keywords,
        r.strength AS rel_strength,

        a.id AS source_id,
        a.name AS source_name,
        a.type AS source_type,
        a.description AS source_desc,

        b.id AS target_id,
        b.name AS target_name,
        b.type AS target_type,
        b.description AS target_desc
        """

        rows = self._run_read(query, {})
        out: List[Dict] = []

        for rec in rows:
            # rec is already a dict of primitives (thanks to _rows_as_dicts)
            if rec.get("rel_id") is None:
                continue
            out.append({
                "rel_id": rec.get("rel_id"),
                "rel_desc": rec.get("rel_desc"),
                "rel_keywords": rec.get("rel_keywords"),
                "rel_strength": rec.get("rel_strength"),
                "source_id": rec.get("source_id"),
                "source_name": rec.get("source_name"),
                "source_type": rec.get("source_type"),
                "source_desc": rec.get("source_desc"),
                "target_id": rec.get("target_id"),
                "target_name": rec.get("target_name"),
                "target_type": rec.get("target_type"),
                "target_desc": rec.get("target_desc"),
            })

        return out

    # ---------- admin ----------
    def clear_all(self) -> None:
        """
        Clear the entire graph.
        If we hit a connection error, reconnect once and retry.
        """
        last_exc = None
        for _ in range(2):
            try:
                self._run_write("MATCH (n) DETACH DELETE n", {})
                return
            except RedisConnectionError as e:
                last_exc = e
                self.reconnect()
            except Exception as e:
                last_exc = e
                # some servers might not like DETACH DELETE; fallback
                try:
                    self._run_write("MATCH (n) DELETE n", {})
                    return
                except Exception as e2:
                    last_exc = e2
                    self.reconnect()
        raise RuntimeError("Failed to clear_all after reconnect") from last_exc

    # ---------- low-level helpers ----------
    # def _run_read(self, cypher: str, params: dict) -> List[Dict[str, Any]]:
    #     raw = self._exec_graph_query(cypher, params or {}, readonly=True)
    #     return self._rows_as_dicts(raw)
    def _run_read(self, cypher: str, params: dict) -> List[Dict[str, Any]]:
        """Execute a read query and normalize FalkorDB's raw response rows."""
        self._ensure_open()
        raw = self._exec_graph_query(cypher, params or {}, readonly=True)

        # 🔥 這三行最重要：直接看 raw 回傳
        recs = self._rows_as_dicts(raw)
        logger.debug("[_run_read] raw_type=%s len=%s parsed=%d", type(raw).__name__,
                     len(raw) if isinstance(raw, (list, tuple)) else None, len(recs))
        return recs

    def _run_write(self, cypher: str, params: dict) -> Optional[Dict[str, Any]]:
        """Execute a write query and return the first normalized response row."""
        raw = self._exec_graph_query(cypher, params or {}, readonly=False)
        rows = self._rows_as_dicts(raw)
        return rows[0] if rows else None

    def _run_write_batch(self, statements: List[str]) -> None:
        """Execute a batch of write statements against the selected graph."""
        self._ensure_open()
        for s in statements:
            self._graph.query(s, {})

    def _ensure_open(self) -> None:
        """Open the database connection lazily before a graph operation."""
        if self._db is None or self._graph is None:
            self.open()

    @staticmethod
    def _rows_as_dicts(query_result: Any) -> List[Dict[str, Any]]:
        """
        Supports:
        A) Raw RedisGraph/FalkorDB reply: [header, rows, stats]
        - header can be:
            A1) [[col, type], ...]
            A2) [col, col, ...]   <-- your current case
        B) FalkorDB-py QueryResult: .header (list[str]), .result_set (list[list])
        """
        if query_result is None:
            return []

        def _to_str(x: Any) -> str:
            """Convert raw header tokens into regular Python strings."""
            if isinstance(x, (bytes, bytearray)):
                return x.decode("utf-8", errors="ignore")
            return str(x)

        # ----- Raw reply path -----
        if isinstance(query_result, (list, tuple)) and len(query_result) >= 2:
            header_raw = query_result[0]
            rows_raw = query_result[1]

            # header_raw must be list/tuple to proceed
            if not isinstance(header_raw, (list, tuple)) or len(header_raw) == 0:
                return []

            header: List[str] = []

            # Case A1: header = [[col,type], ...]
            if isinstance(header_raw[0], (list, tuple)):
                for h in header_raw:
                    if not h:
                        continue
                    header.append(_to_str(h[0]))
            else:
                # Case A2: header = [col, col, ...]
                header = [_to_str(h) for h in header_raw]

            if not header:
                return []

            if not isinstance(rows_raw, (list, tuple)):
                return []

            out: List[Dict[str, Any]] = []
            for row in rows_raw:
                d: Dict[str, Any] = {}
                if not isinstance(row, (list, tuple)):
                    # single scalar row
                    d[header[0]] = row
                else:
                    for i, col in enumerate(header):
                        d[col] = row[i] if i < len(row) else None
                out.append(d)
            return out

        # ----- QueryResult object path -----
        header = getattr(query_result, "header", None) or []
        result_set = getattr(query_result, "result_set", None) or []
        if not header:
            return []
        out: List[Dict[str, Any]] = []
        for row in result_set:
            d = {}
            for i, col in enumerate(header):
                d[col] = row[i] if i < len(row) else None
            out.append(d)
        return out

    # ---------- raw GRAPH.QUERY helpers (bypass falkordb-py query wrapper bug) ----------
    @staticmethod
    def _cypher_literal(v: Any) -> str:
        """Convert Python value to a Cypher literal for the CYPHER param=val prefix."""
        if v is None:
            return "NULL"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        # everything else -> string
        s = str(v)
        s = s.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{s}'"

    def _build_cypher_prefix(self, params: Optional[Dict[str, Any]]) -> str:
        """
        Build: CYPHER k=val k2=val2 ...
        This is the server-side parameterization format:
        GRAPH.QUERY graph "CYPHER a='x' MATCH ... {name:$a} ..."
        """
        if not params:
            return ""
        parts = [f"{k}={self._cypher_literal(v)}" for k, v in params.items()]
        return "CYPHER " + " ".join(parts) + " "

    def _exec_graph_query(self, query: str, params: Optional[Dict[str, Any]] = None, readonly: bool = False) -> Any:
        """
        Execute query via Redis command directly, bypassing falkordb-py Graph.query().
        Returns raw Redis reply: [header, rows, stats]
        """
        self._ensure_open()
        if not hasattr(self._db, "execute_command"):
            raise RuntimeError("Underlying redis client does not support execute_command")

        full_query = self._build_cypher_prefix(params) + query
        cmd = "GRAPH.RO_QUERY" if readonly else "GRAPH.QUERY"
        return self._db.execute_command(cmd, self.cfg.graph_name, full_query)

    # ---------- FalkorDB-specific helpers ----------
    def _connect(self) -> FalkorDB:
        """Create a FalkorDB client from the configured Redis-style URI."""
        uri = (self.cfg.uri or "").strip()
        if not uri:
            raise RuntimeError("Missing cfg.uri (expected redis://... for FalkorDB).")

        # 直接用 URL 連，避免 __init__ kwargs 不相容
        return FalkorDB.from_url(uri)

    def _create_unique_constraint_node(self, label: str, props: List[str]) -> Any:
        """
        GRAPH.CONSTRAINT CREATE <graph> UNIQUE NODE <label> PROPERTIES <n> <prop...>
        """
        self._ensure_open()
        if not hasattr(self._db, "execute_command"):
            raise RuntimeError("Underlying client does not support execute_command")

        args: List[Any] = [
            "GRAPH.CONSTRAINT",
            "CREATE",
            self.cfg.graph_name,
            "UNIQUE",
            "NODE",
            label,
            "PROPERTIES",
            str(len(props)),
            *props,
        ]
        return self._db.execute_command(*args)

    def _create_unique_constraint_rel(self, rel_type: str, props: List[str]) -> Any:
        """
        GRAPH.CONSTRAINT CREATE <graph> UNIQUE RELATIONSHIP <type> PROPERTIES <n> <prop...>
        """
        self._ensure_open()
        if not hasattr(self._db, "execute_command"):
            raise RuntimeError("Underlying client does not support execute_command")

        args: List[Any] = [
            "GRAPH.CONSTRAINT",
            "CREATE",
            self.cfg.graph_name,
            "UNIQUE",
            "RELATIONSHIP",
            rel_type,
            "PROPERTIES",
            str(len(props)),
            *props,
        ]
        return self._db.execute_command(*args)


# --- factory by env (方便 server/main 使用) ---
def graph_from_env(entity_label: str = "Entity", rel_type: str = "KG_REL") -> Graph:
    """
    Read env vars and build Graph object.

    Kept for compatibility:
      - NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD

    Additional optional:
      - GRAPH_NAME (default "memory")
    """
    uri = os.environ.get("NEO4J_URI")  # compatibility name
    user = os.environ.get("NEO4J_USERNAME", "")
    pwd = os.environ.get("NEO4J_PASSWORD", "")
    graph_name = os.environ.get("GRAPH_NAME", "memory")

    if not uri:
        raise RuntimeError("Missing NEO4J_URI in environment. (Expect redis://... for FalkorDB.)")

    return Graph(GraphConfig(uri=uri, user=user, password=pwd, graph_name=graph_name,
                             entity_label=entity_label, rel_type=rel_type))