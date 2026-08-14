# graph/neo4j.py
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from neo4j import GraphDatabase, Driver
import os
from neo4j.exceptions import ServiceUnavailable

@dataclass
class GraphConfig:
    uri: str
    user: str
    password: str
    entity_label: str = "Entity"
    rel_type: str = "KG_REL"

class Graph:
    """
    Wrapper around the Neo4j operations.
    """
    def __init__(self, cfg: GraphConfig) -> None:
        """Store the Neo4j connection settings and lazy driver handle."""
        self.cfg = cfg
        self._driver: Optional[Driver] = None

    # ---------- lifecycle ----------
    def open(self) -> "Graph":
        """Open the Neo4j driver if needed and return this wrapper."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.cfg.uri, auth=(self.cfg.user, self.cfg.password), max_connection_lifetime=30)
        return self

    def close(self) -> None:
        """Close the current Neo4j driver and clear the cached handle."""
        if self._driver:
            self._driver.close()
            self._driver = None
    
    def reconnect(self) -> "Graph":
        """
        Drop the old driver and establish a fresh connection.
        Call this after a snapshot, or when a connection has gone defunct.
        """
        try:
            if self._driver:
                self._driver.close()
        except Exception:
            pass
        self._driver = None
        return self.open()

    # Enables: with Graph(cfg) as g:
    def __enter__(self) -> "Graph":
        """Enter a context-managed graph session wrapper."""
        return self.open()
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close the graph driver when leaving a context manager block."""
        self.close()

    # ---------- schema ----------
    def init_schema(self) -> None:
        """Create the unique constraints required by the graph schema."""
        stmts = [
            f"CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:{self.cfg.entity_label}) REQUIRE e.id IS UNIQUE",
            f"CREATE CONSTRAINT rel_id IF NOT EXISTS FOR ()-[r:{self.cfg.rel_type}]-() REQUIRE r.id IS UNIQUE",
        ]
        self._run_write_batch(stmts)
        for s in stmts:
            print("[Schema] Created or exists:", s)

    # ---------- upsert ----------
    def sync_entities(self, entity_idx: Dict) -> int:
        """
        Sync a batch of entities into Neo4j.
        - new entity -> create
        - already present -> update properties (the old description is kept and
          overwritten only when new content is supplied)
        Returns the number of records processed successfully.
        """
        if not entity_idx: return 0
        now_iso = datetime.utcnow().isoformat()

        by_id = {}
        for meta in entity_idx.values():
            if not meta or "id" not in meta: 
                continue
            by_id[meta["id"]] = {
                "id": meta["id"],
                "name": meta.get("name"),
                "type": meta.get("type"),
                "description": meta.get("description"),
                "now": now_iso,
            }
        rows = list(by_id.values())
        if not rows: return 0

        cypher = f"""
        UNWIND $rows AS row
        MERGE (e:{self.cfg.entity_label} {{id: row.id}})
        ON CREATE SET
            e.name        = row.name,
            e.type        = row.type,
            e.description = row.description,
            e.created_at  = row.now,
            e.updated_at  = row.now
        ON MATCH SET
            e.name        = coalesce(row.name, e.name),
            e.type        = coalesce(row.type, e.type),
            e.description = CASE
                WHEN row.description IS NOT NULL AND row.description <> e.description
                THEN row.description ELSE e.description END,
            e.updated_at  = row.now
        RETURN count(*) AS ct
        """
        rec = self._run_write(cypher, {"rows": rows})
        return rec["ct"] if rec else 0

    def sync_relationships(self, relationship_metas: List[Dict]) -> int:
        """
        Sync a batch of relationships into Neo4j.
        - new relationship -> create
        - already present -> update properties
        Returns the number of records processed successfully.
        """
        if not relationship_metas: return 0
        now_iso = datetime.utcnow().isoformat()
        rows = []
        for m in relationship_metas:
            sid, tid, rid = m.get("source_id"), m.get("target_id"), m.get("id")
            if not sid or not tid or not rid: 
                continue
            rows.append({
                "id": rid,
                "sid": sid,
                "tid": tid,
                "description": m.get("description"),
                "keywords": m.get("keywords"),
                "strength": int(m.get("strength", 0)) if m.get("strength") is not None else None,
                "source_type": m.get("source_type"),
                "target_type": m.get("target_type"),
                "now": now_iso,
            })
        if not rows: return 0

        cypher = f"""
        UNWIND $rows AS row
        MATCH (s:{self.cfg.entity_label} {{id: row.sid}})
        MATCH (t:{self.cfg.entity_label} {{id: row.tid}})
        MERGE (s)-[r:{self.cfg.rel_type} {{id: row.id}}]->(t)
        ON CREATE SET
            r.description  = row.description,
            r.keywords     = row.keywords,
            r.strength     = row.strength,
            r.source_type  = row.source_type,
            r.target_type  = row.target_type,
            r.created_at   = row.now,
            r.updated_at   = row.now
        ON MATCH SET
            r.description  = row.description,
            r.keywords     = row.keywords,
            r.strength     = row.strength,
            r.source_type  = row.source_type,
            r.target_type  = row.target_type,
            r.updated_at   = row.now
        RETURN count(*) AS ct
        """
        rec = self._run_write(cypher, {"rows": rows})
        return rec["ct"] if rec else 0

    # ---------- queries ----------
    def get_node_subgraph(self, entity_ids: List[str]) -> Dict[str, Dict]:
        """
        Query the subgraph around a batch of entities.
        - locate the nodes with the given ids
        - return them together with their neighbours (the linked nodes plus the
          relationship details)
        """
        if not entity_ids: return {}
        query = f"""
        UNWIND $ids AS id
        MATCH (e:{self.cfg.entity_label} {{id:id}})
        OPTIONAL MATCH (e)-[r]-(n:{self.cfg.entity_label})
        WITH id AS source_id, e, r, n
        WITH
          source_id,
          e.name        AS source_name,
          e.type        AS source_type,
          e.description AS source_desc,
          collect(DISTINCT CASE WHEN n IS NULL THEN NULL ELSE {{
            rel_id:       r.id,
            rel_desc:     r.description,
            rel_keywords: r.keywords,
            rel_strength: r.strength,
            neighbor_id:   n.id,
            neighbor_name: n.name,
            neighbor_type: n.type,
            neighbor_desc: n.description
          }} END) AS raw_neighbors
        RETURN
          source_id,
          source_name,
          source_type,
          source_desc,
          [x IN raw_neighbors WHERE x IS NOT NULL] AS neighbors
        """
        recs = self._run_read(query, {"ids": entity_ids})
        out: Dict[str, Dict] = {}
        for rec in recs:
            out[rec["source_id"]] = {
                "self": {
                    "id":   rec["source_id"],
                    "name": rec["source_name"],
                    "type": rec["source_type"],
                    "desc": rec["source_desc"],
                },
                "neighbors": rec["neighbors"] or [],
            }
        return out

    def get_edge_subgraph(self, rel_ids: List[str]) -> List[Dict]:
        """
        Query the subgraph around a batch of relationships.
        - takes rel_ids
        - returns the relationships themselves plus their source/target nodes
        """
        if not rel_ids: return []
        query = f"""
        UNWIND $rids AS rid
        MATCH (a:{self.cfg.entity_label})-[r:{self.cfg.rel_type}]->(b:{self.cfg.entity_label})
        WHERE r.id = rid
        WITH DISTINCT r, a, b
        RETURN {{
          rel_id: r.id,
          rel_desc: r.description,
          rel_keywords: r.keywords,
          rel_strength: r.strength,
          source_id: a.id,
          source_name: a.name,
          source_type: a.type,
          source_desc: a.description,
          target_id: b.id,
          target_name: b.name,
          target_type: b.type,
          target_desc: b.description
        }} AS edge_info
        """
        recs = self._run_read(query, {"rids": rel_ids})
        return [rec["edge_info"] for rec in recs]

    # ---------- admin ----------
    def clear_all(self) -> None:
        """
        Wipe the entire graph.
        On ServiceUnavailable -- e.g. the container has just restarted and the
        connection is broken -- reconnect once and retry.
        """
        last_exc = None
        for _ in range(2):  # two attempts at most
            try:
                self._run_write("MATCH (n) DETACH DELETE n", {})
                return
            except ServiceUnavailable as e:
                last_exc = e
                # The connection is defunct; reconnect and try once more
                self.reconnect()
        raise RuntimeError("Failed to clear_all after reconnect") from last_exc

    # ---------- low-level helpers ----------
    def _run_read(self, cypher: str, params: dict) -> list:
        """Execute a read query and materialize all records."""
        self._ensure_open()
        with self._driver.session() as sess:
            res = sess.run(cypher, **params)
            return list(res)

    def _run_write(self, cypher: str, params: dict) -> Any:
        """Execute a write query and return its first record if present."""
        self._ensure_open()
        with self._driver.session() as sess:
            rec = sess.run(cypher, **params).single()
            return rec

    def _run_write_batch(self, statements: List[str]) -> None:
        """Execute a sequence of schema or write statements in one session."""
        self._ensure_open()
        with self._driver.session() as sess:
            for s in statements:
                sess.run(s)

    def _ensure_open(self) -> None:
        """Open the driver on demand before issuing a query."""
        if self._driver is None:
            self.open()

# --- factory by env (convenience for server/main) ---
def graph_from_env(entity_label: str = "Entity", rel_type: str = "KG_REL") -> Graph:
    """
    Read the settings from the environment (NEO4J_URI, NEO4J_USERNAME,
    NEO4J_PASSWORD) and build a Graph object.
    """
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME")
    pwd  = os.environ.get("NEO4J_PASSWORD")
    if not all([uri, user, pwd]):
        raise RuntimeError("Missing NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD in environment.")
    return Graph(GraphConfig(uri, user, pwd, entity_label, rel_type))
