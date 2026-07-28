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
    封裝 Neo4j 的操作
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
        強制丟掉舊 driver，重新建立連線
        （給你在 snapshot 之後、或遇到 defunct connection 時呼叫）
        """
        try:
            if self._driver:
                self._driver.close()
        except Exception:
            pass
        self._driver = None
        return self.open()

    # 讓你可以 with Graph(cfg) as g:
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
        將一批實體同步進 Neo4j  
        - 新的 entity → 建立  
        - 已存在 → 更新屬性（保留舊 description，僅在有新內容時覆蓋）
        回傳成功處理的筆數
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
        將一批關係 (relationships) 同步進 Neo4j  
        - 新的關係 → 建立  
        - 已存在 → 更新屬性  
        回傳成功處理的筆數
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
        查詢一批 entity 的子圖  
        - 找出指定 id 的節點  
        - 把它的 neighbors（關聯的節點 + 關係資訊）一起返回
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
        查詢一批 relationship 的子圖  
        - 輸入 rel_ids  
        - 返回關係本身與 source/target 節點資訊
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
        清空整個圖。  
        如果遇到 ServiceUnavailable（例如 container 剛重啟，連線壞掉），
        會重連一次再試。
        """
        last_exc = None
        for _ in range(2):  # 最多試兩次
            try:
                self._run_write("MATCH (n) DETACH DELETE n", {})
                return
            except ServiceUnavailable as e:
                last_exc = e
                # 連線已經 defunct，重連再試一次
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

# --- factory by env (方便 server/main 使用) ---
def graph_from_env(entity_label: str = "Entity", rel_type: str = "KG_REL") -> Graph:
    """
    從環境變數 (NEO4J_URI, USERNAME, PASSWORD) 讀取設定，  
    建立 Graph 物件
    """
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME")
    pwd  = os.environ.get("NEO4J_PASSWORD")
    if not all([uri, user, pwd]):
        raise RuntimeError("Missing NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD in environment.")
    return Graph(GraphConfig(uri, user, pwd, entity_label, rel_type))
