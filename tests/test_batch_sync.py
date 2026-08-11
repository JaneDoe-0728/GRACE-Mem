"""
Validation script: test batch sync_entities / sync_relationships against FalkorDB.

Run from repo root:
    python -m test.test_batch_sync

Checks:
1. Entity count matches expected after sync_entities
2. Relationship count matches expected after sync_relationships
3. Property values are preserved correctly
4. MERGE idempotency: running twice produces the same counts
5. Timing logged for comparison vs. baseline serial approach
"""

import os
import sys
import time

# Load .env so NEO4J_URI / GRAPH_NAME are set without exporting them manually
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

from KG.graph.falkordb import Graph, GraphConfig


TEST_GRAPH = os.environ.get("GRAPH_NAME", "memory") + "__batch_test"


def make_graph() -> Graph:
    uri = os.environ.get("NEO4J_URI")
    if not uri:
        print("ERROR: NEO4J_URI not set. Export it or add to .env.")
        sys.exit(1)
    cfg = GraphConfig(uri=uri, user="", password="", graph_name=TEST_GRAPH)
    g = Graph(cfg)
    g.open()
    return g


def count_nodes(g: Graph) -> int:
    rows = g._run_read(f"MATCH (e:{g.cfg.entity_label}) RETURN count(e) AS n", {})
    return int((rows[0] or {}).get("n", 0)) if rows else 0


def count_rels(g: Graph) -> int:
    rows = g._run_read(f"MATCH ()-[r:{g.cfg.rel_type}]->() RETURN count(r) AS n", {})
    return int((rows[0] or {}).get("n", 0)) if rows else 0


def get_entity(g: Graph, eid: str) -> dict:
    rows = g._run_read(
        f"MATCH (e:{g.cfg.entity_label} {{id: $eid}}) RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS desc",
        {"eid": eid},
    )
    return rows[0] if rows else {}


def get_rel(g: Graph, rid: str) -> dict:
    rows = g._run_read(
        f"MATCH ()-[r:{g.cfg.rel_type} {{id: $rid}}]->() RETURN r.id AS id, r.description AS desc, r.strength AS strength",
        {"rid": rid},
    )
    return rows[0] if rows else {}


def run():
    g = make_graph()
    print(f"\n=== Batch sync validation (graph: {TEST_GRAPH}) ===\n")

    # --- Clean slate ---
    g.clear_all()
    assert count_nodes(g) == 0, "Expected empty graph after clear_all"

    # --- Test entities ---
    entity_idx = {
        "k1": {"id": "e1", "name": "Alice", "type": "person", "description": "A researcher"},
        "k2": {"id": "e2", "name": "Bob",   "type": "person", "description": "An engineer"},
        "k3": {"id": "e3", "name": "ACME",  "type": "org",    "description": "A company"},
    }

    t0 = time.monotonic()
    ok = g.sync_entities(entity_idx)
    elapsed_e = time.monotonic() - t0

    assert ok == 3, f"Expected 3 entities written, got {ok}"
    n = count_nodes(g)
    assert n == 3, f"Expected 3 nodes in graph, found {n}"

    alice = get_entity(g, "e1")
    assert alice.get("name") == "Alice", f"Wrong name: {alice}"
    assert alice.get("desc") == "A researcher", f"Wrong desc: {alice}"
    print(f"[PASS] entities: ok={ok}, node_count={n}, elapsed={elapsed_e:.3f}s")

    # --- Idempotency: run again, counts must not grow ---
    ok2 = g.sync_entities(entity_idx)
    n2 = count_nodes(g)
    assert n2 == 3, f"Idempotency failed: node_count grew to {n2}"
    print(f"[PASS] entity idempotency: still {n2} nodes")

    # --- Update semantics: description update ---
    entity_idx["k1"]["description"] = "A senior researcher"
    g.sync_entities({"k1": entity_idx["k1"]})
    alice2 = get_entity(g, "e1")
    assert alice2.get("desc") == "A senior researcher", f"Description not updated: {alice2}"
    print(f"[PASS] entity update: description updated correctly")

    # --- Test relationships ---
    rel_metas = [
        {"id": "r1", "source_id": "e1", "target_id": "e2",
         "description": "Alice knows Bob", "keywords": ["knows"], "strength": 5,
         "source_type": "person", "target_type": "person"},
        {"id": "r2", "source_id": "e1", "target_id": "e3",
         "description": "Alice works at ACME", "keywords": ["works_at"], "strength": 8,
         "source_type": "person", "target_type": "org"},
    ]

    t0 = time.monotonic()
    ok_r = g.sync_relationships(rel_metas)
    elapsed_r = time.monotonic() - t0

    assert ok_r == 2, f"Expected 2 relationships written, got {ok_r}"
    nr = count_rels(g)
    assert nr == 2, f"Expected 2 edges in graph, found {nr}"

    r1 = get_rel(g, "r1")
    assert r1.get("desc") == "Alice knows Bob", f"Wrong rel desc: {r1}"
    assert int(r1.get("strength") or 0) == 5, f"Wrong strength: {r1}"
    print(f"[PASS] relationships: ok={ok_r}, rel_count={nr}, elapsed={elapsed_r:.3f}s")

    # --- Rel idempotency ---
    ok_r2 = g.sync_relationships(rel_metas)
    nr2 = count_rels(g)
    assert nr2 == 2, f"Rel idempotency failed: count grew to {nr2}"
    print(f"[PASS] relationship idempotency: still {nr2} edges")

    # --- Missing endpoint: should not crash or create dangling edge ---
    bad_rel = [{"id": "r_bad", "source_id": "e_missing", "target_id": "e1",
                "description": "bad", "keywords": None, "strength": None,
                "source_type": None, "target_type": None}]
    ok_bad = g.sync_relationships(bad_rel)
    nr3 = count_rels(g)
    assert nr3 == 2, f"Bad rel created a dangling edge: count={nr3}"
    print(f"[PASS] missing-endpoint guard: no extra edge (ok_bad={ok_bad})")

    # --- Cleanup ---
    g.clear_all()
    g.close()
    print(f"\n=== ALL CHECKS PASSED ===")
    print(f"  sync_entities elapsed : {elapsed_e:.3f}s  (3 rows, 1 batch query)")
    print(f"  sync_relationships elapsed: {elapsed_r:.3f}s  (2 rows, 1 batch query)")


if __name__ == "__main__":
    run()
