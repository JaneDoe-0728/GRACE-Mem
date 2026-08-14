"""Rebuild a graph backend from a cached extraction, skipping re-ingestion.

Ingestion is the expensive phase and its output is deterministic given the
cache, so a rerun that only changes retrieval restores the graph from cached
entities and relationships instead of paying for extraction again. This is what
makes the ablation sweeps affordable.

Entities are synced before relationships, and that order is required: an edge
whose endpoints do not yet exist cannot be written.
"""

from __future__ import annotations

from pathlib import Path


def artifacts_dir(output_dir: Path, dataset_name: str) -> Path:
    return output_dir / f"artifacts_{dataset_name}"


def restore_graph_from_cache(graph, cache: dict) -> None:
    entities = cache.get("entities", {})
    relationships = cache.get("relationships", {})

    print(f"  [RESTORE] Syncing {len(entities)} entities to FalkorDB...")
    graph.sync_entities(entities)

    if relationships:
        rel_values = list(relationships.values())
        print(f"  [RESTORE] Syncing {len(rel_values)} relationships to FalkorDB...")
        graph.sync_relationships(rel_values)

    print("  [RESTORE] Done.")

