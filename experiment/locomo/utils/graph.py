"""Export and restore the knowledge graph as a JSON snapshot.

A run evaluates many samples against the same conversation, and rebuilding the
graph per sample is the dominant cost. Exporting once and restoring per sample
removes it.

The exported file is therefore load-bearing rather than a convenience, and it
is validated on both sides: `validate_graph_export` before restoring, and
`validate_vdb_artifacts` on the vector stores that accompany it. A truncated
export restored without checking produces a partial graph, which does not fail
-- it quietly lowers recall for every question in the sample and looks like a
retrieval regression.

ARTIFACTS_SRC resolves KG_ARTIFACTS_DIR at import time, so each worker process
must set that variable before importing this module or they will share one
artifacts directory and overwrite each other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from grace_mem.storage.paths import resolve_artifacts_dir
from experiment.locomo.utils.log import log_event

# Working VDB dir, honoring KG_ARTIFACTS_DIR for per-process isolation. Resolved
# at import time; each process must set the env var before it starts.
ARTIFACTS_SRC = resolve_artifacts_dir()
GRAPH_EXPORT_FILE = "graph_export.json"
SNAPSHOT_META_FILE = "snapshot_meta.json"


def validate_graph_export(path: Path) -> None:
    """Raise RuntimeError if the graph export at path is missing or structurally invalid."""
    if not path.exists():
        raise RuntimeError(f"graph_export.json missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"graph_export.json invalid: {exc}")
    if not isinstance(data.get("entities"), list):
        raise RuntimeError("graph_export.json: 'entities' key missing or not a list")
    if not isinstance(data.get("relationships"), list):
        raise RuntimeError("graph_export.json: 'relationships' key missing or not a list")
    if not data["entities"]:
        raise RuntimeError(
            f"graph_export.json: entities list is empty — "
            f"graph was not populated (path={path})"
        )


def write_graph_export(path: Path, graph, *, validate: bool = False) -> Optional[dict[str, Any]]:
    """Export graph state to path. Returns the export payload, or None if export failed."""
    export_data = export_graph(graph)
    if export_data is None:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(export_data, ensure_ascii=False), encoding="utf-8")
    if validate:
        validate_graph_export(path)
    return export_data


def restore_graph_from_export_file(graph, export_path: Path) -> bool:
    """Restore graph from export_path. Raises FileNotFoundError when the export file is missing."""
    if not export_path.exists():
        raise FileNotFoundError(
            f"graph_export.json missing — snapshot corrupt: {export_path}"
        )

    data = json.loads(export_path.read_text(encoding="utf-8"))
    _restore_graph(graph, data)
    return True


def validate_vdb_artifacts(base_dir: Path) -> None:
    """Raise RuntimeError if VDB artifact files under base_dir are missing or empty."""
    errors: list[str] = []

    def _check_chroma(label: str, chroma_dir: Path) -> None:
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
            errors.append(f"{label} missing or empty")

    for chroma_name in ("entities_chroma", "relationships_chroma", "summaries_chroma"):
        _check_chroma(chroma_name, base_dir / chroma_name)

    for pkl_name in ("entities_cache.pkl", "relationships_cache.pkl", "entities_bm25.pkl"):
        _check_file(pkl_name, base_dir / pkl_name)

    if errors:
        raise RuntimeError(
            f"VDB artifact validation failed at {base_dir}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def export_graph(graph) -> Optional[dict[str, Any]]:
    """Export FalkorDB graph state. Returns None if FalkorDB is unreachable."""
    return _export_graph(graph)


def _export_graph(graph) -> Optional[dict[str, Any]]:
    """Export all Entity nodes and KG_REL relationships as plain dicts."""
    label = graph.cfg.entity_label
    rel = graph.cfg.rel_type
    try:
        entities: list[dict[str, Any]] = graph._run_read(
            f"MATCH (e:{label}) "
            "RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description",
            {},
        )
        relationships: list[dict[str, Any]] = graph._run_read(
            f"MATCH (a:{label})-[r:{rel}]->(b:{label}) "
            "RETURN r.id AS id, r.description AS description, r.keywords AS keywords, "
            f"r.strength AS strength, a.id AS source_id, b.id AS target_id, "
            f"a.type AS source_type, b.type AS target_type",
            {},
        )
        return {"entities": entities, "relationships": relationships}
    except Exception as exc:
        log_event("SNAPSHOT][WARN", "Graph export failed", error=f"{type(exc).__name__}: {exc}")
        return None


def _restore_graph(graph, data: dict[str, Any]) -> None:
    """Clear graph and re-insert entities/relationships from JSON export."""
    try:
        graph.clear_all()
        graph.init_schema()

        entities_idx: dict[str, Any] = {
            str(entity["id"]): entity
            for entity in (data.get("entities") or [])
            if entity.get("id")
        }
        if entities_idx:
            graph.sync_entities(entities_idx)

        rel_metas: list[dict[str, Any]] = [
            {
                "id": rel_meta.get("id"),
                "source_id": rel_meta.get("source_id"),
                "target_id": rel_meta.get("target_id"),
                "description": rel_meta.get("description"),
                "keywords": rel_meta.get("keywords"),
                "strength": rel_meta.get("strength"),
                "source_type": rel_meta.get("source_type"),
                "target_type": rel_meta.get("target_type"),
            }
            for rel_meta in (data.get("relationships") or [])
            if rel_meta.get("id") and rel_meta.get("source_id") and rel_meta.get("target_id")
        ]
        if rel_metas:
            graph.sync_relationships(rel_metas)
    except Exception as exc:
        log_event("SNAPSHOT][WARN", "Graph restore failed", error=f"{type(exc).__name__}: {exc}")
