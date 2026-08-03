"""Per-session ingest-state snapshots for locomo-plus.

Snapshot layout under <run_root>:
  artifacts/conv<conv_id>/session<sess_id>/
    entities_chroma/
    relationships_chroma/
    summaries_chroma/
    entities_bm25.pkl
    entities_cache.pkl
    relationships_cache.pkl
    entities_meta.jsonl
    relationships_meta.jsonl
    summaries_meta.jsonl
    graph_export.json    # JSON export of FalkorDB graph (entities + relationships)
    snapshot_meta.json   # {conv_id, session_id, created_at}

Usage contract:
  - Call save_snapshot() AFTER MGR.flush_persist() to guarantee VDB files are on disk.
  - Call load_snapshot_files_only() BEFORE importing KG.pipeline.factory so that
    the VDB is initialized from the snapshot state, not from an empty artifacts dir.
  - After importing pipeline, call restore_graph() to reload the FalkorDB graph.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from locomo.utils.graph import (
    ARTIFACTS_SRC,
    GRAPH_EXPORT_FILE,
    SNAPSHOT_META_FILE,
    restore_graph_from_export_file,
    validate_vdb_artifacts,
    write_graph_export,
)
from locomo.utils.log import log_event
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def snapshot_dir(run_root: Path, conv_id: str, session_id: int) -> Path:
    return run_root / "artifacts" / f"{conv_id}" / f"session{session_id}"


def snapshot_exists(run_root: Path, conv_id: str, session_id: int) -> bool:
    return (snapshot_dir(run_root, conv_id, session_id) / SNAPSHOT_META_FILE).exists()


def highest_existing_snapshot(
    run_root: Path, conv_id: str, session_ids: List[int]
) -> int:
    """Return the highest session_id whose snapshot exists (contiguous from 1), or 0."""
    highest = 0
    for sid in sorted(session_ids):
        if snapshot_exists(run_root, conv_id, sid):
            highest = sid
        else:
            break
    return highest


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_snapshot(run_root: Path, conv_id: str, session_id: int, graph) -> Path:
    """Copy KG/storage/artifacts + export FalkorDB graph to snapshot dir.

    Call AFTER MGR.flush_persist() so all VDB files are on disk.
    Writes to a tmp directory first and renames atomically to prevent half-written snapshots.
    """
    dst = snapshot_dir(run_root, conv_id, session_id)
    tmp_dst = dst.parent / f".tmp_{dst.name}"

    if tmp_dst.exists():
        shutil.rmtree(tmp_dst)

    try:
        tmp_dst.mkdir(parents=True, exist_ok=True)

        # 1) Copy VDB artifacts
        if ARTIFACTS_SRC.exists():
            shutil.copytree(ARTIFACTS_SRC, tmp_dst, dirs_exist_ok=True)

        # 2) Validate copied VDB files before writing graph export
        validate_vdb_artifacts(tmp_dst)

        # 3) Export FalkorDB graph as JSON; validate=True raises on invalid file
        result = write_graph_export(tmp_dst / GRAPH_EXPORT_FILE, graph, validate=True)
        if result is None:
            raise RuntimeError(
                f"Graph export returned None — FalkorDB unreachable or empty: "
                f"conv_id={conv_id} session_id={session_id}"
            )

        # 4) Write metadata
        (tmp_dst / SNAPSHOT_META_FILE).write_text(
            json.dumps(
                {
                    "conv_id": conv_id,
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # 5) Atomic rename — dst is only visible after all writes succeed
        if dst.exists():
            shutil.rmtree(dst)
        tmp_dst.rename(dst)
        return dst
    except Exception:
        logger.exception(
            "Graph snapshot save failed: conv_id=%s session_id=%s dst=%s src=%s",
            conv_id,
            session_id,
            dst,
            ARTIFACTS_SRC,
        )
        shutil.rmtree(tmp_dst, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Load (two-phase: files first, then graph)
# ---------------------------------------------------------------------------

def load_snapshot_files_only(run_root: Path, conv_id: str, session_id: int) -> None:
    """Copy snapshot VDB files into KG/storage/artifacts.

    MUST be called before importing KG.pipeline.factory (and therefore before
    any VDB or ChromaDB clients are created), so the pipeline initializes from
    the correct on-disk state.
    """
    src = snapshot_dir(run_root, conv_id, session_id)
    if not (src / SNAPSHOT_META_FILE).exists():
        raise FileNotFoundError(
            f"Snapshot not found: conv_id={conv_id!r} session_id={session_id}. "
            "Run the snapshot builder first."
        )
    if ARTIFACTS_SRC.exists():
        shutil.rmtree(ARTIFACTS_SRC)
    ARTIFACTS_SRC.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src,
        ARTIFACTS_SRC,
        ignore=shutil.ignore_patterns(GRAPH_EXPORT_FILE, SNAPSHOT_META_FILE),
        dirs_exist_ok=True,
    )


def restore_graph(run_root: Path, conv_id: str, session_id: int, graph) -> None:
    """Restore FalkorDB graph from the snapshot's JSON export.

    Call AFTER importing pipeline (so the graph connection is open).
    """
    src = snapshot_dir(run_root, conv_id, session_id)
    export_path = src / GRAPH_EXPORT_FILE
    restore_graph_from_export_file(graph, export_path)


# ---------------------------------------------------------------------------
# locomo-plus snapshot builder (entry point called by cli.py dispatcher)
# ---------------------------------------------------------------------------

def _resolve_conv_id_and_sessions(
    sample_index: int,
    qa_item: dict,
    source_json_path: "Path",
) -> tuple:
    """Validate conversation_id and injected_session_id; load source session records.

    Returns (conv_id, injected_session_id_or_None, source_session_records, is_cognitive).
    Raises ValueError on validation failure.
    """
    from locomo.helpers import (
        index_source_conversations,
        build_session_records_for_conv,
        is_cognitive_item,
    )

    conv_id = qa_item.get("conversation_id")
    if not conv_id:
        raise ValueError(
            f"sample_index={sample_index}: 'conversation_id' is missing or null. "
            "Update the locomo-plus dataset to include conversation_id for each item."
        )
    conv_id = str(conv_id).strip()

    is_cognitive = is_cognitive_item(qa_item)
    injected_session_id: Optional[int] = None

    if is_cognitive:
        raw_inj = qa_item.get("injected_session_id")
        if raw_inj is None:
            raise ValueError(
                f"sample_index={sample_index}: category=Cognitive but 'injected_session_id' is missing."
            )
        try:
            injected_session_id = int(raw_inj)
        except (TypeError, ValueError):
            raise ValueError(
                f"sample_index={sample_index}: 'injected_session_id' must be an integer, "
                f"got {raw_inj!r}"
            )
        if injected_session_id < 1:
            raise ValueError(
                f"sample_index={sample_index}: 'injected_session_id' must be >= 1, "
                f"got {injected_session_id}"
            )

    source_convs = index_source_conversations(source_json_path)
    if conv_id not in source_convs:
        available = sorted(source_convs.keys())[:10]
        raise ValueError(
            f"sample_index={sample_index}: conversation_id={conv_id!r} not found in "
            f"{source_json_path}. Available (first 10): {available}"
        )

    session_records = build_session_records_for_conv(conv_id, source_convs[conv_id])
    return conv_id, injected_session_id, session_records, is_cognitive


def _snapshot_builder(args) -> None:
    """Build per-session snapshots for one conversation.

    Designed to run in a fresh subprocess. Resumes from the highest existing
    snapshot so partial builds are not wasted.
    """
    # locomo/snapshot.py → .parent = locomo/ → .parent = experiment/
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    run_root = Path(args.run_root)
    conv_id: str = args.conv_id
    up_to_session: Optional[int] = args.up_to_session  # None → all sessions

    from locomo.helpers import (
        highest_existing_snapshot,
        load_snapshot_files_only,
        restore_graph,
        save_snapshot,
        snapshot_exists,
    )
    from locomo.helpers import (
        build_session_records_for_conv,
        index_source_conversations,
        resolve_dataset_path,
    )

    source_json = resolve_dataset_path(
        dataset="locomo",
        kind="qa_json",
        explicit_path=args.source_json,
    )
    source_convs = index_source_conversations(source_json)
    if conv_id not in source_convs:
        raise SystemExit(f"[SNAP_BUILD] conversation_id={conv_id!r} not in {source_json}")

    records = build_session_records_for_conv(conv_id, source_convs[conv_id])
    all_session_ids = sorted(r["session_id"] for r in records)
    target_ids = [s for s in all_session_ids if up_to_session is None or s <= up_to_session]

    # Find highest contiguous existing snapshot so we can resume
    resume_from = highest_existing_snapshot(run_root, conv_id, target_ids)

    # Load VDB files from last existing snapshot BEFORE importing pipeline
    if resume_from > 0:
        log_event("SNAP_BUILD", "Resuming from existing snapshot", conv=conv_id, session=resume_from)
        load_snapshot_files_only(run_root, conv_id, resume_from)

    # Import pipeline now (VDB initialises from whatever is in ARTIFACTS_SRC)
    from KG.storage import MGR
    from KG.graph.falkordb import graph_from_env
    from KG.pipeline.factory import build_pipeline
    from locomo.stages import ingest

    _pipeline = build_pipeline()
    ingestor = _pipeline["ingestor"]
    graph = _pipeline["graph"]

    if resume_from > 0:
        restore_graph(run_root, conv_id, resume_from, graph)
    else:
        graph.clear_all()
        graph.init_schema()

    # Ingest sessions that are missing snapshots
    records_map = {r["session_id"]: r for r in records}
    for sess_id in target_ids:
        if sess_id <= resume_from:
            continue  # already snapshotted
        if sess_id not in records_map:
            log_event("SNAP_BUILD][WARN", "Session not in source records; skipping", session=sess_id)
            continue

        log_event("SNAP_BUILD", "Ingesting session", conv=conv_id, session=sess_id)
        # chunk_turns must match the run being snapshotted; a mismatch silently
        # shifts every summary_id and the restored artifacts stop lining up.
        df = ingest.session_records_to_df(
            [records_map[sess_id]], conv_id=conv_id, chunk_turns=args.chunk_turns
        )
        ingest.ingest_by_session_one_turn(
            ingestor,
            df,
            prev_k=args.prev_k,
            entity_sim_topk=args.entity_sim_topk,
            entity_sim_threshold=args.entity_sim_threshold,
        )
        MGR.flush_persist()
        snap_path = save_snapshot(run_root, conv_id, sess_id, graph)
        log_event("SNAP_BUILD", "Saved snapshot", path=snap_path)

    graph.close()
    log_event("SNAP_BUILD", "Done", conv=conv_id, up_to=up_to_session)
