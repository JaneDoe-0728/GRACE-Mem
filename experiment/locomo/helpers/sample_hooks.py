import shutil
import subprocess
import sys
from pathlib import Path

from grace_mem.storage.cache import CacheStore
from experiment.locomo.utils.graph import (
    ARTIFACTS_SRC,
    GRAPH_EXPORT_FILE,
    restore_graph_from_export_file,
    write_graph_export,
)
from experiment.locomo.utils.log import log_event

PIPELINE_MODULE = "experiment.locomo.pipeline.runner"


def ensure_worker_repo_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.append(repo_root_str)


def artifact_dir_for_sample(args) -> Path | None:
    artifact_dir = getattr(args, "artifact_dir", None)
    if artifact_dir is None:
        return None
    return Path(artifact_dir) / f"sample_{args.sample_index}" / "artifacts"


def restore_artifacts_from_dir(artifact_dir: Path) -> None:
    log_event("ARTIFACT", "Loading artifacts", path=artifact_dir)
    if not artifact_dir.exists():
        raise FileNotFoundError(f"--artifact-dir not found: {artifact_dir}")
    if ARTIFACTS_SRC.exists():
        shutil.rmtree(ARTIFACTS_SRC)
    ARTIFACTS_SRC.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        artifact_dir,
        ARTIFACTS_SRC,
        ignore=shutil.ignore_patterns(GRAPH_EXPORT_FILE),
        dirs_exist_ok=True,
    )


def reload_mgr_state_from_artifacts(mgr) -> None:
    """Reset in-memory storage handles and reload cache from the restored artifacts dir."""
    mgr.reset_all(delete_files=False)
    reloaded_cache = CacheStore.load(cache_dir=mgr.ART)
    mgr.cache.update(reloaded_cache)
    log_event(
        "ARTIFACT",
        "Reloaded MGR state from restored artifacts",
        path=mgr.ART,
        entities=len(mgr.cache.get("entities", {})),
        relationships=len(mgr.cache.get("relationships", {})),
    )


def restore_graph_from_artifact_dir(graph, artifact_dir: Path) -> None:
    graph_export = artifact_dir / GRAPH_EXPORT_FILE
    if graph_export.exists():
        log_event("ARTIFACT", "Restoring FalkorDB graph", path=graph_export)
        restore_graph_from_export_file(graph, graph_export)
        return

    log_event("ARTIFACT", "No graph_export.json; initialising empty graph", path=artifact_dir)
    graph.clear_all()
    graph.init_schema()


def export_graph_to_artifacts(graph, *, sample_index: int | None = None) -> None:
    export_path = ARTIFACTS_SRC / GRAPH_EXPORT_FILE
    if write_graph_export(export_path, graph) is None:
        return

    if sample_index is None:
        log_event("ARTIFACT", "Exported FalkorDB graph", path=export_path)
        return
    log_event("ARTIFACT", "Exported FalkorDB graph", sample=sample_index, path=export_path)


def validate_and_export_graph(graph, *, sample_index: int) -> None:
    export_path = ARTIFACTS_SRC / GRAPH_EXPORT_FILE
    graph_data = write_graph_export(export_path, graph, validate=True)
    if graph_data is None:
        raise RuntimeError(
            f"Graph export returned None for sample {sample_index} — "
            "FalkorDB may be unreachable or the graph was never written"
        )

    log_event(
        "GRAPH", "Exported",
        entities=len(graph_data["entities"]),
        relationships=len(graph_data["relationships"]),
        path=export_path,
    )


def invoke_snapshot_builder(*, args, conv_id: str, max_session_id: int, run_root: Path) -> None:
    snap_cmd = [
        sys.executable,
        "-m",
        PIPELINE_MODULE,
        "--build-snapshots",
        "--conv-id",
        conv_id,
        "--up-to-session",
        str(max_session_id),
        "--run-root",
        str(run_root),
        "--prev-k",
        str(args.prev_k),
        "--entity-sim-topk",
        str(args.entity_sim_topk),
        "--entity-sim-threshold",
        str(args.entity_sim_threshold),
    ]
    if args.source_json:
        snap_cmd.extend(["--source-json", str(args.source_json)])

    result = subprocess.run(snap_cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"On-demand snapshot builder failed for conv={conv_id} (exit {result.returncode})"
        )
