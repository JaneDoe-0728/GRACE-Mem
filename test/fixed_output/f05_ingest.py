"""
EXP-F05 — KG Ingest Determinism

Verifies that ingesting the same conversation corpus from a fresh state produces
identical graph artifacts across independent trials.

Protocol per trial:
  1. python refresh_system.py   (clear all state)
  2. Ingest first 3 sessions of locomo10.json
  3. Export canonical artifacts and hash
  Repeat ≥ 3 times.

Blocking dependency: EXP-F02-a must PASS before these results are meaningful.

Usage:
    python test/exp_f05_ingest.py [run-tag]

Writes:
    test/fixed_output/results/<run-tag>/EXP-F05.json
    test/fixed_output/results/<run-tag>/snapshot.json  (last trial)
"""
from __future__ import annotations

import json
import subprocess
import sys
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiment"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment.common.reproducibility import activate_reproducibility
from experiment.experiment_config import INGEST_PARAMS, REPRODUCIBILITY_PARAMS
from shared import (
    canonical_entities,
    canonical_json,
    canonical_relations,
    canonical_summaries,
    export_chroma_all,
    export_falkordb_canonical,
    finalize_report,
    make_base_report,
    probe_falkordb,
    run_output_dir,
    sha256_hex,
    write_report,
)

EXP_ID       = "EXP-F05"
SEED         = 42
REPEAT       = 3        # spec: ≥ 3 trials
NUM_SESSIONS = 3        # first N sessions of locomo10.json
REPO_ROOT    = Path(__file__).resolve().parents[2]
LOCOMO_JSON  = REPO_ROOT / "experiment" / "locomo" / "data" / "locomo10.json"

_STRIP_INGEST = frozenset({"created_at", "updated_at", "run_id", "temp_path", "ts"})


# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_first_n_sessions(n: int) -> List[Dict]:
    """Return first n session records from locomo10.json using the project helpers."""
    from experiment.locomo.helpers.dataset import build_session_records_from_json
    records = build_session_records_from_json(LOCOMO_JSON)
    return records[:n]


def build_ingest_df(session_records: List[Dict]):
    import pandas as pd
    from experiment.locomo.stages.ingest import sessions_to_one_turn_df
    return sessions_to_one_turn_df(session_records, make_session_uid=True)


# ── State management ──────────────────────────────────────────────────────────

def clear_state(warnings: List[str]) -> bool:
    """Run refresh_system.py to clear all pipeline state."""
    try:
        from KG.storage import MGR
        # This test process reuses the storage singleton across trials.
        # Close local Chroma clients before the subprocess deletes artifacts.
        MGR.reset_all(delete_files=False)
    except Exception as exc:
        warnings.append(f"local MGR reset before refresh_system.py failed: {exc}")

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "refresh_system.py")],
        cwd=str(REPO_ROOT),
        input="yes\n",
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        warnings.append(f"refresh_system.py failed (rc={result.returncode}): {result.stderr[:500]}")
        return False
    time.sleep(1.0)  # give services time to settle after wipe
    return True


# ── Canonical artifact export ─────────────────────────────────────────────────

def export_canonical_artifacts(pipeline: Dict[str, Any]) -> Dict[str, Any]:
    mgr   = pipeline["mgr"]
    graph = pipeline["graph"]

    mgr.flush_persist()

    # Entities from ChromaDB (use private attrs; they are populated by ingest)
    raw_ents = export_chroma_all(mgr._entities_vdb) if mgr._entities_vdb else []
    ents = canonical_entities([{k: v for k, v in r.items() if k not in _STRIP_INGEST} for r in raw_ents])

    # Relations from ChromaDB
    raw_rels = export_chroma_all(mgr._relationships_vdb) if mgr._relationships_vdb else []
    rels = canonical_relations([{k: v for k, v in r.items() if k not in _STRIP_INGEST} for r in raw_rels])

    # Summaries from ChromaDB
    raw_sums = export_chroma_all(mgr._summaries_vdb) if mgr._summaries_vdb else []
    sums = canonical_summaries([{k: v for k, v in r.items() if k not in _STRIP_INGEST} for r in raw_sums])

    # FalkorDB full export
    graph_export = export_falkordb_canonical(graph)

    return {
        "entities":    ents,
        "relations":   rels,
        "summaries":   sums,
        "graph_export": graph_export,
        "counts": {
            "entity_count":   len(ents),
            "relation_count": len(rels),
            "summary_count":  len(sums),
        },
    }


def hash_artifacts(artifacts: Dict[str, Any]) -> Dict[str, str]:
    return {
        "entity_table_hash":   sha256_hex(canonical_json(artifacts["entities"])),
        "relation_table_hash": sha256_hex(canonical_json(artifacts["relations"])),
        "summary_table_hash":  sha256_hex(canonical_json(artifacts["summaries"])),
        "graph_export_hash":   sha256_hex(canonical_json(artifacts["graph_export"])),
    }


# ── Trial runner ──────────────────────────────────────────────────────────────

def run_trial(
    trial_id: int,
    session_records: List[Dict],
    warnings: List[str],
    failure_diagnosis: List[str],
) -> Dict[str, Any]:
    activate_reproducibility(seed=SEED, deterministic=True)

    from KG.pipeline.factory import build_pipeline
    from experiment.locomo.stages.ingest import ingest_by_session_one_turn

    pipeline = build_pipeline()
    ingestor = pipeline["ingestor"]

    df = build_ingest_df(session_records)
    ingest_by_session_one_turn(
        ingestor,
        df,
        prev_k=INGEST_PARAMS.get("prev_k", 2),
        entity_sim_topk=INGEST_PARAMS.get("entity_sim_topk", 3),
        entity_sim_threshold=INGEST_PARAMS.get("entity_sim_threshold", 0.6),
    )

    artifacts = export_canonical_artifacts(pipeline)
    hashes    = hash_artifacts(artifacts)

    pipeline["graph"].close()

    return {
        "trial_id":       trial_id,
        "entity_count":   artifacts["counts"]["entity_count"],
        "relation_count": artifacts["counts"]["relation_count"],
        "summary_count":  artifacts["counts"]["summary_count"],
        "artifact_hashes": hashes,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

_DEFAULT_LLM_URL   = "http://localhost:1234/v1"
_DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",       default=None,               help="Run-tag for report path")
    parser.add_argument("--llm-url",   default=_DEFAULT_LLM_URL,   dest="llm_url",   help="LM Studio base URL")
    parser.add_argument("--llm-model", default=_DEFAULT_LLM_MODEL, dest="llm_model", help="LLM model name")
    args = parser.parse_args()

    llm_url   = args.llm_url
    llm_model = args.llm_model

    # The ingest pipeline calls llm_post() which reads these env vars
    import os
    os.environ["LLM_API"]    = llm_url
    os.environ["MODEL_NAME"] = llm_model

    if not probe_falkordb():
        report = make_base_report(EXP_ID, repeat_count=REPEAT, llm_url=llm_url, llm_model=llm_model)
        report["status"] = "SKIP"
        report["warnings"].append("FalkorDB not reachable on localhost:6379")
        path = write_report(report, EXP_ID, args.tag)
        print(json.dumps(report, indent=2))
        print(f"\n[{EXP_ID}] SKIP  report → {path}", file=sys.stderr)
        return 0

    session_records = load_first_n_sessions(NUM_SESSIONS)
    if not session_records:
        print(f"[{EXP_ID}] ERROR: could not load locomo sessions from {LOCOMO_JSON}", file=sys.stderr)
        return 1

    run_tag = args.tag
    report = make_base_report(
        EXP_ID,
        repeat_count=REPEAT,
        config_snapshot={
            "seed":          SEED,
            "num_sessions":  NUM_SESSIONS,
            "locomo_json":   str(LOCOMO_JSON),
            "llm_base_url":  llm_url,
            "llm_model":     llm_model,
            **INGEST_PARAMS,
            **REPRODUCIBILITY_PARAMS,
        },
        llm_url=llm_url,
        llm_model=llm_model,
    )
    warnings: List[str]          = []
    failure_diagnosis: List[str] = []

    last_artifacts: Optional[Dict] = None

    for trial_id in range(REPEAT):
        print(f"[{EXP_ID}] Trial {trial_id + 1}/{REPEAT}: clearing state…", file=sys.stderr)
        if not clear_state(warnings):
            failure_diagnosis.append(f"Trial {trial_id}: refresh_system.py failed")
            continue

        print(f"[{EXP_ID}] Trial {trial_id + 1}/{REPEAT}: ingesting {NUM_SESSIONS} sessions…", file=sys.stderr)
        try:
            trial = run_trial(trial_id, session_records, warnings, failure_diagnosis)
            report["trials"].append(trial)

            # Stash last trial artifacts as snapshot for F06
            if trial_id == REPEAT - 1:
                activate_reproducibility(seed=SEED, deterministic=True)
                from KG.pipeline.factory import build_pipeline
                pipeline = build_pipeline()
                last_artifacts = export_canonical_artifacts(pipeline)
                pipeline["graph"].close()

        except Exception as exc:
            failure_diagnosis.append(f"Trial {trial_id} failed: {exc}")
            import traceback; traceback.print_exc()

    report["warnings"]          = warnings
    report["failure_diagnosis"] = failure_diagnosis

    finalize_report(
        report,
        primary_keys=["entity_table_hash", "relation_table_hash",
                      "summary_table_hash", "graph_export_hash"],
    )

    # Write main report
    path = write_report(report, EXP_ID, run_tag)

    # Write snapshot for F06 re-use
    if last_artifacts:
        snap_path = run_output_dir(run_tag) / "snapshot.json"
        snap_path.write_text(json.dumps(last_artifacts, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{EXP_ID}] snapshot → {snap_path}", file=sys.stderr)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[{EXP_ID}] status={report['status']}  report → {path}", file=sys.stderr)
    return 0 if report["status"] in ("PASS", "WARN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
