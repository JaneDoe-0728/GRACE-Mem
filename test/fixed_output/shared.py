"""
Shared utilities for the Fixed-Output Audit Suite (EXP-F01 .. EXP-F08).

Provides:
  - Canonicalization helpers (strip non-semantic fields, sort lists, recursive key sort)
  - SHA-256 hashing over canonical JSON
  - Report construction, status computation, and file writing
  - Connectivity probes for FalkorDB and LM Studio
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Root paths ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "test" / "fixed_output" / "results"

# Non-semantic fields removed before hashing
_STRIP_FIELDS = frozenset({
    "timestamp", "ts", "created_at", "updated_at",
    "run_id", "request_id", "temp_path", "runtime_ms",
    "elapsed_sec", "log_metadata",
})

# ── Canonicalization ─────────────────────────────────────────────────────────

def strip_non_semantic(record: Dict[str, Any], extra: frozenset[str] = frozenset()) -> Dict[str, Any]:
    """Remove non-semantic fields from a flat dict."""
    drop = _STRIP_FIELDS | extra
    return {k: v for k, v in record.items() if k not in drop}


def canonical_json(obj: Any) -> str:
    """Recursively sort dict keys, then serialize to compact JSON."""
    return json.dumps(_sort_recursive(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sort_recursive(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sort_recursive(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_sort_recursive(item) for item in obj]
    return obj


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_obj(obj: Any) -> str:
    """Canonical JSON → SHA-256 hex."""
    return sha256_hex(canonical_json(obj))


# ── Entity / relation / summary canonicalization ─────────────────────────────

def canonical_entities(rows: List[Dict]) -> List[Dict]:
    """Keep semantic fields; sort by (id, name)."""
    out = []
    for r in rows:
        out.append({
            "id":          str(r.get("id") or r.get("stable_id") or ""),
            "name":        str(r.get("name") or ""),
            "type":        str(r.get("type") or ""),
            "description": str(r.get("description") or ""),
        })
    return sorted(out, key=lambda e: (e["id"], e["name"]))


def canonical_relations(rows: List[Dict]) -> List[Dict]:
    """Keep semantic fields; sort by (source_id, predicate, target_id, session_id, message_id)."""
    out = []
    for r in rows:
        out.append({
            "source_id":   str(r.get("source_id") or r.get("source") or ""),
            "predicate":   str(r.get("predicate") or ""),
            "target_id":   str(r.get("target_id") or r.get("target") or ""),
            "description": str(r.get("description") or ""),
            "session_id":  str(r.get("session_id") or ""),
            "message_id":  str(r.get("message_id") or ""),
        })
    return sorted(out, key=lambda r: (
        r["source_id"], r["predicate"], r["target_id"],
        r["session_id"], r["message_id"],
    ))


def canonical_summaries(rows: List[Dict]) -> List[Dict]:
    """Keep semantic fields; sort by (session_id, message_id)."""
    out = []
    for r in rows:
        out.append({
            "session_id":  str(r.get("session_id") or ""),
            "message_id":  str(r.get("message_id") or ""),
            "text":        str(r.get("summary_text") or r.get("text") or ""),
        })
    return sorted(out, key=lambda s: (s["session_id"], s["message_id"]))


def canonical_candidates(items: List[Dict], *, id_key: str, score_key: str = "score") -> List[Dict]:
    """Sort by (-score, id) per spec rule 5."""
    out = []
    for item in items:
        out.append({
            "id":    str(item.get(id_key) or ""),
            "score": float(item.get(score_key) or 0.0),
        })
    return sorted(out, key=lambda c: (-c["score"], c["id"]))


# ── Report construction ───────────────────────────────────────────────────────

_DEFAULT_EMBEDDER = "qwen3-embedding-0.6b"
_DEFAULT_RERANKER = "qwen3-reranker-0.6b"


def make_base_report(
    exp_name: str,
    *,
    repeat_count: int,
    concurrency: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    seed: int = 42,
    deterministic: bool = True,
    config_snapshot: Optional[Dict] = None,
    llm_url: str = "",
    llm_model: str = "",
) -> Dict[str, Any]:
    model_backend = {
        "llm":      f"{llm_model} @ {llm_url}" if llm_url or llm_model else "N/A",
        "embedder": _DEFAULT_EMBEDDER,
        "reranker": _DEFAULT_RERANKER,
    }
    return {
        "experiment_name":  exp_name,
        "run_start_utc":    datetime.now(timezone.utc).isoformat(),
        "config_snapshot":  config_snapshot or {},
        "model_backend":    model_backend,
        "seed":             seed,
        "deterministic":    deterministic,
        "repeat_count":     repeat_count,
        "concurrency":      concurrency,
        "temperature":      temperature,
        "top_p":            top_p,
        "max_tokens":       max_tokens,
        "trials":           [],
        "unique_hash_counts": {},
        "status":           "PENDING",
        "warnings":         [],
        "failure_diagnosis": [],
    }


def compute_unique_hash_counts(trials: List[Dict[str, str]]) -> Dict[str, int]:
    """Count unique values per hash key across all trials."""
    all_keys: set[str] = set()
    for t in trials:
        all_keys.update(t.get("artifact_hashes", {}).keys())
    counts: Dict[str, int] = {}
    for key in sorted(all_keys):
        values = {t["artifact_hashes"].get(key) for t in trials if t.get("artifact_hashes")}
        counts[key] = len(values - {None})
    return counts


def compute_status(
    unique_hash_counts: Dict[str, int],
    *,
    primary_keys: List[str],
    warn_keys: Optional[List[str]] = None,
) -> str:
    """
    PASS  – all primary keys have unique_count == 1
    WARN  – primary identical; some warn_keys have unique_count > 1
    FAIL  – any primary key has unique_count > 1
    """
    for k in primary_keys:
        if unique_hash_counts.get(k, 0) > 1:
            return "FAIL"
    if warn_keys:
        for k in warn_keys:
            if unique_hash_counts.get(k, 0) > 1:
                return "WARN"
    # Check remaining keys
    remaining = {k for k in unique_hash_counts if k not in primary_keys and k not in (warn_keys or [])}
    for k in remaining:
        if unique_hash_counts.get(k, 0) > 1:
            return "WARN"
    return "PASS"


def finalize_report(
    report: Dict[str, Any],
    *,
    primary_keys: List[str],
    warn_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    uhc = compute_unique_hash_counts(report["trials"])
    report["unique_hash_counts"] = uhc
    report["status"] = compute_status(uhc, primary_keys=primary_keys, warn_keys=warn_keys)
    return report


def run_output_dir(run_tag: Optional[str] = None) -> Path:
    """Return the output directory for one fixed-output run tag."""
    tag = run_tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_report(report: Dict[str, Any], exp_id: str, run_tag: Optional[str] = None) -> Path:
    out_dir = run_output_dir(run_tag)
    path = out_dir / f"{exp_id}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ── Connectivity probes ───────────────────────────────────────────────────────

def probe_falkordb(host: str = "localhost", port: int = 6379, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe_lm_studio(base_url: str = "http://localhost:1234", timeout: float = 5.0) -> bool:
    try:
        url = base_url.rstrip("/") + "/v1/models"
        req = urllib.request.Request(url, headers={"Authorization": "Bearer dummy"})
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


# ── LM Studio OpenAI-compat call ─────────────────────────────────────────────

def lm_studio_chat(
    messages: List[Dict[str, str]],
    *,
    base_url: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    api_key: str = "dummy",
    timeout: float = 120.0,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "top_p":       top_p,
        "max_tokens":  max_tokens,
        "seed":        seed,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


# ── FalkorDB canonical export ─────────────────────────────────────────────────

def export_falkordb_canonical(graph) -> Dict[str, Any]:
    """
    Export all Entity nodes and KG_REL edges from FalkorDB as sorted dicts.
    Strips non-semantic fields before returning.
    """
    node_rows = graph._run_read(
        "MATCH (e:Entity) RETURN e.id AS id, e.name AS name, "
        "e.type AS type, e.description AS description ORDER BY e.id",
        {},
    )
    rel_rows = graph._run_read(
        "MATCH (s:Entity)-[r:KG_REL]->(t:Entity) "
        "RETURN s.id AS source_id, r.predicate AS predicate, t.id AS target_id, "
        "r.description AS description, r.session_id AS session_id, r.message_id AS message_id "
        "ORDER BY s.id, r.predicate, t.id",
        {},
    )
    return {
        "nodes": sorted(
            [{k: str(v or "") for k, v in row.items()} for row in node_rows],
            key=lambda r: r.get("id", ""),
        ),
        "edges": sorted(
            [{k: str(v or "") for k, v in row.items()} for row in rel_rows],
            key=lambda r: (r.get("source_id", ""), r.get("predicate", ""), r.get("target_id", "")),
        ),
    }


def export_chroma_all(vdb) -> List[Dict]:
    """Dump all metadata records from a ChromaDB collection."""
    from KG.storage.chroma_vdb import _deserialize_metadata  # noqa: PLC0415
    with vdb._lock:
        results = vdb._collection.get(include=["metadatas"])
    ids = results.get("ids") or []
    metas = results.get("metadatas") or []
    out = []
    for mid, meta in zip(ids, metas):
        rec = _deserialize_metadata(meta) if meta else {}
        if "id" not in rec:
            rec["id"] = mid
        out.append(rec)
    return out


# ── CLI run-tag helper ────────────────────────────────────────────────────────

def run_tag_from_argv() -> Optional[str]:
    """Return first CLI arg if provided, else None (auto-timestamp)."""
    return sys.argv[1] if len(sys.argv) > 1 else None
