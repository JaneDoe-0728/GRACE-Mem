"""Readme conformance tests — local only, not committed.

Encodes every claim in readme.md / experiment/readme.md that was verified by
hand, so a future edit that drifts from the code fails here instead of in a
user's first five minutes.

Layout:
  * Offline tests            — no network, no models, no DB. Always run.
  * Integration tests        — need FalkorDB + an OpenAI-compatible endpoint +
                               the downloaded models. Auto-skip when absent.

Run everything:      uv run pytest test/test_readme_claims.py -v
Offline only:        uv run pytest test/test_readme_claims.py -v -m "not integration"
"""

from __future__ import annotations

import ast
import importlib
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _top_level_names(py_file: Path) -> set[str]:
    """Module-level def/class/assignment names, parsed without importing."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _method_names(py_file: Path, class_name: str) -> set[str]:
    """Method names of one class, parsed without importing."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"class {class_name} not found in {py_file}")


def _cli_help(script: str) -> str:
    proc = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"{script} --help failed:\n{proc.stderr[-2000:]}"
    return proc.stdout


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ══════════════════════════════════════════════════════════════════════════
# Setup section: requirements, .env.example, setup_env.sh, docker-compose
# ══════════════════════════════════════════════════════════════════════════

def test_pyproject_declares_supported_python_range():
    """readme: 'Python 3.10-3.13'."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10,<3.14"' in text


def test_uv_lockfile_is_in_sync_with_pyproject():
    """readme step 4: `uv sync` must be able to install without re-resolving."""
    if shutil.which("uv") is None:
        pytest.skip("uv not installed")
    proc = subprocess.run(
        ["uv", "lock", "--check"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, f"uv.lock is stale:\n{proc.stderr[-2000:]}"


def test_env_example_documents_every_section_the_readme_names():
    """readme: 'Field descriptions live in the comments of .env.example
    (LLM, Judge, Agent filter, FalkorDB sections).'"""
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for key in (
        "LLM_API",
        "MODEL_NAME",
        "JUDGE_LLM_API",
        "JUDGE_MODEL_NAME",
        "GREP_AGENT_LLM_API",
        "GREP_AGENT_MODEL_NAME",
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
        "GRAPH_NAME",
        "FALKORDB_PASSWORD",
    ):
        assert re.search(rf"^#?{key}=", text, re.M), f"{key} missing from .env.example"


def test_env_vars_the_graph_layer_reads_are_the_ones_the_readme_names():
    """readme: 'graph_from_env(...) requires NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD'."""
    source = (REPO_ROOT / "KG" / "graph" / "falkordb.py").read_text(encoding="utf-8")
    for key in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GRAPH_NAME"):
        assert key in source


def test_setup_env_runs_the_four_steps_in_the_documented_order():
    """readme step 4: 'uv sync -> docker compose up -d -> download_model.py -> verify'."""
    text = (REPO_ROOT / "setup_env.sh").read_text(encoding="utf-8")
    order = [text.index(m) for m in ("uv sync", "docker compose up -d", "download_model.py")]
    assert order == sorted(order), "setup_env.sh steps are out of documented order"
    assert "config.json" in text, "setup_env.sh does not verify the model files"
    assert "ping" in text, "setup_env.sh does not verify FalkorDB reachability"


def test_download_model_targets_the_two_documented_models():
    """readme: embedding + reranker models; reranker is Qwen3-Reranker-0.6B."""
    text = (REPO_ROOT / "download_model.py").read_text(encoding="utf-8")
    assert "Qwen/Qwen3-Embedding-0.6B" in text
    assert "Qwen/Qwen3-Reranker-0.6B" in text
    assert "embedding_models" in text and "reranker" in text


def test_docker_compose_serves_falkordb_on_the_documented_port_and_password():
    """readme 'FalkorDB (Docker)': port 6379, password falkordb, admin UI on 3000."""
    text = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "falkordb/falkordb" in text
    assert "--requirepass falkordb" in text
    assert '"6379:6379"' in text
    assert '"3000:3000"' in text


def test_no_hardcoded_endpoint_ips_anywhere():
    """readme: 'no IPs are hardcoded anywhere'."""
    proc = subprocess.run(
        ["git", "ls-files", "*.py", "*.sh", "*.yml"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    pattern = re.compile(r"https?://(\d{1,3}\.){3}\d{1,3}")
    offenders: list[str] = []
    for rel in proc.stdout.split():
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            m = pattern.search(line)
            if m and not m.group(0).endswith(("127.0.0.1", "0.0.0.0")):
                offenders.append(f"{rel}:{lineno}: {m.group(0)}")
    assert not offenders, "hardcoded endpoint IPs found:\n" + "\n".join(offenders)


# ══════════════════════════════════════════════════════════════════════════
# Module layout: every path in the readme tree must exist
# ══════════════════════════════════════════════════════════════════════════

README_LAYOUT_PATHS = [
    "KG/pipeline/factory.py",
    "KG/pipeline/retriever.py",
    "KG/pipeline/ingestor.py",
    "KG/pipeline/retrieval_steps/search.py",
    "KG/pipeline/retrieval_steps/filtering.py",
    "KG/pipeline/retrieval_steps/temporal.py",
    "KG/pipeline/retrieval_steps/evidence.py",
    "KG/graph/falkordb.py",
    "KG/llm/client.py",
    "KG/llm/prompts",
    "KG/services/entity_manager.py",
    "KG/services/relationship_manager.py",
    "KG/services/provenance.py",
    "KG/storage/chroma_manager.py",
    "KG/storage/chroma_vdb.py",
    "KG/storage/bm25.py",
    "KG/storage/cache.py",
    "KG/utils/common.py",
    "KG/utils/reranker.py",
    "KG/utils/query_time_parser.py",
    "KG/utils/temporal",
    "KG/utils/logger_config.py",
    "KG/llm/prompts/keyword/extraction.py",
    "KG/llm/prompts/extraction/two_step.py",
    "KG/llm/prompts/entity_ops/rules.py",
    "KG/llm/prompts/entity_ops/examples.py",
    "experiment/readme.md",
    "experiment/experiment_config.py",
    "experiment/longmem/watchdog.py",
    "experiment/locomo/pipeline.py",
    "experiment/longmem/agent_filter/replay_run.py",
    "experiment/longmem/agent_filter/harness.py",
    "experiment/locomo/grep_replay.py",
    "agent_filter/README.md",
    "docker-compose.yml",
    "setup_env.sh",
    "download_model.py",
    ".env.example",
]


@pytest.mark.parametrize("rel", README_LAYOUT_PATHS)
def test_every_path_named_in_the_readme_exists(rel):
    assert (REPO_ROOT / rel).exists(), f"readme references {rel}, which does not exist"


IMPORTABLE_MODULES = [
    "KG.pipeline.factory",
    "KG.pipeline.retriever",
    "KG.pipeline.ingestor",
    "KG.pipeline.retrieval_steps.search",
    "KG.pipeline.retrieval_steps.filtering",
    "KG.pipeline.retrieval_steps.temporal",
    "KG.pipeline.retrieval_steps.evidence",
    "KG.graph.falkordb",
    "KG.llm.client",
    "KG.services.entity_manager",
    "KG.services.relationship_manager",
    "KG.services.provenance",
    "KG.storage.chroma_manager",
    "KG.storage.chroma_vdb",
    "KG.storage.bm25",
    "KG.storage.cache",
    "KG.utils.common",
    "KG.utils.reranker",
    "KG.utils.query_time_parser",
    "KG.utils.logger_config",
]


@pytest.mark.parametrize("module", IMPORTABLE_MODULES)
def test_documented_modules_import_cleanly(module):
    importlib.import_module(module)


# ══════════════════════════════════════════════════════════════════════════
# "Quick API at a glance": every documented symbol must exist
# ══════════════════════════════════════════════════════════════════════════

DOCUMENTED_METHODS = [
    # (file, class, methods)
    ("KG/pipeline/retriever.py", "Retriever",
     ["generate_query_keywords", "assemble_context_from_query", "build_kg_context"]),
    ("KG/pipeline/retrieval_steps/evidence.py", "EvidenceBuilder",
     ["build_evidence_block"]),
    ("KG/pipeline/ingestor.py", "Ingestor",
     ["summarize_turn", "extract_entities_only", "extract_relationships_only",
      "apply_extraction_and_sync", "ingest_turn", "summarize_and_ingest_turn"]),
    ("KG/services/entity_manager.py", "EntityManager",
     ["normalize_entities", "find_similar_for_hybrid", "apply_ops"]),
    ("KG/services/relationship_manager.py", "RelationshipManager",
     ["upsert_from_extraction"]),
    ("KG/storage/chroma_manager.py", "VDBManager",
     ["get_entities_vdb", "get_relationships_vdb", "get_summaries_vdb",
      "get_entities_bm25", "persist_async", "reset_all"]),
    ("KG/storage/chroma_vdb.py", "SimpleChromaVDB",
     ["add", "upsert", "search", "delete", "update", "save", "load"]),
    ("KG/storage/bm25.py", "EntitiesBM25", ["add", "get_scores"]),
    ("KG/storage/cache.py", "CacheStore", ["load", "save", "clear", "reset"]),
    ("KG/llm/client.py", "LLMClient",
     ["chat", "stream_chat", "generate_llm_extract", "generate_llm_keyword",
      "generate_entity_ops"]),
    ("KG/services/provenance.py", "Provenance", ["prov_to_events", "merge_prov"]),
    ("KG/utils/reranker.py", "LLMPointwiseReranker", ["rank_pairs"]),
]


@pytest.mark.parametrize("rel,cls,methods", DOCUMENTED_METHODS,
                         ids=[f"{c}" for _, c, _ in DOCUMENTED_METHODS])
def test_documented_class_methods_exist(rel, cls, methods):
    have = _method_names(REPO_ROOT / rel, cls)
    missing = [m for m in methods if m not in have]
    assert not missing, f"{cls} is missing documented methods: {missing}"


DOCUMENTED_FUNCTIONS = [
    ("KG/graph/falkordb.py", ["graph_from_env"]),
    ("KG/storage/cache.py", ["build_id_to_meta_maps"]),
    ("KG/utils/reranker.py", ["get_reranker"]),
    ("KG/utils/query_time_parser.py", ["parse_query_time", "detect_and_parse_time_expressions"]),
    ("KG/utils/logger_config.py", ["setup_logger", "make_module_jlog", "_StepTimer"]),
    ("KG/utils/common.py",
     ["EntityType", "Entity", "Relationship", "ExtractionResult",
      "KeywordExtractionResult", "canonical_entity_id", "canonical_rel_id",
      "tokenize_en", "pickle_dump", "pickle_load"]),
    ("KG/pipeline/factory.py", ["build_pipeline"]),
    ("KG/llm/prompts/keyword/extraction.py", ["keyword_extraction_PROMPT"]),
    ("KG/llm/prompts/entity_ops/rules.py", ["ENTITY_OPS_RULES_V2"]),
    ("KG/llm/prompts/entity_ops/examples.py", ["ENTITY_OPS_FEW_SHOT"]),
]


def test_entity_ops_prompt_package_exports_what_entity_manager_imports():
    """readme 'LLM prompts': the entity-ops rules/examples used by EntityManager."""
    from KG.llm.prompts.entity_ops import ENTITY_OPS_RULES_V2, ENTITY_OPS_FEW_SHOT

    assert ENTITY_OPS_RULES_V2.strip() and ENTITY_OPS_FEW_SHOT.strip()
    consumer = (REPO_ROOT / "KG/services/entity_manager.py").read_text(encoding="utf-8")
    assert "ENTITY_OPS_RULES_V2" in consumer and "ENTITY_OPS_FEW_SHOT" in consumer


@pytest.mark.parametrize("rel,names", DOCUMENTED_FUNCTIONS, ids=[r for r, _ in DOCUMENTED_FUNCTIONS])
def test_documented_module_level_symbols_exist(rel, names):
    have = _top_level_names(REPO_ROOT / rel)
    missing = [n for n in names if n not in have]
    assert not missing, f"{rel} is missing documented symbols: {missing}"


def test_two_step_extraction_prompts_exist():
    """readme 'LLM prompts': entity_extraction_only / relationship_extraction_only."""
    text = (REPO_ROOT / "KG/llm/prompts/extraction/two_step.py").read_text(encoding="utf-8")
    assert "entity_extraction_only" in text
    assert "relationship_extraction_only" in text


def test_factory_returns_the_four_documented_keys():
    """readme: 'return {"retriever": ..., "ingestor": ..., "graph": ..., "mgr": ...}'."""
    text = (REPO_ROOT / "KG/pipeline/factory.py").read_text(encoding="utf-8")
    for key in ("retriever", "ingestor", "graph", "mgr"):
        assert f'"{key}"' in text


@pytest.mark.parametrize("field,expected", [
    ("summary_embed_dim", 1024),
    ("similar_entity_top_k", 3),
    ("entity_sim_threshold", 0.7),
])
def test_ingestor_config_defaults_match_the_readme(field, expected):
    """readme 'Config parameters (IngestorConfig)' quotes these defaults."""
    from KG.pipeline.ingestor import IngestorConfig
    assert getattr(IngestorConfig(), field) == expected


def test_retriever_config_exposes_every_documented_knob():
    """readme 'Config parameters (RetrieverConfig)'."""
    from KG.pipeline.retriever import RetrieverConfig
    cfg = RetrieverConfig()
    for field in (
        "ent_topk", "ent_threshold", "rel_topk", "rel_threshold",
        "filter_ent_topk", "filter_ent_threshold",
        "filter_rel_topk", "filter_rel_threshold",
        "use_reranker", "reranker_threshold", "reranker_topk",
        "summary_topk_per_item", "summary_vec_threshold",
        "use_full_summary", "fallback_to_raw",
    ):
        assert hasattr(cfg, field), f"RetrieverConfig has no {field}"


def test_default_evidence_path_matches_the_experiment_pipeline():
    """The default config must take the same evidence path the benchmarks use.

    Regression: with use_split_embeddings=False the evidence builder reached a
    legacy branch that called get_summary_text_by_id(summary_id, full=True) —
    a TypeError swallowed by build_kg_context, which then silently returned
    "(no KG context)".
    """
    from KG.pipeline.retriever import RetrieverConfig
    from experiment.experiment_config import RERANKER_PARAMS

    cfg = RetrieverConfig()
    assert cfg.use_split_embeddings is True
    assert cfg.split_single_entry_raw is True
    assert RERANKER_PARAMS["use_split_embeddings"] is True


def test_summary_text_lookup_is_never_called_with_a_full_kwarg():
    """SummariesVDB.get_summary_text_by_id takes only summary_id."""
    from KG.storage.chroma_vdb import SummariesVDB
    import inspect

    params = inspect.signature(SummariesVDB.get_summary_text_by_id).parameters
    assert list(params) == ["self", "summary_id"]

    source = (REPO_ROOT / "KG/pipeline/retrieval_steps/evidence.py").read_text(encoding="utf-8")
    bad = re.findall(r"get_summary_text_by_id\([^)]*full\s*=", source)
    assert not bad, f"caller passes an unsupported kwarg: {bad}"


# ══════════════════════════════════════════════════════════════════════════
# Experiment entry points: every flag the docs advertise must parse
# ══════════════════════════════════════════════════════════════════════════

LOCOMO_FLAGS = [
    "--dataset", "--sessions-jsonl", "--dataset-json", "--out-root", "--run-tag",
    "--sample-ids", "--artifact-dir", "--adv", "--stage", "--retrieval-mode",
    "--replay-run-dir", "--baseline-run-dir", "--no-judge", "--adaptive", "--tau",
    "--prev-k", "--entity-sim-topk", "--entity-sim-threshold",
]

LONGMEM_FLAGS = [
    "--run-tag", "--type", "--data-folder", "--file-pattern", "--child",
    "--child-file", "--dataset-id", "--num", "--stage", "--no-judge",
    "--artifact-dir", "--force", "--output-root",
]


@pytest.mark.slow
def test_locomo_pipeline_accepts_every_documented_flag():
    help_text = _cli_help("experiment/locomo/pipeline.py")
    missing = [f for f in LOCOMO_FLAGS if f not in help_text]
    assert not missing, f"locomo/pipeline.py is missing documented flags: {missing}"


@pytest.mark.slow
def test_longmem_watchdog_accepts_every_documented_flag():
    help_text = _cli_help("experiment/longmem/watchdog.py")
    missing = [f for f in LONGMEM_FLAGS if f not in help_text]
    assert not missing, f"longmem/watchdog.py is missing documented flags: {missing}"


@pytest.mark.slow
def test_locomo_aggregate_accepts_the_flags_the_experiment_readme_shows():
    help_text = _cli_help("experiment/locomo/aggregate.py")
    for flag in ("--dataset", "--root"):
        assert flag in help_text, f"aggregate.py is missing {flag}"


def test_experiment_config_is_the_single_source_of_truth():
    """experiment/readme.md: 'edit only experiment_config.py'."""
    from experiment import experiment_config

    for name in ("REPRODUCIBILITY_PARAMS", "INGEST_PARAMS", "RETRIEVAL_PARAMS", "RERANKER_PARAMS"):
        assert isinstance(getattr(experiment_config, name), dict)
    for key in ("ent_topk", "rel_topk", "filter_ent_topk", "filter_rel_topk",
                "summary_topk_per_item", "summary_vec_threshold"):
        assert key in experiment_config.RETRIEVAL_PARAMS
    for key in ("use_reranker", "reranker_threshold", "reranker_topk"):
        assert key in experiment_config.RERANKER_PARAMS


# ══════════════════════════════════════════════════════════════════════════
# Integration: the two runnable examples in the readme
# ══════════════════════════════════════════════════════════════════════════

def _integration_blockers() -> list[str]:
    missing: list[str] = []
    if not (REPO_ROOT / ".env").exists():
        missing.append("no .env (cp .env.example .env)")
    if not (REPO_ROOT / "models/embedding_models/qwen3-0.6b/config.json").exists():
        missing.append("embedding model not downloaded")
    if not (REPO_ROOT / "models/reranker/qwen3-reranker-0.6b/config.json").exists():
        missing.append("reranker model not downloaded")

    from urllib.parse import urlparse
    try:
        from dotenv import dotenv_values
    except ImportError:
        return missing + ["python-dotenv unavailable"]

    env = dotenv_values(REPO_ROOT / ".env") if (REPO_ROOT / ".env").exists() else {}

    uri = (env.get("NEO4J_URI") or "").strip()
    if not uri:
        missing.append("NEO4J_URI unset")
    else:
        parsed = urlparse(uri)
        if not _port_open(parsed.hostname or "localhost", parsed.port or 6379):
            missing.append(f"FalkorDB unreachable at {parsed.hostname}:{parsed.port}")

    llm = (env.get("LLM_API") or "").strip().strip('"')
    if not llm:
        missing.append("LLM_API unset")
    else:
        parsed = urlparse(llm)
        if not _port_open(parsed.hostname or "localhost", parsed.port or 80):
            missing.append(f"LLM endpoint unreachable at {parsed.hostname}:{parsed.port}")
    return missing


@pytest.fixture(scope="module")
def live_pipeline(tmp_path_factory):
    """Isolated pipeline: its own artifacts dir and its own FalkorDB graph."""
    blockers = _integration_blockers()
    if blockers:
        pytest.skip("integration prerequisites missing: " + "; ".join(blockers))

    art = tmp_path_factory.mktemp("kg_artifacts")
    os.environ["KG_ARTIFACTS_DIR"] = str(art)
    os.environ["GRAPH_NAME"] = "readme_claims_test"

    from KG.pipeline.factory import build_pipeline

    pipeline = build_pipeline()
    pipeline["graph"].clear_all()
    yield pipeline
    try:
        pipeline["graph"].clear_all()
        pipeline["graph"].close()
    except Exception:
        pass


@pytest.mark.integration
def test_ingest_example_from_the_readme_runs_end_to_end(live_pipeline):
    """readme 'KG ingest (Ingestor)' example, verbatim."""
    ingestor = live_pipeline["ingestor"]

    results = ingestor.summarize_and_ingest_turn(
        session_id=1,
        message_id=42,
        user_text="I went to an AI workshop yesterday.",
        assistant_text="That's great! What did you learn?",
        prev_k=2,
        dialogue_datetime="2023/02/18 (Sat) 08:08",
    )

    assert results["summary_id"], "no summary was written"
    assert results.get("results") is not None

    live_pipeline["mgr"].persist_async()

    entities = live_pipeline["mgr"].cache["entities"]
    assert entities, "ingest produced no entities"
    names = {str(meta.get("name", "")).lower() for meta in entities.values()}
    assert any("workshop" in n for n in names), f"expected a workshop entity, got {names}"


@pytest.mark.integration
def test_ingest_resolves_relative_dates_against_dialogue_datetime(live_pipeline):
    """readme: 'the raw dialogue is first rewritten by
    detect_and_parse_time_expressions, then fed into extraction'."""
    entities = live_pipeline["mgr"].cache["entities"]
    dates = {
        str(meta.get("name"))
        for meta in entities.values()
        if str(meta.get("type", "")).lower() == "date"
    }
    assert "2023-02-17" in dates, (
        f"'yesterday' relative to 2023/02/18 should resolve to 2023-02-17; got {dates}"
    )


@pytest.mark.integration
def test_ingest_syncs_entities_into_falkordb(live_pipeline):
    """readme: 'Sync to FalkorDB: graph.sync_entities() / sync_relationships()'."""
    ids = [meta["id"] for meta in live_pipeline["mgr"].cache["entities"].values()]
    assert ids
    nodes, rels = live_pipeline["graph"].get_node_subgraph(ids)
    assert nodes, "entities were never written to FalkorDB"


@pytest.mark.integration
def test_retrieve_example_from_the_readme_returns_real_context(live_pipeline):
    """readme 'KG context retrieval (Retriever)' example, verbatim.

    Regression: this returned the literal string "(no KG context)" because a
    TypeError inside the evidence builder was swallowed by build_kg_context.
    """
    retriever = live_pipeline["retriever"]

    kg_context = retriever.build_kg_context(
        question="What workshop did the user attend?",
        ent_topk=5,
        filter_ent_topk=3,
        query_time="2023/02/18 (Sat) 08:08",
    )

    trace = getattr(retriever, "last_retrieval_trace", {}) or {}
    assert trace.get("stop_reason") != "build_kg_context_failed", (
        f"build_kg_context raised internally: {trace.get('exception')}"
    )
    assert kg_context != "(no KG context)"
    assert "AI workshop" in kg_context
    assert "=== Entities ===" in kg_context


@pytest.mark.integration
def test_keyword_extraction_returns_both_keyword_levels(live_pipeline):
    """readme: 'the LLM produces high-level (global) and low-level (local) keywords'."""
    kw = live_pipeline["retriever"].generate_query_keywords("What workshop did the user attend?")
    assert kw.low_level_keywords, "no low-level keywords"
    assert isinstance(kw.high_level_keywords, list)


@pytest.mark.integration
def test_hybrid_search_loads_the_persisted_bm25_index(live_pipeline):
    """readme: 'Entities: vector similarity (VDB) + BM25 (dual name/desc index)'."""
    mgr = live_pipeline["mgr"]
    bm25 = mgr.get_entities_bm25(load_if_empty=True)
    assert len(bm25.metas) > 0, "BM25 index did not load the ingested entities"
