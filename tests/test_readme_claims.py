"""README conformance tests.

Encodes executable contracts from README.md / experiment/README.md so a future
edit that drifts from the code fails here instead of in a user's first five
minutes.

Layout:
  * Offline tests            — no network, no models, no DB. Always run.
  * Integration tests        — need FalkorDB + an OpenAI-compatible endpoint +
                               the downloaded models. Auto-skip when absent.

Run everything:      uv run pytest tests/test_readme_claims.py -v
Offline only:        uv run pytest tests/test_readme_claims.py -v -m "not integration"
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
from urllib.parse import unquote

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DOCUMENTS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "EVALUATION.md",
    REPO_ROOT / "experiment" / "README.md",
    REPO_ROOT / "experiment" / "agent_filter" / "README.md",
    REPO_ROOT / "tests" / "README.md",
)


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


def _module_help(module: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"{module} --help failed:\n{proc.stderr[-2000:]}"
    return proc.stdout


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _heading_anchors(markdown: str) -> set[str]:
    anchors: set[str] = set()
    for heading in re.findall(r"^#{1,6}\s+(.+)$", markdown, re.MULTILINE):
        slug = re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")
        anchors.add(slug)
    return anchors


# ══════════════════════════════════════════════════════════════════════════
# Setup section: requirements, .env.example, tools/setup_env.sh, docker-compose
# ══════════════════════════════════════════════════════════════════════════

def test_pyproject_points_to_the_tracked_root_readme():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^readme\s*=\s*"([^"]+)"', text, re.MULTILINE)

    assert match is not None
    assert match.group(1) == "README.md"
    assert (REPO_ROOT / match.group(1)).is_file()


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_local_documentation_links_and_anchors_resolve(document: Path):
    markdown = document.read_text(encoding="utf-8")
    for raw_target in re.findall(r"(?<!!)\[[^]]+\]\(([^)]+)\)", markdown):
        target = unquote(raw_target.strip())
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path_part, _, anchor = target.partition("#")
        linked_path = (document.parent / path_part).resolve() if path_part else document.resolve()
        assert linked_path.exists(), f"{document}: missing local link {raw_target}"
        if anchor and linked_path.is_file():
            linked_markdown = linked_path.read_text(encoding="utf-8")
            assert anchor in _heading_anchors(linked_markdown), (
                f"{document}: missing anchor {raw_target}"
            )

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
    source = (REPO_ROOT / "grace_mem" / "adapters" / "graph" / "falkordb.py").read_text(encoding="utf-8")
    for key in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GRAPH_NAME"):
        assert key in source


def test_setup_env_runs_the_four_steps_in_the_documented_order():
    """readme step 4: 'uv sync -> docker compose up -d -> tools/download_models.py -> verify'."""
    text = (REPO_ROOT / "tools/setup_env.sh").read_text(encoding="utf-8")
    order = [text.index(m) for m in ("uv sync", "docker compose up -d", "tools/download_models.py")]
    assert order == sorted(order), "tools/setup_env.sh steps are out of documented order"
    assert "config.json" in text, "tools/setup_env.sh does not verify the model files"
    assert "ping" in text, "tools/setup_env.sh does not verify FalkorDB reachability"


def test_download_model_targets_the_two_documented_models():
    """readme: embedding + reranker models; reranker is Qwen3-Reranker-0.6B."""
    text = (REPO_ROOT / "tools/download_models.py").read_text(encoding="utf-8")
    assert "Qwen/Qwen3-Embedding-0.6B" in text
    assert "Qwen/Qwen3-Reranker-0.6B" in text
    assert "embedding_models" in text and "reranker" in text
    assert "revision=None" not in text
    assert len(re.findall(r'"[0-9a-f]{40}"', text)) == 2


def test_local_test_suite_stays_out_of_the_public_repository():
    automated = subprocess.run(
        ["git", "check-ignore", "-q", "tests/test_architecture.py"],
        cwd=REPO_ROOT,
        timeout=30,
    )
    manual = subprocess.run(
        ["git", "check-ignore", "-q", "tests/test_api.py"],
        cwd=REPO_ROOT,
        timeout=30,
    )

    assert automated.returncode == 0, "automated regression tests should remain local-only"
    assert manual.returncode == 0, "live/manual probes should remain local-only"


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
# Documented package and operations paths
# ══════════════════════════════════════════════════════════════════════════

DOCUMENTED_PATHS = [
    "grace_mem/bootstrap.py",
    "grace_mem/retrieval/pipeline.py",
    "grace_mem/ingestion/pipeline.py",
    "grace_mem/retrieval/steps/search.py",
    "grace_mem/retrieval/steps/filtering.py",
    "grace_mem/retrieval/steps/temporal_relevance.py",
    "grace_mem/retrieval/evidence.py",
    "grace_mem/adapters/graph/falkordb.py",
    "grace_mem/adapters/llm/client.py",
    "grace_mem/ingestion/prompts",
    "grace_mem/ingestion/managers/entity_manager.py",
    "grace_mem/ingestion/managers/relationship_manager.py",
    "grace_mem/domain/provenance.py",
    "grace_mem/adapters/vector_store/chroma_manager.py",
    "grace_mem/adapters/vector_store/chroma_vdb.py",
    "grace_mem/adapters/sparse_index/bm25.py",
    "grace_mem/adapters/cache/cache.py",
    "grace_mem/domain/entities.py",
    "grace_mem/domain/relationships.py",
    "grace_mem/domain/extraction.py",
    "grace_mem/ingestion/parsing.py",
    "grace_mem/text.py",
    "grace_mem/retrieval/reranker.py",
    "grace_mem/temporal/query_time_parser.py",
    "grace_mem/temporal",
    "grace_mem/runtime/logger_config.py",
    "grace_mem/retrieval/prompts/keyword/extraction.py",
    "grace_mem/ingestion/prompts/extraction/two_step.py",
    "grace_mem/ingestion/prompts/entity_ops/rules.py",
    "grace_mem/ingestion/prompts/entity_ops/examples.py",
    "experiment/README.md",
    "experiment/experiment_config.py",
    "experiment/common/evaluation/judge.py",
    "experiment/common/evaluation/oracle.py",
    "experiment/common/evaluation/score.py",
    "experiment/longmem/pipeline/watchdog.py",
    "experiment/locomo/pipeline/runner.py",
    "experiment/agent_filter/replay/longmem.py",
    "experiment/agent_filter/harness.py",
    "experiment/agent_filter/replay/locomo.py",
    "experiment/agent_filter/README.md",
    "docker-compose.yml",
    "tools/setup_env.sh",
    "tools/download_models.py",
    "tools/download_datasets.py",
    "experiment/longmem/tools/convert_dataset.py",
    ".env.example",
]


@pytest.mark.parametrize("rel", DOCUMENTED_PATHS)
def test_documented_public_paths_exist(rel):
    assert (REPO_ROOT / rel).exists(), f"documentation contract is missing {rel}"


IMPORTABLE_MODULES = [
    "grace_mem.bootstrap",
    "grace_mem.retrieval.pipeline",
    "grace_mem.ingestion.pipeline",
    "grace_mem.retrieval.steps.search",
    "grace_mem.retrieval.steps.filtering",
    "grace_mem.retrieval.steps.temporal_relevance",
    "grace_mem.retrieval.evidence",
    "grace_mem.adapters.graph.falkordb",
    "grace_mem.adapters.llm.client",
    "grace_mem.ingestion.managers.entity_manager",
    "grace_mem.ingestion.managers.relationship_manager",
    "grace_mem.domain.provenance",
    "grace_mem.adapters.vector_store.chroma_manager",
    "grace_mem.adapters.vector_store.chroma_vdb",
    "grace_mem.adapters.sparse_index.bm25",
    "grace_mem.adapters.cache.cache",
    "grace_mem.domain",
    "grace_mem.ingestion.parsing",
    "grace_mem.text",
    "grace_mem.retrieval.reranker",
    "grace_mem.temporal.query_time_parser",
    "grace_mem.runtime.logger_config",
]


@pytest.mark.parametrize("module", IMPORTABLE_MODULES)
def test_documented_modules_import_cleanly(module):
    importlib.import_module(module)


# ══════════════════════════════════════════════════════════════════════════
# Public package contracts referenced by architecture and operations docs
# ══════════════════════════════════════════════════════════════════════════

DOCUMENTED_METHODS = [
    # (file, class, methods)
    ("grace_mem/retrieval/pipeline.py", "Retriever",
     ["assemble_context_from_query", "build_kg_context"]),
    ("grace_mem/retrieval/evidence.py", "EvidenceBuilder",
     ["build_evidence_block"]),
    ("grace_mem/ingestion/pipeline.py", "Ingestor",
     ["summarize_turn", "extract_entities_only", "extract_relationships_only",
      "apply_extraction_and_sync", "ingest_turn", "summarize_and_ingest_turn"]),
    ("grace_mem/ingestion/managers/entity_manager.py", "EntityManager",
     ["normalize_entities", "find_similar_for_hybrid", "apply_ops"]),
    ("grace_mem/ingestion/managers/relationship_manager.py", "RelationshipManager",
     ["upsert_from_extraction"]),
    ("grace_mem/adapters/vector_store/chroma_manager.py", "VDBManager",
     ["get_entities_vdb", "get_relationships_vdb", "get_summaries_vdb",
      "get_entities_bm25", "persist_async", "reset_all"]),
    ("grace_mem/adapters/vector_store/chroma_vdb.py", "SimpleChromaVDB",
     ["add", "upsert", "search", "delete", "update", "save", "load"]),
    ("grace_mem/adapters/sparse_index/bm25.py", "EntitiesBM25", ["add", "get_scores"]),
    ("grace_mem/adapters/cache/cache.py", "CacheStore", ["load", "save", "clear"]),
    ("grace_mem/adapters/llm/client.py", "LLMClient",
     ["chat", "generate_llm_extract", "generate_llm_keyword"]),
    ("grace_mem/domain/provenance.py", "Provenance", ["prov_to_events", "merge_prov"]),
    ("grace_mem/retrieval/reranker.py", "LLMPointwiseReranker", ["rank_pairs"]),
]


@pytest.mark.parametrize("rel,cls,methods", DOCUMENTED_METHODS,
                         ids=[f"{c}" for _, c, _ in DOCUMENTED_METHODS])
def test_documented_class_methods_exist(rel, cls, methods):
    have = _method_names(REPO_ROOT / rel, cls)
    missing = [m for m in methods if m not in have]
    assert not missing, f"{cls} is missing documented methods: {missing}"


DOCUMENTED_FUNCTIONS = [
    ("grace_mem/adapters/graph/falkordb.py", ["graph_from_env"]),
    ("grace_mem/adapters/cache/cache.py", ["build_id_to_meta_maps"]),
    ("grace_mem/retrieval/reranker.py", ["get_reranker"]),
    ("grace_mem/temporal/query_time_parser.py", ["parse_query_time", "detect_and_parse_time_expressions"]),
    ("grace_mem/runtime/logger_config.py", ["setup_logger", "make_module_jlog", "_StepTimer"]),
    ("grace_mem/domain/entities.py", ["EntityType", "Entity", "canonical_entity_id"]),
    ("grace_mem/domain/relationships.py", ["Relationship", "canonical_rel_id"]),
    ("grace_mem/domain/extraction.py", ["ExtractionResult", "KeywordExtractionResult"]),
    ("grace_mem/text.py", ["tokenize_en"]),
    ("grace_mem/bootstrap.py", ["build_pipeline"]),
    ("grace_mem/retrieval/keywords.py", ["generate_query_keywords"]),
    ("grace_mem/retrieval/prompts/keyword/extraction.py", ["KEYWORD_EXTRACTION_PROMPT"]),
    ("grace_mem/ingestion/prompts/entity_ops/rules.py", ["ENTITY_OPS_RULES_V2"]),
    ("grace_mem/ingestion/prompts/entity_ops/examples.py", ["ENTITY_OPS_FEW_SHOT"]),
]


def test_entity_ops_prompt_package_exports_what_entity_manager_imports():
    """readme 'LLM prompts': the entity-ops rules/examples used by EntityManager."""
    from grace_mem.ingestion.prompts.entity_ops import ENTITY_OPS_FEW_SHOT, ENTITY_OPS_RULES_V2

    assert ENTITY_OPS_RULES_V2.strip() and ENTITY_OPS_FEW_SHOT.strip()
    consumer = (REPO_ROOT / "grace_mem/ingestion/managers/entity_manager.py").read_text(encoding="utf-8")
    assert "ENTITY_OPS_RULES_V2" in consumer and "ENTITY_OPS_FEW_SHOT" in consumer


@pytest.mark.parametrize("rel,names", DOCUMENTED_FUNCTIONS, ids=[r for r, _ in DOCUMENTED_FUNCTIONS])
def test_documented_module_level_symbols_exist(rel, names):
    have = _top_level_names(REPO_ROOT / rel)
    missing = [n for n in names if n not in have]
    assert not missing, f"{rel} is missing documented symbols: {missing}"


def test_two_step_extraction_prompts_exist():
    """readme 'LLM prompts': entity_extraction_only / relationship_extraction_only."""
    text = (REPO_ROOT / "grace_mem/ingestion/prompts/extraction/two_step.py").read_text(encoding="utf-8")
    assert "entity_extraction_only" in text
    assert "relationship_extraction_only" in text


def test_factory_returns_the_four_documented_keys():
    """readme: 'return {"retriever": ..., "ingestor": ..., "graph": ..., "mgr": ...}'."""
    text = (REPO_ROOT / "grace_mem/bootstrap.py").read_text(encoding="utf-8")
    for key in ("retriever", "ingestor", "graph", "mgr"):
        assert f'"{key}"' in text


def test_default_stage_model_matches_the_experiment_guide():
    from experiment.locomo.cli import DEFAULT_STAGES as LOCOMO_DEFAULT_STAGES
    from experiment.longmem.helpers.args import DEFAULT_STAGES as LONGMEM_DEFAULT_STAGES

    expected = ("ingest", "qa_eval", "judge")
    assert LOCOMO_DEFAULT_STAGES == expected
    assert LONGMEM_DEFAULT_STAGES == expected


@pytest.mark.parametrize("field,expected", [
    ("summary_embed_dim", 1024),
    ("similar_entity_top_k", 3),
    ("entity_sim_threshold", 0.7),
])
def test_ingestor_config_defaults_match_the_readme(field, expected):
    """readme 'Config parameters (IngestorConfig)' quotes these defaults."""
    from grace_mem.ingestion.pipeline import IngestorConfig
    assert getattr(IngestorConfig(), field) == expected


def test_retriever_config_exposes_every_documented_knob():
    """readme 'Config parameters (RetrieverConfig)'."""
    from grace_mem.retrieval.config import RetrieverConfig
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
    from experiment.experiment_config import RERANKER_PARAMS
    from grace_mem.retrieval.config import RetrieverConfig

    cfg = RetrieverConfig()
    assert cfg.use_split_embeddings is True
    assert cfg.split_single_entry_raw is True
    assert RERANKER_PARAMS["use_split_embeddings"] is True


def test_summary_text_lookup_is_never_called_with_a_full_kwarg():
    """SummariesVDB.get_summary_text_by_id takes only summary_id."""
    import inspect

    from grace_mem.adapters.vector_store.chroma_vdb import SummariesVDB

    params = inspect.signature(SummariesVDB.get_summary_text_by_id).parameters
    assert list(params) == ["self", "summary_id"]

    source = (REPO_ROOT / "grace_mem/retrieval/evidence.py").read_text(encoding="utf-8")
    bad = re.findall(r"get_summary_text_by_id\([^)]*full\s*=", source)
    assert not bad, f"caller passes an unsupported kwarg: {bad}"


# ══════════════════════════════════════════════════════════════════════════
# Experiment entry points: every flag the docs advertise must parse
# ══════════════════════════════════════════════════════════════════════════

LOCOMO_FLAGS = [
    "--sessions-jsonl", "--dataset-json", "--out-root", "--run-tag",
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
    help_text = _module_help("experiment.locomo.pipeline.runner")
    missing = [f for f in LOCOMO_FLAGS if f not in help_text]
    assert not missing, f"locomo/pipeline/runner.py is missing documented flags: {missing}"


@pytest.mark.slow
def test_longmem_watchdog_accepts_every_documented_flag():
    help_text = _module_help("experiment.longmem.pipeline.watchdog")
    missing = [f for f in LONGMEM_FLAGS if f not in help_text]
    assert not missing, f"longmem/pipeline/watchdog.py is missing documented flags: {missing}"


@pytest.mark.slow
def test_locomo_aggregate_accepts_the_flags_the_experiment_readme_shows():
    help_text = _module_help("experiment.locomo.analysis.aggregate")
    for flag in ("--root",):
        assert flag in help_text, f"aggregate.py is missing {flag}"


def test_experiment_config_is_the_single_source_of_truth():
    """experiment/README.md: shared defaults live in experiment_config.py."""
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

    from grace_mem.bootstrap import build_pipeline

    pipeline = build_pipeline()
    pipeline["graph"].clear_all()
    yield pipeline
    try:
        pipeline["graph"].clear_all()
        pipeline["graph"].close()
    except Exception:
        pass


@pytest.fixture(scope="module")
def ingested_turn(live_pipeline):
    """Run the README's ingest example once, for the assertions that read it back.

    A fixture rather than the first test, because three tests inspect the state
    it leaves behind. Depending on test order instead meant running one of them
    alone failed on an empty cache -- which is what hid the real bug in
    test_ingest_syncs_entities_into_falkordb for as long as these were skipped.
    """
    results = live_pipeline["ingestor"].summarize_and_ingest_turn(
        session_id=1,
        message_id=42,
        user_text="I went to an AI workshop yesterday.",
        assistant_text="That's great! What did you learn?",
        prev_k=2,
        dialogue_datetime="2023/02/18 (Sat) 08:08",
    )
    live_pipeline["mgr"].flush_persist()
    return results


@pytest.mark.integration
def test_ingest_example_from_the_readme_runs_end_to_end(live_pipeline, ingested_turn):
    """readme 'KG ingest (Ingestor)' example, verbatim."""
    assert ingested_turn["summary_id"], "no summary was written"
    assert ingested_turn.get("results") is not None

    entities = live_pipeline["mgr"].cache["entities"]
    assert entities, "ingest produced no entities"
    names = {str(meta.get("name", "")).lower() for meta in entities.values()}
    assert any("workshop" in n for n in names), f"expected a workshop entity, got {names}"


@pytest.mark.integration
def test_ingest_resolves_relative_dates_against_dialogue_datetime(live_pipeline, ingested_turn):
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
def test_ingest_syncs_entities_into_falkordb(live_pipeline, ingested_turn):
    """readme: 'Sync to FalkorDB: graph.sync_entities() / sync_relationships()'."""
    ids = [meta["id"] for meta in live_pipeline["mgr"].cache["entities"].values()]
    assert ids

    # One mapping of entity id -> {"self", "neighbors"}, not a (nodes, rels)
    # pair: every production caller reads it as a dict, and both test doubles
    # return one.
    subgraph = live_pipeline["graph"].get_node_subgraph(ids)

    assert subgraph, "entities were never written to FalkorDB"
    assert set(subgraph) <= set(ids), "the graph returned ids that were not asked for"
    assert all("self" in node for node in subgraph.values())


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
def test_hybrid_search_loads_the_persisted_bm25_index(live_pipeline):
    """readme: 'Entities: vector similarity (VDB) + BM25 (dual name/desc index)'."""
    mgr = live_pipeline["mgr"]
    bm25 = mgr.get_entities_bm25(load_if_empty=True)
    assert len(bm25.metas) > 0, "BM25 index did not load the ingested entities"
