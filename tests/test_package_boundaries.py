"""Import-direction rules the feature-first restructure must preserve.

These are the invariants `docs/package-structure.md` depends on. They are
asserted here rather than only in `tests/test_architecture.py` because that file
is part of the gitignored local suite, and these rules have to survive into the
repository -- a move that inverts a dependency is exactly the kind of mistake a
reviewer reading a large `git mv` diff will not see.

Every rule is written against the *current* module paths and holds today. Each
one is the same rule the target tree needs, so the move commits update the paths
here and nothing else:

    grace_mem.pipeline.ingest_steps    ->  grace_mem/ingestion/
    grace_mem.retrieval.steps ->  grace_mem/retrieval/
"""

from pathlib import Path

from tools.import_graph import build_graph, discover_modules

ROOT = Path(__file__).resolve().parents[1]

# The domain layer. Intra-domain imports are allowed; anything else is not.
DOMAIN_MODULES = (
    "grace_mem.domain",
    "grace_mem.domain.entities",
    "grace_mem.domain.extraction",
    "grace_mem.domain.relationships",
)
# The two capabilities, at their current paths.
INGESTION_PREFIX = "grace_mem.pipeline.ingest_steps"
RETRIEVAL_PREFIX = "grace_mem.retrieval.steps"


def _graph() -> dict[str, set[str]]:
    return build_graph(discover_modules(project_root=ROOT))


def _edges_from(graph: dict[str, set[str]], prefix: str) -> set[tuple[str, str]]:
    return {
        (source, target)
        for source, targets in graph.items()
        if source == prefix or source.startswith(f"{prefix}.")
        for target in targets
    }


def test_core_does_not_import_the_benchmark_harness():
    """grace_mem/ is the memory system; experiment/ is the harness that drives it.

    An edge in this direction means harness code has been left in the core, which
    is what splitting error_analysis was for.
    """
    graph = _graph()

    inverted = {
        (source, target)
        for source, targets in graph.items()
        for target in targets
        if source.startswith("grace_mem.") and target.startswith("experiment.")
    }

    assert inverted == set()


def test_domain_models_import_nothing_from_grace_mem():
    """The domain layer must stay constructible without any infrastructure.

    A model that reaches for a vector store, an LLM client, or a cache is a model
    that cannot be tested or reasoned about on its own.
    """
    graph = _graph()

    reached = {
        (module, target)
        for module in DOMAIN_MODULES
        for target in graph.get(module, set())
        if target.startswith("grace_mem.") and not target.startswith("grace_mem.domain")
    }

    assert reached == set()


def test_ingestion_and_retrieval_do_not_import_each_other():
    """The two capabilities share through domain models and adapters, never directly.

    A direct edge would make either impossible to move, test, or reason about
    without dragging in the other.
    """
    graph = _graph()

    crossing = {
        edge
        for edge in _edges_from(graph, INGESTION_PREFIX)
        if edge[1] == RETRIEVAL_PREFIX or edge[1].startswith(f"{RETRIEVAL_PREFIX}.")
    } | {
        edge
        for edge in _edges_from(graph, RETRIEVAL_PREFIX)
        if edge[1] == INGESTION_PREFIX or edge[1].startswith(f"{INGESTION_PREFIX}.")
    }

    assert crossing == set()
