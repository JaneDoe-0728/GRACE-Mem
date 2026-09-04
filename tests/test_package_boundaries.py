"""Import-direction rules the feature-first restructure must preserve.

These are the invariants the package layout depends on. They are
asserted here rather than only in `tests/test_architecture.py` because that file
is part of the gitignored local suite, and these rules have to survive into the
repository -- a move that inverts a dependency is exactly the kind of mistake a
reviewer reading a large `git mv` diff will not see.

Every rule is written against the *current* module paths and holds today. Each
one is the same rule the target tree needs, so the move commits update the paths
here and nothing else:

    grace_mem.ingestion.steps    ->  grace_mem/ingestion/
    grace_mem.retrieval.steps ->  grace_mem/retrieval/
"""

from pathlib import Path

from tests.import_graph import build_graph, discover_modules

ROOT = Path(__file__).resolve().parents[1]

# The data-model layer. Intra-layer imports are allowed; anything else is not.
DATA_MODEL_MODULES = (
    "grace_mem.data_model",
    "grace_mem.data_model.entities",
    "grace_mem.data_model.extraction",
    "grace_mem.data_model.relationships",
)
# The two capabilities. These are whole-package prefixes on purpose: scoped to
# `.steps` the rule missed `retrieval.steps.search -> ingestion.parsing`, which
# existed for as long as the rule did.
INGESTION_PREFIX = "grace_mem.ingestion"
RETRIEVAL_PREFIX = "grace_mem.retrieval"
# Services wrap an external technology and must not own a decision about
# entities, evidence or turns -- an import into a capability means one moved in.
SERVICES_PREFIX = "grace_mem.services"
CAPABILITY_PREFIXES = (INGESTION_PREFIX, RETRIEVAL_PREFIX)


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


def test_data_model_imports_nothing_from_grace_mem():
    """The data_model layer must stay constructible without any infrastructure.

    A model that reaches for a vector store, an LLM client, or a cache is a model
    that cannot be tested or reasoned about on its own.
    """
    graph = _graph()

    reached = {
        (module, target)
        for module in DATA_MODEL_MODULES
        for target in graph.get(module, set())
        if target.startswith("grace_mem.") and not target.startswith("grace_mem.data_model")
    }

    assert reached == set()


def test_ingestion_and_retrieval_do_not_import_each_other():
    """The two capabilities share through data models and services, never directly.

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


def test_services_do_not_import_a_capability():
    """A service wraps a technology; it does not decide anything about the domain.

    LLMClient used to build an EntityOpsProcessor, which put "is this extracted
    entity the same node as that existing one?" -- the central ingestion
    judgement -- inside the HTTP client that happens to make the call for it.
    """
    graph = _graph()

    reached = {
        edge
        for edge in _edges_from(graph, SERVICES_PREFIX)
        for prefix in CAPABILITY_PREFIXES
        if edge[1] == prefix or edge[1].startswith(f"{prefix}.")
    }

    assert reached == set()
