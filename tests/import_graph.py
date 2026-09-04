"""Inspect project-local Python imports and report dependency cycles.

Static analysis only -- modules are parsed with `ast`, never imported. That is
the point: importing this project's modules constructs Chroma clients and loads
model weights, and a cycle check must not need a working environment to run.

Only project-local imports are graphed. Third-party edges are dropped by
`_known_module`, which keeps the output about this codebase's own layering.

Lives in `tests/` rather than `tools/` because its only callers are
`test_package_boundaries.py` and `test_architecture.py`; `tools/` is for
things a user of the repository runs.

Usage:
    python -m tests.import_graph
    python -m tests.import_graph --check   # exit 1 on a newly introduced cycle
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = ("grace_mem", "experiment")


def module_name(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    """Convert a file path to its dotted module name.

    An `__init__.py` yields its package name rather than "pkg.__init__", so
    that a package and its `__init__` are one node in the graph -- otherwise
    every relative import inside a package would look like an edge to a
    separate module.
    """
    relative = path.relative_to(project_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def discover_modules(
    roots: Iterable[str] = DEFAULT_ROOTS,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Path]:
    """Map every module under `roots` to its file path.

    Returns:
        module name -> path. Sorted traversal, so the graph and its reported
        cycles come out in a stable order run to run.
    """
    modules: dict[str, Path] = {}
    for root in roots:
        root_path = project_root / root
        for path in sorted(root_path.rglob("*.py")):
            modules[module_name(path, project_root)] = path
    return modules


def _resolve_from_import(
    current: str,
    node: ast.ImportFrom,
    *,
    is_package: bool,
) -> str | None:
    """Resolve a `from . import x` to an absolute module name.

    `node.level` is the number of leading dots. The base differs by file kind:
    inside a package's `__init__.py` one dot means the package itself, while in
    an ordinary module it means the package containing it -- hence the
    `is_package` branch. Getting this wrong shifts every relative import by one
    level and invents edges that do not exist.

    Returns None when the dots climb above the project root.
    """
    if node.level == 0:
        return node.module

    package = current.split(".") if is_package else current.split(".")[:-1]
    keep = len(package) - node.level + 1
    if keep < 0:
        return None
    prefix = package[:keep]
    if node.module:
        prefix.extend(node.module.split("."))
    return ".".join(prefix)


def _known_module(name: str, modules: set[str]) -> str | None:
    """Find the longest known module that is a prefix of `name`.

    `from grace_mem.adapters.cache.cache import CacheStore` parses as a target of
    "grace_mem.adapters.cache.cache.CacheStore", which is a class, not a module.
    Trimming the tail until something known appears resolves it to the module.
    The same walk drops third-party imports, which never match at any depth.
    """
    candidate = name
    while candidate:
        if candidate in modules:
            return candidate
        candidate = candidate.rpartition(".")[0]
    return None


def build_graph(modules: dict[str, Path]) -> dict[str, set[str]]:
    """Build the module dependency graph by parsing every file's imports.

    `ast.walk` visits the whole tree, so imports nested inside functions count
    as edges too. That is deliberate -- this codebase defers heavy imports into
    functions precisely to break cycles, and a graph that ignored them would
    report the layering as cleaner than it is.

    Self-edges are dropped; a module importing from its own package is not a
    dependency worth reporting.
    """
    known = set(modules)
    graph = {name: set() for name in modules}
    for current, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        is_package = path.name == "__init__.py"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names = [alias.name for alias in node.names]
                for imported in imported_names:
                    target = _known_module(imported, known)
                    if target and target != current:
                        graph[current].add(target)
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_from_import(current, node, is_package=is_package)
                if not base:
                    continue
                imported_modules = {
                    f"{base}.{alias.name}"
                    for alias in node.names
                    if f"{base}.{alias.name}" in known
                }
                targets = imported_modules or {_known_module(base, known)}
                graph[current].update(
                    target for target in targets if target and target != current
                )
    return graph


def strongly_connected_components(graph: dict[str, set[str]]) -> list[list[str]]:
    """Find import cycles via Tarjan's strongly-connected-components algorithm.

    A cycle in an import graph is exactly a strongly connected component of
    more than one node: every module in it can reach every other, so no import
    order satisfies them all. Tarjan finds all of them in one depth-first pass.

    Single-node components are filtered out -- every module is trivially
    connected to itself and that is not a cycle.

    Recursive, so a pathologically deep import chain could exhaust the stack.
    At this codebase's depth that is not close.

    Returns:
        Cycles, each sorted, the list sorted -- stable enough to diff in CI.
    """
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[list[str]] = []

    def visit(node: str) -> None:
        """Depth-first visit assigning Tarjan's index and lowlink to `node`.

        `lowlink` is the smallest index reachable from this node's subtree. A
        node whose lowlink equals its own index is the root of a component, and
        everything above it on the stack belongs to that component.
        """
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for target in graph[node]:
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])

        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1:
                components.append(sorted(component))

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return sorted(components)


def layer_edges(graph: dict[str, set[str]]) -> dict[tuple[str, str], int]:
    """Count imports crossing top-level package boundaries.

    Collapses the module graph to its first path component, which answers the
    architectural question: how much does `experiment` reach into `grace_mem`,
    and does anything go back the other way. Within-package edges are excluded
    as internal detail.
    """
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for source, targets in graph.items():
        source_layer = source.split(".")[0]
        for target in targets:
            target_layer = target.split(".")[0]
            if source_layer != target_layer:
                counts[(source_layer, target_layer)] += 1
    return dict(sorted(counts.items()))


def main(argv: list[str] | None = None) -> int:
    """Report module counts, cross-package edges, and cycles.

    Returns:
        1 if --check was passed and cycles exist, else 0. Without --check the
        exit code stays 0 so the report can be read without failing a shell.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", default=list(DEFAULT_ROOTS))
    parser.add_argument("--check", action="store_true", help="Exit non-zero when cycles exist")
    args = parser.parse_args(argv)

    modules = discover_modules(args.roots)
    graph = build_graph(modules)
    cycles = strongly_connected_components(graph)

    print(f"modules: {len(modules)}")
    print(f"internal edges: {sum(len(targets) for targets in graph.values())}")
    print("cross-root edges:")
    for (source, target), count in layer_edges(graph).items():
        print(f"  {source} -> {target}: {count}")
    print("cycles:")
    if cycles:
        for component in cycles:
            print("  " + " -> ".join(component))
    else:
        print("  none")

    return 1 if args.check and cycles else 0


if __name__ == "__main__":
    raise SystemExit(main())
