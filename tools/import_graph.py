"""Inspect project-local Python imports and report dependency cycles."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = ("KG", "experiment")


def module_name(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    relative = path.relative_to(project_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def discover_modules(
    roots: Iterable[str] = DEFAULT_ROOTS,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Path]:
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
    candidate = name
    while candidate:
        if candidate in modules:
            return candidate
        candidate = candidate.rpartition(".")[0]
    return None


def build_graph(modules: dict[str, Path]) -> dict[str, set[str]]:
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
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[list[str]] = []

    def visit(node: str) -> None:
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
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for source, targets in graph.items():
        source_layer = source.split(".")[0]
        for target in targets:
            target_layer = target.split(".")[0]
            if source_layer != target_layer:
                counts[(source_layer, target_layer)] += 1
    return dict(sorted(counts.items()))


def main(argv: list[str] | None = None) -> int:
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
