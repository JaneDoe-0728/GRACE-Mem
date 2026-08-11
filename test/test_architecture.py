import ast
from pathlib import Path
import subprocess
import sys

from tools.import_graph import build_graph, discover_modules, strongly_connected_components


ROOT = Path(__file__).resolve().parents[1]


def test_cycle_detector_reports_strongly_connected_modules():
    graph = {
        "cli": {"experiment"},
        "experiment": {"retriever"},
        "retriever": {"experiment"},
    }

    assert strongly_connected_components(graph) == [["experiment", "retriever"]]


def test_internal_import_graph_has_no_cycles():
    graph = build_graph(discover_modules(project_root=ROOT))

    assert strongly_connected_components(graph) == []


def test_package_relative_imports_resolve_inside_the_package():
    graph = build_graph(discover_modules(project_root=ROOT))

    assert "KG.llm.prompts.config" in graph["KG.llm.prompts"]
    assert "KG.llm" not in graph["KG.llm.prompts"]
    assert "KG.utils.temporal.patterns" in graph["KG.utils.temporal.classifier"]
    assert "KG.utils.temporal" not in graph["KG.utils.temporal.classifier"]


def test_core_to_experiment_dependencies_are_explicitly_bounded():
    graph = build_graph(discover_modules(project_root=ROOT))
    reverse_dependencies = {
        (source, target)
        for source, targets in graph.items()
        for target in targets
        if source.startswith("KG.") and target.startswith("experiment.")
    }

    assert reverse_dependencies == set()


def test_experiment_imports_use_canonical_package_names():
    invalid_imports: list[str] = []
    for path in (ROOT / "experiment").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name == "locomo" or name.startswith("locomo.") or name == "experiment_config":
                    invalid_imports.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")

    assert invalid_imports == []


def test_package_modules_do_not_mutate_sys_path_at_import_time():
    invalid_calls: list[str] = []
    for package_root in (ROOT / "KG", ROOT / "experiment"):
        for path in package_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                    continue
                function = node.value.func
                if not isinstance(function, ast.Attribute) or function.attr not in {"append", "insert"}:
                    continue
                owner = function.value
                if (
                    isinstance(owner, ast.Attribute)
                    and isinstance(owner.value, ast.Name)
                    and owner.value.id == "sys"
                    and owner.attr == "path"
                ):
                    invalid_calls.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert invalid_calls == []


def test_script_modules_preserve_sys_path_when_imported_as_packages():
    script = """
import importlib
import sys

modules = (
    'experiment.longmem.analysis.collect',
    'experiment.longmem.analysis.summarize',
    'experiment.locomo.vote_merge',
    'experiment.oracle',
    'experiment.score',
)
for module_name in modules:
    before = list(sys.path)
    importlib.import_module(module_name)
    assert sys.path == before, module_name
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_snapshot_imports_in_a_fresh_interpreter():
    result = subprocess.run(
        [sys.executable, "-c", "import experiment.locomo.snapshot"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_locomo_internal_modules_do_not_import_helpers_facade():
    invalid_imports: list[str] = []
    locomo_root = ROOT / "experiment" / "locomo"
    facade = locomo_root / "helpers" / "__init__.py"
    for path in locomo_root.rglob("*.py"):
        if path == facade:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "experiment.locomo.helpers"
            ):
                invalid_imports.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert invalid_imports == []


def test_helpers_facade_does_not_eagerly_import_optional_layers():
    script = """
import sys
import experiment.locomo.helpers as helpers

blocked = {
    'experiment.locomo.helpers.llm',
    'experiment.locomo.aggregate',
    'experiment.locomo.summary',
    'experiment.locomo.utils.graph',
}
assert blocked.isdisjoint(sys.modules)
assert helpers.normalize_dataset_name('locomo') == 'locomo'
assert 'experiment.locomo.helpers.dataset' in sys.modules
assert blocked.isdisjoint(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
