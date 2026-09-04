"""The seams where one part of the pipeline invokes another.

A subprocess call and a reset helper have this in common: when they break, they
break quietly. The aggregate step logs a warning and returns None; a reset that
raises half-way leaves the files it meant to delete. Neither shows up as a failed
run, so both are pinned here.
"""

from __future__ import annotations

import logging
from pathlib import Path

from experiment.locomo.analysis.aggregate import _aggregate_locomo_run, parse_args


def test_the_pipeline_sends_the_aggregate_cli_only_arguments_it_accepts(
    tmp_path: Path, monkeypatch
) -> None:
    """`--dataset locomo` outlived the locomo-plus branch that defined it.

    argparse exits 2 on the unknown argument before doing any work, so every
    judged run ended with a warning and no _correctness_aggregate.json -- while
    the same script's documented manual invocation kept working, which is why it
    went unnoticed. This feeds the argv the pipeline actually builds back into the
    parser it is aimed at.
    """
    import subprocess

    sent: list[list[str]] = []

    def capture(cmd, **kwargs):
        sent.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", capture)
    _aggregate_locomo_run(tmp_path, include_adversarial=True)

    assert sent, "the aggregate step no longer shells out"
    parsed = parse_args(sent[0][2:])  # drop the interpreter and the script path

    assert parsed.root == str(tmp_path)
    assert parsed.exclude_adversarial is False


def test_aggregating_a_judged_run_writes_both_outputs(tmp_path: Path) -> None:
    """End to end through the real subprocess, which is where the break was."""
    sample = tmp_path / "sample_1"
    sample.mkdir()
    (sample / "qa_judge.csv").write_text(
        "question,gold,generated,category,correctness\n"
        "what color,blue,blue,1,1\n"
        "what size,big,small,2,0\n",
        encoding="utf-8",
    )

    result = _aggregate_locomo_run(tmp_path, include_adversarial=False)

    assert result is not None, "the aggregate subprocess failed"
    assert result.output_json.exists()
    assert result.merged_csv is not None and result.merged_csv.exists()


def test_reset_all_finishes_even_when_a_client_refuses_to_close(
    tmp_path: Path, caplog
) -> None:
    """close() re-raises the first vector-store failure, and reset_all delegates
    to it. Letting that escape skipped the rmtree and the cache clear below --
    refresh_system left every stale artifact on disk, and the LoCoMo sample hook
    ran the next sample against the previous sample's entity cache."""
    from grace_mem.services.vector_store.chroma_manager import VDBManager

    manager = VDBManager(tmp_path)
    stale = manager.ENT_CHROMA_DIR
    stale.mkdir(parents=True, exist_ok=True)
    (stale / "chroma.sqlite3").write_text("stale", encoding="utf-8")
    manager.cache["entities"] = {"e1": {"name": "left over"}}

    def explode(**kwargs):
        raise RuntimeError("chroma client already shut down")

    manager.close = explode  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING):
        manager.reset_all(delete_files=True)

    assert not stale.exists(), "the reset stopped at the failing close()"
    assert not manager.cache.get("entities"), "the in-memory cache survived the reset"
    assert any("close()" in record.message for record in caplog.records)


def test_recorded_entrypoints_are_module_paths_that_still_import() -> None:
    """run_metadata.json records how a run was launched, so someone can rerun it.

    Two of the three named modules this PR deleted (`longmem.run_batch`,
    `longmem.watchdog`); the third was rewritten to `locomo.pipeline.runner`,
    which is not importable either -- the package root is `experiment`. A
    provenance field nobody can act on is worse than no field, and nothing reads
    it programmatically, so the value that earns its place is the one that runs.
    """
    import ast
    import importlib.util

    root = Path(__file__).resolve().parent.parent
    recorded: dict[str, str] = {}
    for path in sorted(root.glob("experiment/**/*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values, strict=False):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "entrypoint"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    recorded[f"{path.relative_to(root)}:{value.lineno}"] = value.value

    assert recorded, "no entrypoint strings found; the metadata field moved"
    for where, module in sorted(recorded.items()):
        assert importlib.util.find_spec(module) is not None, (
            f"{where} records an entrypoint that cannot be imported: {module}"
        )


def test_no_cli_hardcodes_an_ingestion_default() -> None:
    """INGEST_PARAMS is the single source of truth for the ingest knobs.

    A second copy in an entry point is invisible while the orchestrated path passes
    its values explicitly, and wrong the moment someone runs the module directly:
    locomo/stages/ingest.py carried entity_sim_topk=4 and entity_sim_threshold=0.5
    against the configured 3 and 0.6, so `python -m ...stages.ingest` resolved
    entities at a different similarity than every other path. 39462f0 deleted three
    such copies and missed this one, so this checks the shape rather than the site.

    A default may name a module constant (CHUNK_TURNS), as long as that constant is
    itself derived from INGEST_PARAMS.
    """
    import ast

    flags = {"--prev-k", "--entity-sim-topk", "--entity-sim-threshold", "--chunk-turns"}
    root = Path(__file__).resolve().parent.parent
    offenders: list[str] = []

    def mentions_config(node: ast.AST) -> bool:
        return any(
            isinstance(sub, ast.Name) and sub.id.lstrip("_") == "INGEST_PARAMS"
            for sub in ast.walk(node)
        )

    for path in sorted(root.glob("experiment/**/*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Module-level constants that are themselves read from the config.
        derived = {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign) and mentions_config(node.value)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in flags
            ):
                continue
            default = next((kw.value for kw in node.keywords if kw.arg == "default"), None)
            if default is None:
                continue
            # A string default is a mode sentinel, not a value: gold_recall's
            # "auto" infers the chunk size from the run's own artifacts, which is
            # stronger than reading the config, since it describes the run being
            # analysed rather than the one that would be produced today.
            if isinstance(default, ast.Constant) and isinstance(default.value, str):
                continue
            ok = mentions_config(default) or (
                isinstance(default, ast.Name) and default.id in derived
            )
            if not ok:
                offenders.append(
                    f"{path.relative_to(root)}:{node.lineno} {node.args[0].value}"
                    f" = {ast.unparse(default)}"
                )

    assert not offenders, "ingest defaults not read from INGEST_PARAMS: " + "; ".join(offenders)
