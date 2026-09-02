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
    from grace_mem.adapters.vector_store.chroma_manager import VDBManager

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
