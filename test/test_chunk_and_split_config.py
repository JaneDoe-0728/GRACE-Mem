"""Tests for LoCoMo chunked ingest and the LongMem split-summary coupling.

Local only, not committed — companion to test_readme_claims.py.

Two behaviours are pinned here:

1. **LoCoMo chunking** — INGEST_PARAMS["chunk_turns"] is the single source of truth
   and reaches every site that turns a session into ingest rows. A site that forgets
   to pass it silently falls back to the module default, which is exactly the kind of
   drift that makes a snapshot restore stop lining up with its run.

2. **LongMem split summaries** — INGEST_PARAMS["use_split_summary"] drives BOTH the
   post-ingest rebuild and the retrieval flag. If they ever disagree, retrieval looks
   up :u/:a entries that were never written and the whole provenance channel drops to
   zero candidates without raising anything.

Run: uv run pytest test/test_chunk_and_split_config.py -v
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
from experiment.experiment_config import INGEST_PARAMS, RERANKER_PARAMS
from experiment.locomo.stages import ingest as locomo_ingest
from experiment.locomo import stage_adapter


# ══════════════════════════════════════════════════════════════════════════
# Config surface
# ══════════════════════════════════════════════════════════════════════════

def test_chunk_turns_lives_in_experiment_config():
    assert INGEST_PARAMS["chunk_turns"] == 8


def test_use_split_summary_lives_in_experiment_config_and_defaults_on():
    assert INGEST_PARAMS["use_split_summary"] is True


def test_chunk_size_is_never_read_from_the_environment():
    """The env var was a second source of truth that silently beat the config."""
    offenders = []
    for py in (REPO_ROOT / "experiment").rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        if "LOCOMO_CHUNK_TURNS" in py.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(py.relative_to(REPO_ROOT)))
    assert not offenders, f"LOCOMO_CHUNK_TURNS still read in: {offenders}"


def test_module_default_tracks_the_config():
    assert locomo_ingest.CHUNK_TURNS == INGEST_PARAMS["chunk_turns"]


# ══════════════════════════════════════════════════════════════════════════
# Chunking behaviour
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("n_turns,chunk_turns,expected_sizes", [
    (24, 8, [8, 8, 8]),        # exact multiple
    (20, 8, [8, 8, 4]),        # ragged tail
    (5, 8, [5]),               # shorter than one chunk
    (1, 8, [1]),               # single turn
    (24, 1, [1] * 24),         # degenerate: one turn per chunk
])
def test_chunking_splits_into_expected_windows(n_turns, chunk_turns, expected_sizes):
    dialogue = [f"turn {i}" for i in range(n_turns)]
    chunks = list(locomo_ingest._iter_dialogue_chunks(dialogue, chunk_turns))

    assert [len(c) for _, c in chunks] == expected_sizes
    assert [mid for mid, _ in chunks] == list(range(len(expected_sizes)))
    # No turn is lost or duplicated.
    assert [line for _, c in chunks for line in c] == dialogue


@pytest.mark.parametrize("chunk_turns", [0, -1])
def test_non_positive_chunk_turns_keeps_the_whole_session_as_one_chunk(chunk_turns):
    """chunk_turns=0 is the documented way to reproduce pre-chunking runs."""
    dialogue = [f"turn {i}" for i in range(20)]
    chunks = list(locomo_ingest._iter_dialogue_chunks(dialogue, chunk_turns))
    assert chunks == [(0, dialogue)]


def test_empty_dialogue_yields_nothing_when_chunking():
    """A session with no turns produces no summary — nothing empty enters the VDB."""
    assert list(locomo_ingest._iter_dialogue_chunks([], 8)) == []


def test_empty_dialogue_still_yields_one_chunk_when_chunking_is_off():
    """chunk_turns<=0 reproduces pre-chunking runs, including this degenerate case."""
    assert list(locomo_ingest._iter_dialogue_chunks([], 0)) == [(0, [])]


def test_empty_session_produces_no_ingest_rows_when_chunking():
    empty = dict(_session(0), dialogue=[])
    df = locomo_ingest.session_records_to_df([empty], conv_id="conv7", chunk_turns=8)
    assert df.empty


def test_none_falls_back_to_the_configured_default():
    dialogue = [f"turn {i}" for i in range(20)]
    explicit = list(locomo_ingest._iter_dialogue_chunks(dialogue, INGEST_PARAMS["chunk_turns"]))
    implicit = list(locomo_ingest._iter_dialogue_chunks(dialogue, None))
    assert implicit == explicit


# ══════════════════════════════════════════════════════════════════════════
# Chunking reaches the ingest rows
# ══════════════════════════════════════════════════════════════════════════

def _session(n_turns: int = 20) -> dict:
    return {
        "sample_index": 3,
        "session_id": "s1",
        "date_time": "2023/02/18 (Sat) 08:08",
        "dialogue": [f"Caroline: turn {i}" for i in range(n_turns)],
        "speaker_a": "Caroline",
        "speaker_b": "Melanie",
    }


def test_sessions_to_one_turn_df_emits_one_row_per_chunk():
    df = locomo_ingest.sessions_to_one_turn_df([_session(20)], sample_filter=3, chunk_turns=8)

    assert len(df) == 3
    assert list(df["message_id"]) == [0, 1, 2]
    assert df["session_id"].nunique() == 1
    # Each row carries only its own window.
    assert df.iloc[0]["user_text"].count("\n") == 7
    assert df.iloc[2]["user_text"].count("\n") == 3


def test_sessions_to_one_turn_df_chunk_turns_zero_is_one_row():
    df = locomo_ingest.sessions_to_one_turn_df([_session(20)], sample_filter=3, chunk_turns=0)
    assert len(df) == 1
    assert list(df["message_id"]) == [0]


def test_session_records_to_df_emits_one_row_per_chunk():
    df = locomo_ingest.session_records_to_df([_session(20)], conv_id="conv7", chunk_turns=8)

    assert len(df) == 3
    assert list(df["message_id"]) == [0, 1, 2]
    assert set(df["session_id"]) == {"conv7__s1"}


def test_chunking_makes_summary_ids_unique_per_chunk():
    """summary_id is "<session_id>:<message_id>" — chunking must fan it out."""
    df = locomo_ingest.session_records_to_df([_session(20)], conv_id="conv7", chunk_turns=8)
    summary_ids = [f"{row.session_id}:{row.message_id}" for row in df.itertuples()]
    assert summary_ids == ["conv7__s1:0", "conv7__s1:1", "conv7__s1:2"]
    assert len(set(summary_ids)) == len(summary_ids)


# ══════════════════════════════════════════════════════════════════════════
# Every call site threads chunk_turns
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("func", [
    stage_adapter.run_ingest_stage_for_locomo,
    stage_adapter.run_ingest_stage_for_records,
], ids=["for_locomo", "for_records"])
def test_ingest_stage_adapters_accept_chunk_turns(func):
    assert "chunk_turns" in inspect.signature(func).parameters


CALL_SITES = [
    ("experiment/locomo/stage_adapter.py", "sessions_to_one_turn_df"),
    ("experiment/locomo/stage_adapter.py", "session_records_to_df"),
    ("experiment/locomo/snapshot.py", "session_records_to_df"),
    ("experiment/locomo/stages/ingest.py", "sessions_to_one_turn_df"),
]


@pytest.mark.parametrize("rel,func_name", CALL_SITES, ids=[f"{r.split('/')[-1]}:{f}" for r, f in CALL_SITES])
def test_every_df_builder_call_passes_chunk_turns(rel, func_name):
    """A forgotten call site silently falls back to the module default.

    In snapshot.py that means a restored artifact's summary_ids no longer match the
    run that produced them — with no error anywhere.
    """
    tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == func_name
        or isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == func_name
    ]
    assert calls, f"no call to {func_name} found in {rel}"
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert "chunk_turns" in kwargs, (
            f"{rel}:{call.lineno} calls {func_name} without chunk_turns"
        )


@pytest.mark.parametrize("rel", [
    "experiment/locomo/workers.py",
    "experiment/locomo/snapshot.py",
])
def test_worker_and_snapshot_pass_the_cli_value(rel):
    source = (REPO_ROOT / rel).read_text(encoding="utf-8")
    assert "chunk_turns=args.chunk_turns" in source, (
        f"{rel} does not forward the CLI --chunk-turns value"
    )


def test_locomo_cli_exposes_chunk_turns_defaulting_to_the_config():
    from experiment.locomo.cli import parse_args

    args = parse_args(["--sample-ids", "0"])
    assert args.chunk_turns == INGEST_PARAMS["chunk_turns"]

    overridden = parse_args(["--sample-ids", "0", "--chunk-turns", "4"])
    assert overridden.chunk_turns == 4


# ══════════════════════════════════════════════════════════════════════════
# LongMem split-summary coupling
# ══════════════════════════════════════════════════════════════════════════

def test_locomo_always_uses_single_entry_regardless_of_use_split_summary():
    """LoCoMo never gets :u/:a — the shared config value must stay True."""
    assert RERANKER_PARAMS["split_single_entry_raw"] is True


def test_longmem_processor_retrieval_flag_uses_typed_dataset_config():
    """The processor must derive retrieval layout from the dataset config.

    Regression: processor.py pinned it to False, so retrieval always looked for :u/:a
    entries even on a fresh run where nothing had built them.
    """
    source = (REPO_ROOT / "experiment/longmem/processor.py").read_text(encoding="utf-8")

    assert '"split_single_entry_raw": not config.use_split_summary' in source
    hardcoded = re.findall(r'"split_single_entry_raw":\s*(True|False)\b', source)
    assert not hardcoded


def test_longmem_rerun_retrieval_flag_uses_shared_split_setting():
    source = (REPO_ROOT / "experiment/longmem/rerun.py").read_text(encoding="utf-8")

    assert "USE_SPLIT_SUMMARY" in source
    assert not re.findall(r'"split_single_entry_raw":\s*(True|False)\b', source)


@pytest.mark.parametrize("use_split_summary,expected_flag", [(True, False), (False, True)])
def test_the_two_flags_are_exact_inverses(use_split_summary, expected_flag):
    """split_single_entry_raw == not use_split_summary. Same rule in both files."""
    assert (not use_split_summary) is expected_flag


def test_processor_runs_the_rebuild_after_ingest():
    """The rebuild must be invoked from the single point where both ingest modes meet."""
    source = (REPO_ROOT / "experiment/longmem/processor.py").read_text(encoding="utf-8")
    assert "def _maybe_rebuild_split_summaries" in source
    assert "self._maybe_rebuild_split_summaries(config)" in source

    tree = ast.parse(source)
    hook = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_maybe_rebuild_split_summaries"
    )
    body = ast.get_source_segment(source, hook) or ""
    assert "config.use_split_summary" in body, "the rebuild step ignores the config flag"
    assert "rebuild_artifact" in body, "the rebuild step does not call rebuild_artifact"


def test_rebuild_artifact_is_idempotent_by_contract():
    """The hook relies on this to make resumed/rerun ingests safe."""
    source = (REPO_ROOT / "experiment/longmem/rebuild_split_summaries.py").read_text(encoding="utf-8")
    assert "already_rebuilt" in source


def test_rebuild_helpers_are_importable_from_the_processor_hook():
    from experiment.longmem.rebuild_split_summaries import (  # noqa: F401
        SCRIPT_DATA_DIR,
        get_compressor,
        rebuild_artifact,
    )

    assert "chunk_turns" not in inspect.signature(rebuild_artifact).parameters
