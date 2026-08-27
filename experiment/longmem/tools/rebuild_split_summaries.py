"""
rebuild_split_summaries.py — Rebuild summaries_chroma/ with split user/assistant embeddings.

For each artifacts_*/ directory in a completed run, replaces the summaries_chroma/
collection with two VDB entries per turn:
  {session_id}:{message_id}:u  → embed(user_text_raw)
  {session_id}:{message_id}:a  → embed(llmlingua(assistant_text) + rewrite_temporal_text())

KG entities/relationships are NOT touched; summary_id prov links remain valid.

Usage:
    python -m experiment.longmem.tools.rebuild_split_summaries \\
        --run_dir experiment/longmem/output/oss-20b-0427 \\
        [--categories knowledge_update single_session_user ...] \\
        [--dry_run]
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from experiment.common.paths import REPO_ROOT
from grace_mem.utils.query_time_parser import parse_query_time
from grace_mem.utils.raw_context_lookup import RawContextLookup
from grace_mem.utils.temporal import build_time_context, rewrite_temporal_text

SCRIPT_DATA_DIR = REPO_ROOT / "experiment" / "longmem" / "script_data"


def get_compressor():
    from llmlingua import PromptCompressor
    return PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True,
    )


def compress_and_rewrite(text: str, compressor, dialogue_datetime: str | None) -> str:
    """llmlingua compress assistant text, then rewrite temporal expressions."""
    import torch
    # Step 1: llmlingua
    try:
        torch.cuda.empty_cache()
        results = compressor.compress_prompt_llmlingua2(
            text,
            rate=0.6,
            force_tokens=["\n", ".", "!", "?", ","],
            chunk_end_tokens=[".", "\n"],
            return_word_label=False,
            drop_consecutive=True,
        )
        compressed = results.get("compressed_prompt", "").strip() or text
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        compressed = text
    except Exception:
        compressed = text

    # Step 2: temporal rewriting (same as ingest pipeline)
    if dialogue_datetime:
        try:
            reference_dt = parse_query_time(dialogue_datetime)
            if reference_dt is not None:
                tctx = build_time_context(
                    reference_dt=reference_dt,
                    reference_time_str=dialogue_datetime,
                    source="rebuild_split",
                )
                rewritten, _ = rewrite_temporal_text(compressed, tctx)
                return rewritten
        except Exception:
            pass

    return compressed


def _read_turns(meta_path: Path) -> list[dict]:
    """Load one canonical metadata row per session/message pair."""
    turns_by_id: dict[tuple[str, int], dict] = {}
    with meta_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                turn = json.loads(line)
                session_id = turn.get("session_id")
                message_id = turn.get("message_id")
                if session_id is None or message_id is None:
                    continue
                key = (str(session_id), int(message_id))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            turns_by_id[key] = turn
    return list(turns_by_id.values())


def _validate_split_ids(ids: set[str], expected_ids: set[str] | None = None) -> int:
    """Validate that an index contains complete user/assistant entry pairs."""
    if not ids:
        raise RuntimeError("split-summary index is empty")

    roles_by_base: dict[str, set[str]] = {}
    for entry_id in ids:
        base_id, separator, role = str(entry_id).rpartition(":")
        if not separator or not base_id or role not in {"u", "a"}:
            raise RuntimeError(f"invalid split-summary entry id: {entry_id!r}")
        roles_by_base.setdefault(base_id, set()).add(role)

    incomplete = sorted(base for base, roles in roles_by_base.items() if roles != {"u", "a"})
    if incomplete:
        raise RuntimeError(
            "split-summary index has incomplete entry pairs: " + ", ".join(incomplete[:10])
        )
    if expected_ids is not None and ids != expected_ids:
        missing = sorted(expected_ids - ids)
        extra = sorted(ids - expected_ids)
        raise RuntimeError(
            "split-summary index validation failed: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    return len(roles_by_base)


def _read_split_ids(chroma_dir: Path) -> set[str]:
    from grace_mem.storage.chroma_vdb import SummariesVDB

    vdb = SummariesVDB(dim=1024, path=str(chroma_dir), collection_name="summaries")
    try:
        results = vdb._collection.get(include=["metadatas"])
        return {str(entry_id) for entry_id in (results.get("ids") or [])}
    finally:
        vdb.close()


def _expected_split_ids(turns: list[dict]) -> set[str]:
    """Compute the split ids an artifact should contain, to find what is missing."""
    expected: set[str] = set()
    for turn in turns:
        base_id = f"{turn['session_id']}:{int(turn['message_id'])}"
        expected.update({f"{base_id}:u", f"{base_id}:a"})
    return expected


def _replace_index_transactionally(
    *,
    chroma_dir: Path,
    temp_dir: Path,
    backup_dir: Path,
    rollback_dir: Path,
) -> None:
    """Install a validated temp index while retaining or restoring the old index."""
    if rollback_dir.exists():
        shutil.rmtree(rollback_dir)

    previous_dir: Path | None = None
    if chroma_dir.exists():
        previous_dir = backup_dir if not backup_dir.exists() else rollback_dir
        os.replace(chroma_dir, previous_dir)

    try:
        os.replace(temp_dir, chroma_dir)
    except BaseException:
        if previous_dir is not None and previous_dir.exists() and not chroma_dir.exists():
            os.replace(previous_dir, chroma_dir)
        raise
    else:
        if previous_dir == rollback_dir and rollback_dir.exists():
            shutil.rmtree(rollback_dir)


def rebuild_artifact(artifact_dir: Path, lookup: RawContextLookup, compressor, dry_run: bool) -> dict:
    """Regenerate the split-turn summaries for one artifact directory.

    Repairs artifacts written before turns were split into :u and :a entries, so
    an older run's stores can be used with current retrieval instead of being
    re-ingested from scratch.
    """
    meta_path = artifact_dir / "summaries_meta.jsonl"
    chroma_dir = artifact_dir / "summaries_chroma"
    backup_dir = artifact_dir / "summaries_chroma_bak"
    temp_dir = artifact_dir / "summaries_chroma.tmp"
    rollback_dir = artifact_dir / "summaries_chroma.rollback"

    if not meta_path.exists():
        return {"status": "skip", "reason": "no_meta"}

    # Recover the old index if a previous process stopped between the two renames.
    if not chroma_dir.exists() and rollback_dir.exists():
        os.replace(rollback_dir, chroma_dir)
    elif chroma_dir.exists() and rollback_dir.exists():
        shutil.rmtree(rollback_dir)
    if not chroma_dir.exists() and backup_dir.exists():
        shutil.copytree(backup_dir, chroma_dir)

    turns = _read_turns(meta_path)

    if not turns:
        return {"status": "skip", "reason": "empty_meta"}

    if dry_run:
        return {"status": "dry_run", "turns": len(turns)}

    expected_ids = _expected_split_ids(turns)

    if backup_dir.exists() and chroma_dir.exists():
        try:
            existing_ids = _read_split_ids(chroma_dir)
            existing_turns = _validate_split_ids(existing_ids, expected_ids)
        except Exception:
            pass
        else:
            return {
                "status": "skip",
                "reason": "already_rebuilt",
                "turns": existing_turns,
                "entries": len(existing_ids),
            }

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    from grace_mem.storage.chroma_vdb import SummariesVDB
    vdb = None
    n_missing = 0
    try:
        vdb = SummariesVDB(dim=1024, path=str(temp_dir), collection_name="summaries")
        for turn in turns:
            session_id = turn["session_id"]
            message_id = int(turn["message_id"])
            dialogue_datetime = turn.get("dialogue_datetime")
            existing_text = str(turn.get("summary_text") or turn.get("text") or "").strip()

            user_text = lookup.get_user_text(str(session_id), message_id) or existing_text
            assistant_text = lookup.get_assistant_text(str(session_id), message_id)
            assistant_summary = (
                compress_and_rewrite(assistant_text, compressor, dialogue_datetime)
                if assistant_text
                else existing_text
            )
            user_text = str(user_text or assistant_summary or "").strip()
            assistant_summary = str(assistant_summary or user_text or "").strip()
            if not user_text or not assistant_summary:
                n_missing += 1
                continue

            vdb.add_split_turns(
                session_id=session_id,
                message_id=message_id,
                user_text=user_text,
                assistant_summary=assistant_summary,
                dialogue_datetime=dialogue_datetime,
            )

        actual_ids = {
            str(entry_id)
            for entry_id in (vdb._collection.get(include=["metadatas"]).get("ids") or [])
        }
        added = _validate_split_ids(actual_ids, expected_ids)
        vdb.close()
        vdb = None

        _replace_index_transactionally(
            chroma_dir=chroma_dir,
            temp_dir=temp_dir,
            backup_dir=backup_dir,
            rollback_dir=rollback_dir,
        )
    except BaseException:
        if vdb is not None:
            vdb.close()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    return {
        "status": "ok",
        "turns": len(turns),
        "added": added,
        "entries": len(expected_ids),
        "missing": n_missing,
    }


def export_artifact(artifact_dir: Path) -> dict:
    """Export summaries_chroma/ entries to summaries_split_meta.jsonl for inspection."""
    chroma_dir = artifact_dir / "summaries_chroma"
    out_path = artifact_dir / "summaries_split_meta.jsonl"

    if not chroma_dir.exists():
        return {"status": "skip", "reason": "no_chroma"}

    from grace_mem.storage.chroma_vdb import SummariesVDB
    vdb = SummariesVDB(dim=1024, path=str(chroma_dir), collection_name="summaries")
    results = vdb._collection.get(include=["metadatas"])
    vdb.close()

    ids = results.get("ids") or []
    metas = results.get("metadatas") or []

    with open(out_path, "w", encoding="utf-8") as f:
        for mid, meta in zip(ids, metas):
            if meta is None:
                continue
            row = dict(meta)
            row["id"] = mid
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"status": "ok", "entries": len(ids), "path": str(out_path)}


def main():
    parser = argparse.ArgumentParser(description="Rebuild summaries_chroma/ with split embeddings")
    parser.add_argument("--run_dir", required=True,
                        help="Path to run output dir (e.g. experiment/longmem/output/oss-20b-0427)")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Limit to specific category subdirs (default: all)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be done without writing")
    parser.add_argument("--export_only", action="store_true",
                        help="Skip rebuild; just export summaries_split_meta.jsonl from existing ChromaDB")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Artifact dir names to skip (e.g. artifacts_gpt4_1d4ab0c9)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: run_dir not found: {run_dir}")
        sys.exit(1)

    artifact_dirs = []
    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if args.categories and subdir.name not in args.categories:
            continue
        for art_dir in sorted(subdir.iterdir()):
            if (
                art_dir.is_dir()
                and art_dir.name.startswith("artifacts_")
                and art_dir.name not in (args.exclude or [])
            ):
                artifact_dirs.append(art_dir)

    print(f"Found {len(artifact_dirs)} artifact directories")

    if args.export_only:
        print("EXPORT ONLY mode — reading from existing ChromaDB\n")
        ok = skip = error = 0
        t0 = time.time()
        for i, art_dir in enumerate(artifact_dirs, 1):
            try:
                result = export_artifact(art_dir)
                status = result.get("status")
                if status == "ok":
                    ok += 1
                    print(f"[{i}/{len(artifact_dirs)}] {art_dir.name}: {result['entries']} entries → {Path(result['path']).name}")
                else:
                    skip += 1
                    print(f"[{i}/{len(artifact_dirs)}] {art_dir.name}: SKIP — {result.get('reason')}")
            except Exception as e:
                error += 1
                print(f"[{i}/{len(artifact_dirs)}] {art_dir.name}: ERROR — {e}")
        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.1f}s — ok={ok}  skip={skip}  error={error}")
        return

    print(f"Loading raw context from {SCRIPT_DATA_DIR} ...")
    lookup = RawContextLookup(SCRIPT_DATA_DIR)
    lookup._ensure_loaded()
    print(f"  Loaded {len(lookup._index)} sessions")

    print("Loading llmlingua compressor ...")
    compressor = get_compressor()
    print("  Ready\n")

    if args.dry_run:
        print("DRY RUN — no files will be written\n")

    ok = skip = error = 0
    t0 = time.time()
    for i, art_dir in enumerate(artifact_dirs, 1):
        try:
            result = rebuild_artifact(art_dir, lookup, compressor, args.dry_run)
            status = result.get("status")
            if status in ("ok", "dry_run"):
                ok += 1
                print(f"[{i}/{len(artifact_dirs)}] {art_dir.name}: {result}")
            else:
                skip += 1
                print(f"[{i}/{len(artifact_dirs)}] {art_dir.name}: SKIP — {result.get('reason')}")
        except Exception as e:
            error += 1
            print(f"[{i}/{len(artifact_dirs)}] {art_dir.name}: ERROR — {e}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — ok={ok}  skip={skip}  error={error}")


if __name__ == "__main__":
    main()
