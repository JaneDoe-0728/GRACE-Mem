"""
rebuild_split_summaries.py — Rebuild summaries_chroma/ with split user/assistant embeddings.

For each artifacts_*/ directory in a completed run, replaces the summaries_chroma/
collection with two VDB entries per turn:
  {session_id}:{message_id}:u  → embed(user_text_raw)
  {session_id}:{message_id}:a  → embed(llmlingua(assistant_text) + rewrite_temporal_text())

KG entities/relationships are NOT touched; summary_id prov links remain valid.

Usage:
    python experiment/longmem/rebuild_split_summaries.py \\
        --run_dir experiment/longmem/output/oss-20b-0427 \\
        [--categories knowledge_update single_session_user ...] \\
        [--dry_run]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from KG.utils.raw_context_lookup import RawContextLookup
from KG.utils.query_time_parser import parse_query_time
from KG.utils.temporal import build_time_context, rewrite_temporal_text

SCRIPT_DATA_DIR = "experiment/longmem/script_data"


def get_compressor():
    from llmlingua import PromptCompressor
    return PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True,
    )


def compress_and_rewrite(text: str, compressor, dialogue_datetime: Optional[str]) -> str:
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


def rebuild_artifact(artifact_dir: Path, lookup: RawContextLookup, compressor, dry_run: bool) -> dict:
    meta_path = artifact_dir / "summaries_meta.jsonl"
    chroma_dir = artifact_dir / "summaries_chroma"
    backup_dir = artifact_dir / "summaries_chroma_bak"

    if not meta_path.exists():
        return {"status": "skip", "reason": "no_meta"}

    # Already rebuilt successfully (backup exists and new chroma exists)
    if backup_dir.exists() and chroma_dir.exists() and not dry_run:
        return {"status": "skip", "reason": "already_rebuilt"}

    turns = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                turns.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not turns:
        return {"status": "skip", "reason": "empty_meta"}

    if dry_run:
        return {"status": "dry_run", "turns": len(turns)}

    # Backup old chroma dir (once only)
    if chroma_dir.exists() and not backup_dir.exists():
        shutil.copytree(chroma_dir, backup_dir)

    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)

    from KG.storage.chroma_vdb import SummariesVDB
    vdb = SummariesVDB(dim=1024, path=str(chroma_dir), collection_name="summaries")

    n_added = 0
    n_missing = 0
    for turn in turns:
        session_id = turn.get("session_id")
        message_id = turn.get("message_id")
        dialogue_datetime = turn.get("dialogue_datetime")

        if session_id is None or message_id is None:
            n_missing += 1
            continue

        user_text = lookup.get_user_text(str(session_id), int(message_id))
        assistant_text = lookup.get_assistant_text(str(session_id), int(message_id))

        if not user_text and not assistant_text:
            n_missing += 1
            continue

        user_text = user_text or ""

        if assistant_text:
            assistant_summary = compress_and_rewrite(assistant_text, compressor, dialogue_datetime)
        else:
            # Fall back to existing summary_text if assistant turn missing from CSV
            assistant_summary = turn.get("summary_text", "")

        if not user_text:
            user_text = turn.get("summary_text", "")

        if not user_text and not assistant_summary:
            n_missing += 1
            continue

        vdb.add_split_turns(
            session_id=session_id,
            message_id=int(message_id),
            user_text=user_text,
            assistant_summary=assistant_summary,
            dialogue_datetime=dialogue_datetime,
        )
        n_added += 1

    vdb.close()
    return {"status": "ok", "turns": len(turns), "added": n_added, "missing": n_missing}


def export_artifact(artifact_dir: Path) -> dict:
    """Export summaries_chroma/ entries to summaries_split_meta.jsonl for inspection."""
    chroma_dir = artifact_dir / "summaries_chroma"
    out_path = artifact_dir / "summaries_split_meta.jsonl"

    if not chroma_dir.exists():
        return {"status": "skip", "reason": "no_chroma"}

    from KG.storage.chroma_vdb import SummariesVDB
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
            if art_dir.is_dir() and art_dir.name.startswith("artifacts_"):
                if art_dir.name not in (args.exclude or []):
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
