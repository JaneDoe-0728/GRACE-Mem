"""
Gold summary oracle baseline:
  1. Find has_answer=True turns in source CSV
  2. Look up the corresponding LLMlingua-compressed summary in summaries_meta.jsonl
  3. Extract user-side only (strip the Assistant portion)
  4. Use as context → LLM answers question → judge

No re-ingest, no retrieval step.
Output: {name}_gold_summary.csv

Usage:
    python experiment/longmem/gold_summary_eval.py
    python experiment/longmem/gold_summary_eval.py --dry-run
    python experiment/longmem/gold_summary_eval.py --category single_session_user
    python experiment/longmem/gold_summary_eval.py --name 001be529
    python experiment/longmem/gold_summary_eval.py --no-judge
    python experiment/longmem/gold_summary_eval.py --force
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from KG.llm import LLMClient
from experiment.longmem.stages.judge import JudgeStage

OUTPUT_DIR = _ROOT / "experiment" / "longmem" / "multi_dataset_output"
SCRIPT_DATA_DIR = _ROOT / "experiment" / "longmem" / "script_data"

_DIR_TO_CATEGORY: dict[str, str] = {
    "single_session_user": "single-session-user",
    "single_session_assistant": "single-session-assistant",
    "multi_session": "multi-session",
    "single_session_preference": "single-session-preference",
    "temporal_reasoning": "temporal-reasoning",
    "knowledge_update": "knowledge-update",
}

_ANSWER_SYSTEM_PROMPT = (
    "You are a concise and accurate assistant. "
    "Use the Retrieved Context. If context is insufficient, use general knowledge, "
    "but prefer retrieved facts. Answer directly."
)

_ASSISTANT_SPLIT_RE = re.compile(r"\n\s*[Aa]ssistant\s*:?", re.MULTILINE)


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _summary_id_for_turn(session_id: str, turn_index: int, role: str) -> str:
    """Map a source CSV turn to its summary_id in summaries_meta.

    Summaries are stored per turn-pair with message_id = assistant turn index.
    user turn (odd)      -> message_id = turn_index + 1
    assistant turn (even)-> message_id = turn_index
    """
    if role.strip().lower() == "user":
        message_id = turn_index + 1
    else:
        message_id = turn_index
    return f"{session_id}:{message_id}"


def _load_summaries_meta(artifacts_dir: Path) -> dict[str, dict]:
    """Return {summary_id: record} from summaries_meta.jsonl."""
    path = artifacts_dir / "summaries_meta.jsonl"
    if not path.exists():
        return {}
    result: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = item.get("summary_id") or item.get("id")
            if sid:
                result[str(sid)] = item
    return result


def _extract_user_side(summary_text: str) -> str:
    """Keep only the user portion of a LLMlingua-compressed summary.

    Splits on the first '\\n Assistant' marker. If the marker was dropped by
    LLMlingua, the whole text is returned (user content dominates in that case).
    """
    parts = _ASSISTANT_SPLIT_RE.split(summary_text, maxsplit=1)
    return parts[0].strip()


def _build_gold_context(
    source_csv: Path,
    artifacts_dir: Path,
) -> tuple[str, list[str]]:
    """Build oracle context from has_answer=True turns.

    Returns (context_string, list_of_matched_summary_ids).
    """
    df = pd.read_csv(source_csv)
    df.columns = [c.lstrip("﻿") for c in df.columns]

    has_answer_col = _find_col(df, ["has_answer"])
    session_col = _find_col(df, ["session_id"])
    turn_col = _find_col(df, ["turn_index", "turn"])
    role_col = _find_col(df, ["role"])

    if not has_answer_col or not session_col or not turn_col:
        return "(no has_answer column in source CSV)", []

    gold_rows = df[df[has_answer_col] == True]
    if gold_rows.empty:
        return "(no has_answer=True rows in source CSV)", []

    summaries_meta = _load_summaries_meta(artifacts_dir)

    snippets: list[str] = []
    matched_ids: list[str] = []

    for _, row in gold_rows.iterrows():
        session_id = str(row[session_col]).strip()
        turn_index = int(row[turn_col])
        role = str(row[role_col]).strip() if role_col else "user"

        summary_id = _summary_id_for_turn(session_id, turn_index, role)
        meta = summaries_meta.get(summary_id)

        if meta is None:
            print(f"    [WARN] summary_id not found: {summary_id}")
            continue

        user_side = _extract_user_side(meta.get("summary_text", ""))
        if not user_side:
            continue

        dt = meta.get("dialogue_datetime", "")
        prefix = f"[{dt}]" if dt else ""
        snippets.append(f"{prefix} {user_side}".strip())
        matched_ids.append(summary_id)

    if not snippets:
        return "(no matching summaries found)", matched_ids

    context = "### User Memory\n" + "\n\n".join(snippets)
    return context, matched_ids


def _answer_with_context(
    llm: LLMClient,
    question: str,
    context: str,
    question_date: str | None,
) -> str:
    system_content = _ANSWER_SYSTEM_PROMPT
    if question_date:
        system_content += f"\n\nCurrent Date/Time: {question_date}"
        system_content += (
            "\nNote: When answering temporal questions "
            "(e.g., 'how long ago', 'how many months'), calculate based on this date."
        )
    system_content += f"\n\n---Retrieved Context---\n{context}\n------------------"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Question: {question}\n\nAnswer:"},
    ]
    resp = llm.chat(messages=messages, temperature=0.0, max_tokens=1024)
    return (resp.choices[0].message.content or "").strip()


def process_csv(
    csv_path: Path,
    *,
    dir_name: str,
    cat_dir: Path,
    judge: JudgeStage | None,
    llm: LLMClient | None,
    dry_run: bool,
    no_judge: bool,
    force: bool,
) -> tuple[int, int]:
    """Process one dataset output CSV. Returns (processed, skipped)."""
    name = csv_path.stem
    out_path = cat_dir / f"{name}_gold_summary.csv"

    if out_path.exists() and not force:
        print(f"  [SKIP] {name}: output already exists ({out_path.name})")
        return 0, 1

    artifacts_dir = cat_dir / f"artifacts_{name}"
    source_csv = SCRIPT_DATA_DIR / dir_name / f"{name}.csv"

    if not artifacts_dir.exists():
        print(f"  [SKIP] {name}: artifacts dir not found")
        return 0, 1

    if not source_csv.exists():
        print(f"  [SKIP] {name}: source CSV not found at {source_csv}")
        return 0, 1

    if dry_run:
        df_src = pd.read_csv(source_csv)
        df_src.columns = [c.lstrip("﻿") for c in df_src.columns]
        has_col = _find_col(df_src, ["has_answer"])
        n = int(df_src[has_col].sum()) if has_col else 0
        summaries_meta = _load_summaries_meta(artifacts_dir)
        print(f"  [DRY] {name}: has_answer=True count={n}, summaries_meta entries={len(summaries_meta)}")
        return 1, 0

    # Read question from already-processed output CSV
    df_out = pd.read_csv(csv_path)
    q_col = _find_col(df_out, ["question"])
    g_col = _find_col(df_out, ["answer", "gold_answer"])

    if not q_col:
        print(f"  [SKIP] {name}: no question column in output CSV")
        return 0, 1

    # Build gold context once (same for all rows — usually 1 question per CSV)
    context, matched_ids = _build_gold_context(source_csv, artifacts_dir)
    print(f"  [CONTEXT] {name}: matched {len(matched_ids)} gold summaries → {len(context)} chars")

    rows: list[dict] = []
    processed = 0

    for _, row in df_out.iterrows():
        question = str(row.get(q_col, "")).strip()
        if not question or question == "nan":
            continue
        gold = str(row.get(g_col, "")).strip() if g_col else ""

        question_date = None
        for col in ["question_date", "dialogue_datetime", "date", "timestamp"]:
            if col in df_out.columns and pd.notna(row.get(col)):
                val = str(row[col]).strip()
                if val and val != "nan":
                    question_date = val
                    break

        t0 = time.time()
        answer = _answer_with_context(llm, question, context, question_date)
        elapsed = round(time.time() - t0, 2)
        print(f"  {question[:60]!r} -> {answer[:60]!r} ({elapsed}s)")

        correctness = ""
        if not no_judge and judge and llm and gold:
            correctness = str(judge.judge_single(llm, question=question, gold=gold, generated=answer))
            print(f"  [JUDGE] {'correct' if correctness == '1' else 'incorrect'} ({correctness})")

        rows.append({
            "question": question,
            "question_date": question_date or "",
            "Retrieved_Context": context,
            "Generated_Answer": answer,
            "answer": gold,
            "correctness": correctness,
        })
        processed += 1

    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  -> saved {out_path.name}")

    return processed, 0


def main():
    parser = argparse.ArgumentParser(
        description="Gold summary oracle: has_answer=True summaries (user-side) as LLM context"
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Path to multi_dataset_output dir")
    parser.add_argument("--category", default="", help="Only process this category (e.g. single_session_user)")
    parser.add_argument("--name", default="", help="Only process this dataset name (e.g. 001be529)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without calling LLM")
    parser.add_argument("--no-judge", action="store_true", help="Skip judge stage")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    llm = None if args.dry_run else LLMClient(timeout=300.0)
    judge = None if (args.dry_run or args.no_judge) else JudgeStage()

    total_processed = total_skipped = 0

    for dir_name in _DIR_TO_CATEGORY:
        if args.category and dir_name != args.category:
            continue
        cat_dir = output_dir / dir_name
        if not cat_dir.exists():
            continue

        csvs = sorted(
            p for p in cat_dir.glob("*.csv")
            if not p.stem.endswith("_abs")
            and not p.stem.endswith("_replay_fact")
            and not p.stem.endswith("_replay_fact_user_only")
            and not p.stem.endswith("_gold_summary")
            and p.stem != "all_answers"
        )
        if args.name:
            csvs = [p for p in csvs if p.stem == args.name]

        print(f"\n[{dir_name}] {len(csvs)} datasets")
        for csv_path in csvs:
            p, s = process_csv(
                csv_path,
                dir_name=dir_name,
                cat_dir=cat_dir,
                judge=judge,
                llm=llm,
                dry_run=args.dry_run,
                no_judge=args.no_judge,
                force=args.force,
            )
            total_processed += p
            total_skipped += s

    print(f"\nDone. processed={total_processed} skipped={total_skipped}")


if __name__ == "__main__":
    main()
