"""
Same as replay_fact_multi_dataset.py but fact extraction only uses user-side turns.
Assistant turns are filtered out before extraction to reduce noise and cost.

Output files use the suffix _replay_fact_user_only.csv to avoid overwriting prior runs.

Usage:
    python experiment/longmem/replay_fact_user_only.py
    python experiment/longmem/replay_fact_user_only.py --dry-run
    python experiment/longmem/replay_fact_user_only.py --category multi_session
    python experiment/longmem/replay_fact_user_only.py --no-judge
    python experiment/longmem/replay_fact_user_only.py --force
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
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

_FACT_EXTRACTION_SYSTEM_PROMPT = """\
You are a precise fact extractor. Given user messages from a conversation, extract all personal facts, events, preferences, and background information about the user.

For each fact provide:
- what: concise description of the fact
- when: time reference if any (or "N/A")
- who: people/entities involved (or "N/A")
- why: reason/context if relevant (or "N/A")

Extract every meaningful fact. Do not skip anything personal or informative about the user.

Respond with a JSON object: {"facts": [{"what": "...", "when": "...", "who": "...", "why": "..."}]}"""

_ANSWER_SYSTEM_PROMPT = (
    "You are a concise and accurate assistant. "
    "Use the Retrieved Context. If context is insufficient, use general knowledge, "
    "but prefer retrieved facts. Answer directly."
)


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _load_evidence_map(logs_dir: Path) -> dict[str, list[str]]:
    """Returns {request_id: [summary_id, ...]} from kg_retrieval_evidence.jsonl."""
    path = logs_dir / "kg_retrieval_evidence.jsonl"
    if not path.exists():
        return {}
    result: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "evidence_items":
                result[e["request_id"]] = [
                    item["summary_id"] for item in e.get("items", [])
                ]
    return result


def _load_question_map(logs_dir: Path) -> dict[str, str]:
    """Returns {request_id: question} from kg_retriever.jsonl."""
    path = logs_dir / "kg_retriever.jsonl"
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "build_kg_context_start":
                result[e["request_id"]] = e.get("question", "")
    return result


def _load_summaries_meta(artifacts_dir: Path) -> dict[str, dict]:
    """Returns {summary_id: {session_id, dialogue_datetime, ...}} from summaries_meta.jsonl."""
    path = artifacts_dir / "summaries_meta.jsonl"
    if not path.exists():
        return {}
    result: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = item.get("id") or item.get("summary_id")
            if sid:
                result[sid] = item
    return result


def _build_session_raw_texts(source_csv: Path) -> dict[str, str]:
    """Returns {session_id: user-only conversation text} from the source dataset CSV."""
    if not source_csv.exists():
        return {}
    df = pd.read_csv(source_csv)
    df.columns = [c.lstrip("﻿") for c in df.columns]

    sid_col = _find_col(df, ["session_id"])
    role_col = _find_col(df, ["role"])
    content_col = _find_col(df, ["content"])
    turn_col = _find_col(df, ["turn_index", "turn"])

    if not all([sid_col, content_col]):
        return {}

    result: dict[str, str] = {}
    for sid, grp in df.groupby(sid_col, sort=False):
        if turn_col:
            grp = grp.sort_values(turn_col)
        lines: list[str] = []
        for _, row in grp.iterrows():
            role = str(row.get(role_col, "")).strip().lower() if role_col else ""
            if role_col and role != "user":
                continue
            content = str(row.get(content_col, "")).strip()
            if not content or content == "nan":
                continue
            lines.append(f"User: {content}")
        result[str(sid)] = "\n".join(lines)
    return result


def _parse_session_id(summary_id: str) -> str | None:
    """Extract session_id from summary_id (format: {session_id}:{message_id})."""
    if ":" not in summary_id:
        return None
    return summary_id.rsplit(":", 1)[0]


def _extract_facts(llm: LLMClient, raw_text: str, dialogue_datetime: str | None) -> list[str]:
    """Run fact extraction on raw text. Falls back to [raw_text] on error."""
    from datetime import datetime

    if dialogue_datetime:
        try:
            dt = datetime.fromisoformat(dialogue_datetime.replace("/", "-").split(" (")[0])
            date_str = dt.strftime("%A, %B %d, %Y")
            date_iso = dt.isoformat()
        except (ValueError, AttributeError):
            date_str = dialogue_datetime
            date_iso = dialogue_datetime
    else:
        now = datetime.utcnow()
        date_str = now.strftime("%A, %B %d, %Y")
        date_iso = now.isoformat()

    user_message = (
        f"Extract facts from the following text chunk.\n\n"
        f"Chunk: 1/1\n"
        f"Event Date: {date_str} ({date_iso})\n"
        f"Context: none\n\n"
        f"Text:\n{raw_text}"
    )

    try:
        resp = llm.chat(
            messages=[
                {"role": "system", "content": _FACT_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        content = resp.choices[0].message.content or ""
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"no JSON object in response; content[:200]={content[:200]!r}")
        json_str = content[start:end]
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            # strip control characters and retry once
            import re as _re
            json_str = _re.sub(r"[\x00-\x1f\x7f]", " ", json_str)
            result = json.loads(json_str)
        def _str(v) -> str:
            if isinstance(v, list):
                return ", ".join(str(x) for x in v)
            return str(v) if v is not None else ""

        facts_raw = result.get("facts") or []
        rendered: list[str] = []
        for f in facts_raw:
            if not isinstance(f, dict):
                continue
            what = _str(f.get("what")).strip()
            if not what:
                continue
            parts = [what]
            when = _str(f.get("when")).strip()
            if when and when.upper() != "N/A":
                parts.append(f"When: {when}")
            who = _str(f.get("who")).strip()
            if who and who.upper() != "N/A":
                parts.append(f"Involving: {who}")
            why = _str(f.get("why")).strip()
            if why and why.upper() != "N/A":
                parts.append(why)
            rendered.append(" | ".join(parts))
        return rendered if rendered else [raw_text]
    except Exception as e:
        print(f"    [WARN] fact extraction failed: {type(e).__name__}: {e}")
        # return individual user lines instead of one big blob
        lines = [ln.removeprefix("User:").strip() for ln in raw_text.splitlines() if ln.strip()]
        return lines if lines else [raw_text]


def _build_context(
    llm: LLMClient,
    selected_ids: list[str],
    summaries_meta: dict[str, dict],
    session_raw_texts: dict[str, str],
) -> str:
    """Build fact-extracted context from user-only text of selected sessions."""
    lines: list[str] = []
    seen_sessions: set[str] = set()
    for sid in selected_ids:
        if not sid:
            continue
        session_id = _parse_session_id(sid)
        if not session_id or session_id in seen_sessions:
            continue
        seen_sessions.add(session_id)

        raw_text = session_raw_texts.get(session_id, "").strip()
        if not raw_text:
            continue

        meta = summaries_meta.get(sid, {})
        dt = meta.get("dialogue_datetime")

        facts = _extract_facts(llm, raw_text, dt)
        dt_prefix = f"[{dt}]" if dt else ""
        lines.append(f"  • {dt_prefix}[session={session_id}]")
        for fact in facts:
            lines.append(f"    • {fact}")

    if not lines:
        return "(no replay context available)"
    return "### Evidence Facts\n" + "\n".join(lines)


def _answer_with_context(llm: LLMClient, question: str, context: str, question_date: str | None) -> str:
    system_content = _ANSWER_SYSTEM_PROMPT
    if question_date:
        system_content += f"\n\nCurrent Date/Time: {question_date}"
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
    """Process one dataset CSV. Returns (processed, skipped)."""
    name = csv_path.stem
    out_path = cat_dir / f"{name}_replay_fact_user_only.csv"

    if out_path.exists() and not force:
        print(f"  [SKIP] {name}: output already exists ({out_path.name})")
        return 0, 1

    logs_dir = cat_dir / f"logs_{name}"
    artifacts_dir = cat_dir / f"artifacts_{name}"
    source_csv = SCRIPT_DATA_DIR / dir_name / f"{name}.csv"

    if not logs_dir.exists() or not artifacts_dir.exists():
        print(f"  [SKIP] {name}: missing logs or artifacts dir")
        return 0, 1

    if not source_csv.exists():
        print(f"  [SKIP] {name}: source CSV not found at {source_csv}")
        return 0, 1

    question_map = _load_question_map(logs_dir)
    evidence_map = _load_evidence_map(logs_dir)
    summaries_meta = _load_summaries_meta(artifacts_dir)
    session_raw_texts = _build_session_raw_texts(source_csv)

    if not question_map or not evidence_map:
        print(f"  [SKIP] {name}: no retrieval logs found")
        return 0, 1

    q_to_evidence: dict[str, list[str]] = {}
    for req_id, q in question_map.items():
        if req_id in evidence_map:
            q_to_evidence[q.strip().lower()] = evidence_map[req_id]

    df = pd.read_csv(csv_path)
    q_col = _find_col(df, ["question"])
    g_col = _find_col(df, ["answer", "gold_answer"])

    if not q_col:
        print(f"  [SKIP] {name}: no question column")
        return 0, 1

    rows = []
    processed = 0
    for _, row in df.iterrows():
        question = str(row.get(q_col, "")).strip()
        gold = str(row.get(g_col, "")).strip() if g_col else ""
        question_date = None
        for col in ["question_date", "dialogue_datetime", "date", "timestamp"]:
            if col in df.columns and pd.notna(row.get(col)):
                question_date = str(row[col]).strip()
                break

        selected_ids = q_to_evidence.get(question.lower(), [])

        if dry_run:
            sessions = {_parse_session_id(s) for s in selected_ids if _parse_session_id(s)}
            print(f"  [DRY] {question[:60]} -> {len(selected_ids)} ids, {len(sessions)} sessions")
            rows.append({
                "question": question,
                "question_date": question_date or "",
                "Retrieved_Context": f"(dry-run: {len(selected_ids)} ids)",
                "Generated_Answer": "(dry-run)",
                "answer": gold,
                "correctness": "",
            })
            processed += 1
            continue

        t0 = time.time()
        context = _build_context(llm, selected_ids, summaries_meta, session_raw_texts)
        answer = _answer_with_context(llm, question, context, question_date)
        elapsed = round(time.time() - t0, 2)
        print(f"  {question[:60]!r} -> {answer[:60]!r} ({elapsed}s)")

        correctness = ""
        if not no_judge and judge and llm and gold:
            correctness = str(judge.judge_single(
                llm, question=question, gold=gold, generated=answer
            ))

        rows.append({
            "question": question,
            "question_date": question_date or "",
            "Retrieved_Context": context,
            "Generated_Answer": answer,
            "answer": gold,
            "correctness": correctness,
        })
        processed += 1

    out_df = pd.DataFrame(rows)
    if not dry_run:
        out_df.to_csv(out_path, index=False)
        print(f"  -> saved {out_path.name}")

    return processed, 0


def main():
    parser = argparse.ArgumentParser(description="Replay fact extraction (user-only) on prior longmem run output")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Path to multi_dataset_output dir")
    parser.add_argument("--category", default="", help="Only process this category (e.g. multi_session)")
    parser.add_argument("--name", default="", help="Only process this dataset name (e.g. 2311e44b)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without calling LLM")
    parser.add_argument("--no-judge", action="store_true", help="Skip judge stage")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    llm = None if args.dry_run else LLMClient(timeout=300.0)
    judge = None if (args.dry_run or args.no_judge) else JudgeStage()

    total_processed = total_skipped = 0

    for dir_name, category in _DIR_TO_CATEGORY.items():
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
