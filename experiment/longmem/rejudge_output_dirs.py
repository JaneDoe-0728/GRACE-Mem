"""Re-judge specific output/<run> dirs with the new category-aware judge prompt.

Writes results to a NEW column (default `correctness_new`); the original
`correctness` column is left untouched. Category is taken from the sub-dir name
(single_session_user, temporal_reasoning, ...), so each row uses the matching
per-category rubric from prompts/judge.py. The judge LLM is asked to reply as
JSON {reasoning, correct}; parsing falls back to Yes/No token parsing.

Aggregate CSVs (progress.csv, all_answers.csv) are skipped. Rows already filled
with 0/1 in the target column are skipped, so the run is resumable.

Usage:
    python experiment/longmem/rejudge_output_dirs.py            # all 8 dirs
    python experiment/longmem/rejudge_output_dirs.py --dirs rerank16
    python experiment/longmem/rejudge_output_dirs.py --dry-run
    python experiment/longmem/rejudge_output_dirs.py --col correctness_v2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from KG.llm import LLMClient
from experiment.longmem.prompts import build_judge_messages
from experiment.longmem.stages.judge import JudgeStage

OUTPUT_DIR = _ROOT / "experiment" / "longmem" / "output"

# The 8 experiments to re-judge (label -> run dir), confirmed with the user.
TARGET_DIRS: list[str] = [
    "split-embed",      # split_summary
    "sweep-topk16",     # split_summary_direct_sum_tk16
    "sweep-topk24",     # split_summary_direct_sum_tk24
    "sweep-topk32",     # split_summary_direct_sum_tk32
    "extraslot-t50",    # split_sum_extra_sum_vec_0.5
    "extraslot-t40",    # split_sum_extra_sum_vec_0.4
    "extraslot-t35",    # split_sum_extra_sum_vec_0.35
    "rerank16",         # split_sum_extra_add_reranker
]

_DIR_TO_CATEGORY: dict[str, str] = {
    "single_session_user": "single-session-user",
    "single_session_assistant": "single-session-assistant",
    "multi_session": "multi-session",
    "single_session_preference": "single-session-preference",
    "temporal_reasoning": "temporal-reasoning",
    "knowledge_update": "knowledge-update",
}

_SKIP_FILES = {"progress.csv", "all_answers.csv"}

_parser = JudgeStage()  # for parse_binary_judge fallback


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower().lstrip("﻿"): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _parse_correct(text: str) -> int:
    """Parse JSON {correct: bool} from the judge reply; fall back to Yes/No."""
    if not text:
        return 0
    # Try to locate a JSON object with a "correct" field.
    for match in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "correct" in obj:
            val = obj["correct"]
            if isinstance(val, bool):
                return 1 if val else 0
            return _parser.parse_binary_judge(str(val))
    # No parseable JSON -> Yes/No token fallback.
    return _parser.parse_binary_judge(text)


# 多票去噪:單票 temperature=0 judge 有 ~12% run-to-run 不一致(within-run
# correctness_new vs correctness_20b),對 borderline 題(同義答案、偏好對齊、
# 棄答)會隨機翻。3 票(temp 0/0.3/0.6)多數決把 adjudicate-v1 錯題回收 15 題
# 假失分(14 題 3/3 全票),整體 77.4→80.4(+3.0pp);真幻覺仍 0 票維持錯。
_VOTE_TEMPS = (0.0, 0.3, 0.6)


def _judge_single(llm, *, question: str, gold: str, generated: str, category: str,
                  is_abstention: bool | None = None, votes: int = 1,
                  escalate: bool = False) -> int:
    messages = build_judge_messages(
        question=question, gold=gold, generated=generated, category=category,
        is_abstention=is_abstention,
    )
    import time
    from requests.exceptions import HTTPError, RequestException

    def _one(temp: float) -> int:
        # Retry with exponential backoff on transient errors: 429 rate limits,
        # 5xx server errors, and network errors (ReadTimeout / ConnectionError).
        # Needed once --workers / --votes go up: many concurrent calls make
        # OpenAI timeouts / 5xx likely, and 429-only retry would crash the run.
        delay = 2.0
        for attempt in range(8):
            try:
                resp = llm.chat(messages=messages, temperature=temp, max_tokens=256)
                return _parse_correct(resp.choices[0].message.content or "")
            except HTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (429, 500, 502, 503, 504) and attempt < 7:
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                    continue
                raise
            except RequestException:
                # transient network errors (ReadTimeout / ConnectionError / etc.)
                if attempt < 7:
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                    continue
                raise
        return 0

    # 棄答題恆定單票:_abs 走專用 rubric(判「是否乾淨棄答」),多票多數決反而把
    # temp=0 判對的 borderline 棄答淹掉(實測 80ec1f4f/29f2956b 單票對、3 票錯),
    # 且真幻覺單票已 0。3 票去噪只對通用 rubric 的同義/偏好 borderline 有益。
    if is_abstention or votes <= 1:
        return _one(0.0)
    temps = [_VOTE_TEMPS[i % len(_VOTE_TEMPS)] for i in range(votes)]
    # escalate(--first 用):非abs 題先判單票(temps[0]),判對就直接 carry 1,
    # 只有判錯才補判剩下的票湊多數決。等價於 JUDGING.md §三「carry 判對題、
    # 錯題 3 票重判」的 carry 上界口徑,但省去 prep 分欄——單題內完成。
    if escalate:
        first = _one(temps[0])
        if first == 1:
            return 1
        tally = first + sum(_one(t) for t in temps[1:])
        return 1 if tally * 2 >= votes else 0
    tally = sum(_one(t) for t in temps)
    return 1 if tally * 2 >= votes else 0  # majority (ties -> correct)


def rejudge_csv(path: Path, *, category: str, llm, col: str, dry_run: bool,
                votes: int = 1, escalate: bool = False) -> tuple[int, int]:
    """Returns (judged, skipped) counts."""
    df = pd.read_csv(path, encoding="utf-8-sig")

    q_col = _find_col(df, ["question"])
    g_col = _find_col(df, ["answer", "gold_answer"])
    gen_col = _find_col(df, ["Generated_Answer", "generated_answer", "model_answer"])

    if not all([q_col, g_col, gen_col]):
        print(f"  [SKIP] {path.name}: missing columns (got {df.columns.tolist()})")
        return 0, 0

    if col not in df.columns:
        df[col] = ""

    # 棄答題以資料集 `_abs` 檔名 tag 為權威來源 → 走專用 abstention rubric。
    is_abs = path.stem.endswith("_abs")

    judged = skipped = 0
    for i, row in df.iterrows():
        existing = str(row.get(col, "")).strip()
        if existing in ("0", "1"):
            skipped += 1
            continue

        question = str(row[q_col]).strip()
        gold = str(row[g_col]).strip()
        generated = str(row[gen_col]).strip()
        if not generated or not question:
            df.at[i, col] = ""
            continue

        if dry_run:
            df.at[i, col] = -1
            judged += 1
            continue

        df.at[i, col] = _judge_single(
            llm, question=question, gold=gold, generated=generated, category=category,
            is_abstention=is_abs, votes=votes, escalate=escalate,
        )
        judged += 1

    if not dry_run and judged:
        df.to_csv(path, index=False)

    return judged, skipped


def _openai_key_from_env_file() -> str | None:
    """Read OPENAI_API_KEY from the environment, or from a (possibly commented) .env line."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    env_path = _ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            m = re.search(r'OPENAI_API_KEY="?(sk-[^"\s]+)', line)
            if m:
                return m.group(1)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--col", default="correctness_new", help="Column for new judge results")
    parser.add_argument("--dry-run", action="store_true", help="Count rows without calling the LLM")
    parser.add_argument("--dirs", nargs="*", default=None, help="Subset of run dirs (default: all 8)")
    parser.add_argument("--judge-model", default="gpt-4o-mini",
                        help="Cloud judge model (default: gpt-4o-mini)")
    parser.add_argument("--judge-base-url", default="https://api.openai.com/v1",
                        help="Cloud judge base URL (default: OpenAI API)")
    parser.add_argument("--workers", type=int, default=6,
                        help="File-level concurrency (default 6; OpenAI endpoints handle this fine, "
                             "local LM Studio is capped by its own parallel setting)")
    parser.add_argument("--votes", type=int, default=1,
                        help="每題 judge 投票數(多數決去噪;3=temp 0/0.3/0.6,實測回收 "
                             "adjudicate-v1 假失分 +3.0pp)。1=單票(原行為)")
    parser.add_argument("--first", action="store_true",
                        help="從未判過的實驗一鍵完成(取代 votes1→prep→votes3 三步):"
                             "非abs 題先判單票、判對即 carry、判錯才補足 --votes 票多數決;"
                             "abs 題單票強化 rubric。空欄→全判,結果直接是合成口徑。"
                             "預設 False(維持既有 carry-via-prep 流程,只判空欄)。")
    args = parser.parse_args()

    run_dirs = args.dirs or TARGET_DIRS
    if args.dry_run:
        llm = None
    else:
        api_key = _openai_key_from_env_file() if (args.judge_base_url or "").rstrip("/").endswith("openai.com/v1") else None
        llm = LLMClient(base_url=args.judge_base_url, model_name=args.judge_model, api_key=api_key)
        print(f"[JUDGE] model={llm.model_name} base_url={llm._base_url}")

    grand_judged = grand_skipped = grand_files = 0

    from concurrent.futures import ThreadPoolExecutor

    for run_dir in run_dirs:
        base = OUTPUT_DIR / run_dir
        if not base.exists():
            print(f"[MISS] {run_dir}: dir not found")
            continue
        print(f"\n=== {run_dir} ===")
        for cat_sub, category in _DIR_TO_CATEGORY.items():
            cat_dir = base / cat_sub
            if not cat_dir.exists():
                continue
            csvs = [p for p in sorted(cat_dir.glob("*.csv")) if p.name not in _SKIP_FILES]
            dj = ds = 0
            if args.workers > 1 and not args.dry_run:
                # 一檔一題,檔案級併發安全(各自讀寫不同 CSV)
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futs = [ex.submit(rejudge_csv, p, category=category, llm=llm,
                                      col=args.col, dry_run=args.dry_run, votes=args.votes,
                                      escalate=args.first)
                            for p in csvs]
                    for fut in futs:
                        j, s = fut.result()
                        dj += j
                        ds += s
                        grand_files += 1
            else:
                for csv_path in csvs:
                    j, s = rejudge_csv(csv_path, category=category, llm=llm, col=args.col,
                                       dry_run=args.dry_run, votes=args.votes,
                                       escalate=args.first)
                    dj += j
                    ds += s
                    grand_files += 1
            grand_judged += dj
            grand_skipped += ds
            print(f"  [{cat_sub}] files={len(csvs)} judged={dj} skipped={ds}")

    print(f"\nDone. files={grand_files} judged={grand_judged} skipped={grand_skipped} col={args.col}")


if __name__ == "__main__":
    main()
