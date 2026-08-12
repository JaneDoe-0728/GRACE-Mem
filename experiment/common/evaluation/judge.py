"""Unified LLM judge for LoCoMo and LongMemEval outputs.

The default protocol uses gpt-4o-mini. A correct first-pass verdict is carried
forward; an incorrect verdict is judged again with temperatures 0.0, 0.3, and
0.6. LongMemEval abstention files (``*_abs.csv``) always use one vote with the
dedicated abstention rubric.

Examples:
    uv run python experiment/common/evaluation/judge.py locomo my-run
    uv run python experiment/common/evaluation/judge.py longmem my-run
    uv run python experiment/common/evaluation/judge.py locomo my-run --votes 1
    uv run python experiment/common/evaluation/judge.py longmem my-run --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Sequence

_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from requests.exceptions import HTTPError, RequestException

from grace_mem.llm import LLMClient
from experiment.locomo.helpers.llm import build_judge_standard_messages
from experiment.longmem.prompts import build_judge_messages

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
SINGLE_VOTE_COLUMN = "correctness_4omini"
MAJORITY_VOTE_COLUMN = "correctness_3vote"
ABSTENTION_COLUMN = "correctness_absrubric"
VOTE_TEMPERATURES = (0.0, 0.3, 0.6)

LOCOMO_OUTPUT = _ROOT / "experiment" / "locomo" / "output" / "standard"
LONGMEM_OUTPUT = _ROOT / "experiment" / "longmem" / "output"
LONGMEM_CATEGORIES = {
    "single_session_user": "single-session-user",
    "single_session_assistant": "single-session-assistant",
    "multi_session": "multi-session",
    "single_session_preference": "single-session-preference",
    "temporal_reasoning": "temporal-reasoning",
    "knowledge_update": "knowledge-update",
}
SKIP_LONGMEM_FILES = {"progress.csv", "all_answers.csv"}


def as_binary(value: object) -> int | None:
    """Return 0/1 for a completed CSV verdict, otherwise None."""
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(parsed) if parsed in (0.0, 1.0) else None


def parse_binary_judge(text: str) -> int:
    """Parse permissive yes/no output used by legacy judge endpoints."""
    value = (text or "").strip().lower()
    if "yes" in value and "no" not in value:
        return 1
    if "no" in value and "yes" not in value:
        return 0
    if "1" in value and "0" not in value:
        return 1
    if "correct" in value and "incorrect" not in value:
        return 1
    return 0


def parse_longmem_verdict(text: str) -> int:
    """Parse LongMemEval JSON ``{correct: bool}``, with legacy fallback."""
    if not text:
        return 0
    for match in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            result = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and "correct" in result:
            value = result["correct"]
            if isinstance(value, bool):
                return int(value)
            return parse_binary_judge(str(value))
    return parse_binary_judge(text)


def parse_locomo_verdict(text: str) -> float | None:
    """Parse LoCoMo correct/partial/wrong output."""
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = None
    if isinstance(result, dict) and "label" in result:
        label = str(result["label"]).strip().lower()
        if label == "correct":
            return 1.0
        if label == "partial":
            return 0.5
        if label == "wrong":
            return 0.0

    value = (text or "").strip().lower()
    has_correct = bool(re.search(r"\bcorrect\b", value))
    has_incorrect = bool(re.search(r"\bincorrect\b", value))
    has_partial = bool(re.search(r"\bpartial\b", value))
    has_wrong = bool(re.search(r"\bwrong\b", value))
    if has_correct and not has_wrong and not has_incorrect:
        return 1.0
    if has_partial and not has_correct and not has_wrong and not has_incorrect:
        return 0.5
    if has_wrong or has_incorrect:
        return 0.0
    return None


_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
}


def _parse_anchor_date(value: str) -> datetime | None:
    value = value.strip().rstrip(".,")
    for fmt in (
        "%d %B %Y",
        "%d %B, %Y",
        "%B %d %Y",
        "%B %d, %Y",
        "%B, %Y",
        "%B %Y",
        "%d %b %Y",
        "%d %b, %Y",
        "%Y",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        import dateparser

        return dateparser.parse(value, settings={"PREFER_DAY_OF_MONTH": "first"})
    except Exception:
        return None


def normalize_temporal_gold(gold: str) -> str | None:
    """Expand relative LoCoMo gold dates into a judge hint."""
    text = gold.strip()
    match = re.search(r"the week before\s+(.+)", text, re.IGNORECASE)
    if match:
        anchor = _parse_anchor_date(match.group(1))
        if anchor:
            end = anchor - timedelta(days=1)
            start = end - timedelta(days=6)
            return f"{start:%Y-%m-%d} to {end:%Y-%m-%d} (the 7 days before {anchor:%Y-%m-%d})"

    match = re.search(
        r"the (monday|tuesday|wednesday|thursday|friday|saturday|sunday) before\s+(.+)",
        text,
        re.IGNORECASE,
    )
    if match:
        anchor = _parse_anchor_date(match.group(2))
        if anchor:
            days_back = (anchor.weekday() - _WEEKDAYS[match.group(1).lower()]) % 7 or 7
            target = anchor - timedelta(days=days_back)
            return f"{target:%Y-%m-%d} (the {match.group(1).title()} before {anchor:%Y-%m-%d})"

    match = re.search(r"the weekend before\s+(.+)", text, re.IGNORECASE)
    if match:
        anchor = _parse_anchor_date(match.group(1))
        if anchor:
            saturday = anchor - timedelta(days=(anchor.weekday() - 5) % 7 or 7)
            return f"{saturday:%Y-%m-%d} to {saturday + timedelta(days=1):%Y-%m-%d} (weekend before {anchor:%Y-%m-%d})"

    match = re.search(r"(\w+) weekends? before\s+(.+)", text, re.IGNORECASE)
    if match:
        count = _NUMBER_WORDS.get(match.group(1).lower())
        anchor = _parse_anchor_date(match.group(2))
        if count and anchor:
            saturday = (
                anchor
                - timedelta(days=(anchor.weekday() - 5) % 7 or 7)
                - timedelta(weeks=count - 1)
            )
            return f"{saturday:%Y-%m-%d} to {saturday + timedelta(days=1):%Y-%m-%d} ({count} weekend(s) before {anchor:%Y-%m-%d})"

    match = re.search(r"few days? before\s+(.+)", text, re.IGNORECASE)
    if match:
        anchor = _parse_anchor_date(match.group(1))
        if anchor:
            return f"approximately {anchor - timedelta(days=7):%Y-%m-%d} to {anchor - timedelta(days=1):%Y-%m-%d}"
    return None


def _response_text(response: object) -> str:
    return str(response.choices[0].message.content or "")


@dataclass(frozen=True)
class JudgeEngine:
    """Benchmark-aware prompt, retry, and voting policy."""

    llm: object
    benchmark: str
    max_attempts: int = 8

    def _messages(
        self,
        *,
        question: str,
        gold: str,
        generated: str,
        category: str | None,
        is_abstention: bool,
    ) -> list[dict[str, str]]:
        if self.benchmark == "locomo":
            hint = normalize_temporal_gold(gold)
            judge_gold = f"{gold}\n[Normalized: {hint}]" if hint else gold
            return build_judge_standard_messages(question=question, gold=judge_gold, gen=generated)
        if self.benchmark == "longmem":
            return build_judge_messages(
                question=question,
                gold=gold,
                generated=generated,
                category=category,
                is_abstention=is_abstention,
            )
        raise ValueError(f"Unsupported benchmark: {self.benchmark}")

    def _parse(self, text: str) -> int:
        if self.benchmark == "longmem":
            return parse_longmem_verdict(text)
        verdict = parse_locomo_verdict(text)
        return int(verdict) if verdict is not None else 0

    def _one(self, messages: list[dict[str, str]], temperature: float) -> int:
        delay = 2.0
        for attempt in range(self.max_attempts):
            try:
                response = self.llm.chat(messages=messages, temperature=temperature, max_tokens=256)
                return self._parse(_response_text(response))
            except HTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                transient = status in (429, 500, 502, 503, 504)
                if not transient or attempt == self.max_attempts - 1:
                    raise
            except RequestException:
                if attempt == self.max_attempts - 1:
                    raise
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        return 0

    def judge(
        self,
        *,
        question: str,
        gold: str,
        generated: str,
        category: str | None = None,
        is_abstention: bool = False,
        votes: int = 1,
    ) -> int:
        if votes < 1:
            raise ValueError("votes must be at least 1")
        messages = self._messages(
            question=question,
            gold=gold,
            generated=generated,
            category=category,
            is_abstention=is_abstention,
        )
        if is_abstention or votes == 1:
            return self._one(messages, 0.0)
        tally = sum(
            self._one(messages, VOTE_TEMPERATURES[index % len(VOTE_TEMPERATURES)])
            for index in range(votes)
        )
        return int(tally * 2 >= votes)

    def judge_with_carry(
        self,
        *,
        question: str,
        gold: str,
        generated: str,
        category: str | None = None,
        is_abstention: bool = False,
        votes: int = 3,
        first_verdict: int | None = None,
    ) -> tuple[int, int]:
        """Return ``(first_vote, final_vote)`` using the published carry rule."""
        first = first_verdict
        if first not in (0, 1):
            first = self.judge(
                question=question,
                gold=gold,
                generated=generated,
                category=category,
                is_abstention=is_abstention,
            )
        if first == 1 or is_abstention or votes == 1:
            return first, first
        final = self.judge(
            question=question,
            gold=gold,
            generated=generated,
            category=category,
            is_abstention=False,
            votes=votes,
        )
        return first, final


def openai_api_key() -> str | None:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    env_path = _ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            match = re.search(r'OPENAI_API_KEY="?(sk-[^"\s]+)', line)
            if match:
                return match.group(1)
    return None


def find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    columns = {column.lower().lstrip("\ufeff"): column for column in frame.columns}
    return next((columns[name.lower()] for name in candidates if name.lower() in columns), None)


def _sample_ids(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return result


def _client_factory(args: argparse.Namespace) -> Callable[[], LLMClient]:
    local = threading.local()

    def factory() -> LLMClient:
        if not hasattr(local, "client"):
            api_key = openai_api_key() if args.judge_base_url.rstrip("/").endswith("openai.com/v1") else None
            local.client = LLMClient(
                base_url=args.judge_base_url,
                model_name=args.judge_model,
                api_key=api_key,
            )
        return local.client

    return factory


def _locomo_paths(run_dir: Path, sample_id: int) -> tuple[Path, Path] | None:
    sample_dir = run_dir / f"sample_{sample_id}"
    existing = sorted(sample_dir.glob("*_judge_4omini.csv"))
    if existing:
        return existing[0], existing[0]
    raw = sorted(path for path in sample_dir.glob("*_eval_*.csv") if "_judge" not in path.name)
    legacy = sorted(path for path in sample_dir.glob("*_judge.csv") if "_judge_4omini" not in path.name)
    source = raw[0] if raw else (legacy[0] if legacy else None)
    if source is None:
        return None
    if source.stem.endswith("_judge"):
        output = source.with_name(source.name.replace("_judge.csv", "_judge_4omini.csv"))
    else:
        output = source.with_name(f"{source.stem}_judge_4omini.csv")
    return source, output


def _judge_locomo_file(
    source: Path,
    output: Path,
    *,
    client_factory: Callable[[], LLMClient],
    votes: int,
    workers: int,
    dry_run: bool,
) -> tuple[int, int, int]:
    frame = pd.read_csv(output if output.exists() else source, encoding="utf-8-sig")
    question_col = find_column(frame, ["question"])
    gold_col = find_column(frame, ["gold_answer", "answer", "gold"])
    generated_col = find_column(frame, ["model_answer", "generated_answer", "Generated_Answer"])
    if not all((question_col, gold_col, generated_col)):
        raise ValueError(f"{source}: missing question, gold, or generated-answer column")

    target_col = SINGLE_VOTE_COLUMN if votes == 1 else MAJORITY_VOTE_COLUMN
    for column in (SINGLE_VOTE_COLUMN, target_col):
        if column not in frame.columns:
            frame[column] = ""

    jobs: list[tuple[int, str, str, str, int | None]] = []
    skipped = 0
    for index, row in frame.iterrows():
        if as_binary(row.get(target_col)) is not None:
            skipped += 1
            continue
        question = str(row[question_col]).strip()
        generated = str(row[generated_col]).strip()
        if not question or not generated:
            continue
        jobs.append(
            (
                index,
                question,
                str(row[gold_col]).strip(),
                generated,
                as_binary(row.get(SINGLE_VOTE_COLUMN)),
            )
        )

    if dry_run:
        return len(jobs), skipped, 0

    def judge_row(job: tuple[int, str, str, str, int | None]) -> tuple[int, int, int]:
        index, question, gold, generated, first = job
        engine = JudgeEngine(client_factory(), "locomo")
        first, final = engine.judge_with_carry(
            question=question,
            gold=gold,
            generated=generated,
            votes=votes,
            first_verdict=first,
        )
        return index, first, final

    judged = carried = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, first, final in pool.map(judge_row, jobs):
            frame.at[index, SINGLE_VOTE_COLUMN] = first
            frame.at[index, target_col] = final
            judged += 1
            carried += int(votes > 1 and first == 1)
            if judged % 100 == 0:
                frame.to_csv(output, index=False, encoding="utf-8-sig")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    return judged, skipped, carried


def _score_locomo(paths: Iterable[Path], column: str, include_adversarial: bool) -> dict:
    total = correct = 0
    by_category: dict[str, list[int]] = {}
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        label_col = find_column(frame, ["category_label"])
        for _, row in frame.iterrows():
            label = str(row.get(label_col, "")).strip() if label_col else ""
            if not include_adversarial and label.lower() == "adversarial":
                continue
            verdict = as_binary(row.get(column))
            if verdict is None:
                continue
            correct += verdict
            total += 1
            bucket = by_category.setdefault(label, [0, 0])
            bucket[0] += verdict
            bucket[1] += 1
    return {
        "column": column,
        "correct": correct,
        "total": total,
        "accuracy_percent": round(100 * correct / total, 2) if total else None,
        "by_category": {
            label: {
                "correct": values[0],
                "total": values[1],
                "accuracy_percent": round(100 * values[0] / values[1], 2),
            }
            for label, values in sorted(by_category.items())
        },
    }


def run_locomo(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    client_factory = _client_factory(args)
    sample_ids = _sample_ids(args.samples)
    target_col = SINGLE_VOTE_COLUMN if args.votes == 1 else MAJORITY_VOTE_COLUMN
    for run_tag in args.run_tags:
        run_dir = output_root / run_tag
        output_paths: list[Path] = []
        for sample_id in sample_ids:
            paths = _locomo_paths(run_dir, sample_id)
            if paths is None:
                print(f"[MISS] {run_tag}/sample_{sample_id}: no eval CSV")
                continue
            source, output = paths
            judged, skipped, carried = _judge_locomo_file(
                source,
                output,
                client_factory=client_factory,
                votes=args.votes,
                workers=args.workers,
                dry_run=args.dry_run,
            )
            output_paths.append(output)
            print(
                f"[{run_tag}/sample_{sample_id}] judged={judged} "
                f"skipped={skipped} carried={carried}"
            )
        if args.dry_run:
            continue
        stats = _score_locomo(output_paths, target_col, args.include_adversarial)
        aggregate = run_dir / f"_correctness_aggregate_{target_col}.json"
        aggregate.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"[{run_tag}] {target_col}={stats['accuracy_percent']}% ({stats['correct']}/{stats['total']})")
    return 0


def _judge_longmem_file(
    path: Path,
    *,
    category: str,
    client_factory: Callable[[], LLMClient],
    votes: int,
    column: str | None,
    dry_run: bool,
) -> tuple[int, int, str]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    question_col = find_column(frame, ["question"])
    gold_col = find_column(frame, ["answer", "gold_answer"])
    generated_col = find_column(frame, ["Generated_Answer", "generated_answer", "model_answer"])
    if not all((question_col, gold_col, generated_col)):
        return 0, 0, "missing-columns"

    is_abstention = path.stem.endswith("_abs")
    target_col = column or (
        ABSTENTION_COLUMN
        if is_abstention
        else (SINGLE_VOTE_COLUMN if votes == 1 else MAJORITY_VOTE_COLUMN)
    )
    if target_col not in frame.columns:
        frame[target_col] = ""

    judged = skipped = 0
    engine = None if dry_run else JudgeEngine(client_factory(), "longmem")
    for index, row in frame.iterrows():
        if as_binary(row.get(target_col)) is not None:
            skipped += 1
            continue
        question = str(row[question_col]).strip()
        generated = str(row[generated_col]).strip()
        if not question or not generated:
            continue
        if dry_run:
            judged += 1
            continue
        first = None if is_abstention else as_binary(row.get(SINGLE_VOTE_COLUMN))
        _, final = engine.judge_with_carry(
            question=question,
            gold=str(row[gold_col]).strip(),
            generated=generated,
            category=category,
            is_abstention=is_abstention,
            votes=votes,
            first_verdict=first,
        )
        frame.at[index, target_col] = final
        judged += 1
    if not dry_run and judged:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    return judged, skipped, target_col


def _score_longmem(
    paths: Iterable[Path],
    *,
    votes: int,
    column: str | None,
) -> dict:
    total = correct = 0
    by_category: dict[str, list[int]] = {}
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        is_abstention = path.stem.endswith("_abs")
        target_col = column or (
            ABSTENTION_COLUMN
            if is_abstention
            else (SINGLE_VOTE_COLUMN if votes == 1 else MAJORITY_VOTE_COLUMN)
        )
        if frame.empty or target_col not in frame.columns:
            continue
        verdict = as_binary(frame.iloc[0].get(target_col))
        if verdict is None:
            continue
        label = path.parent.name
        correct += verdict
        total += 1
        bucket = by_category.setdefault(label, [0, 0])
        bucket[0] += verdict
        bucket[1] += 1
    return {
        "protocol": "longmem-final" if column is None and votes > 1 else "custom",
        "columns": {
            "general": column or (SINGLE_VOTE_COLUMN if votes == 1 else MAJORITY_VOTE_COLUMN),
            "abstention": column or ABSTENTION_COLUMN,
        },
        "correct": correct,
        "total": total,
        "accuracy_percent": round(100 * correct / total, 2) if total else None,
        "by_category": {
            label: {
                "correct": values[0],
                "total": values[1],
                "accuracy_percent": round(100 * values[0] / values[1], 2),
            }
            for label, values in sorted(by_category.items())
        },
    }


def run_longmem(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    client_factory = _client_factory(args)
    total_judged = total_skipped = 0
    for run_tag in args.run_tags:
        run_dir = output_root / run_tag
        if not run_dir.exists():
            print(f"[MISS] {run_tag}: {run_dir} not found")
            continue
        jobs: list[tuple[Path, str]] = []
        for directory, category in LONGMEM_CATEGORIES.items():
            category_dir = run_dir / directory
            jobs.extend(
                (path, category)
                for path in sorted(category_dir.glob("*.csv"))
                if path.name not in SKIP_LONGMEM_FILES
            )

        def judge_file(job: tuple[Path, str]) -> tuple[Path, int, int, str]:
            path, category = job
            judged, skipped, target = _judge_longmem_file(
                path,
                category=category,
                client_factory=client_factory,
                votes=args.votes,
                column=args.column,
                dry_run=args.dry_run,
            )
            return path, judged, skipped, target

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(judge_file, job) for job in jobs]
            for future in as_completed(futures):
                path, judged, skipped, target = future.result()
                total_judged += judged
                total_skipped += skipped
                if judged:
                    print(f"[{run_tag}/{path.parent.name}/{path.name}] judged={judged} target={target}")
        if not args.dry_run:
            stats = _score_longmem(
                (path for path, _ in jobs),
                votes=args.votes,
                column=args.column,
            )
            aggregate = run_dir / "_correctness_aggregate_judge.json"
            aggregate.write_text(json.dumps(stats, indent=2) + "\n")
            print(
                f"[{run_tag}] final={stats['accuracy_percent']}% "
                f"({stats['correct']}/{stats['total']})"
            )
    print(f"Done. judged={total_judged} skipped={total_skipped}")
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser, *, output_root: Path) -> None:
    parser.add_argument("run_tags", nargs="+", help="Run directories under the benchmark output root")
    parser.add_argument("--output-root", default=str(output_root))
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--votes", type=int, default=3, help="Votes used to rejudge a failed first pass")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="benchmark", required=True)

    locomo = subparsers.add_parser("locomo", help="Judge LoCoMo sample eval CSVs")
    _add_common_arguments(locomo, output_root=LOCOMO_OUTPUT)
    locomo.add_argument("--samples", default="0-9")
    locomo.add_argument("--include-adversarial", action="store_true")
    locomo.set_defaults(handler=run_locomo)

    longmem = subparsers.add_parser("longmem", help="Judge LongMemEval per-question CSVs")
    _add_common_arguments(longmem, output_root=LONGMEM_OUTPUT)
    longmem.add_argument("--column", default=None, help="Override the protocol output column")
    longmem.set_defaults(handler=run_longmem)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.votes < 1:
        raise SystemExit("--votes must be at least 1")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
