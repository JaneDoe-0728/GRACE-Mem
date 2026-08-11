"""Score LoCoMo or LongMemEval runs with one benchmark-aware CLI.

Examples:
    uv run python experiment/score.py my-locomo-run
    uv run python experiment/score.py my-longmem-run
    uv run python experiment/score.py run-r1 run-r2 run-r3 --agent
    uv run python experiment/score.py /path/to/run --column correctness_custom
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from experiment.judge import ABSTENTION_COLUMN, MAJORITY_VOTE_COLUMN, as_binary
from experiment.locomo.stages.judge import compute_f1_and_bleu1

LOCOMO_OUTPUT = _ROOT / "experiment" / "locomo" / "output" / "standard"
LONGMEM_OUTPUT = _ROOT / "experiment" / "longmem" / "output"
LOCOMO_DATA = _ROOT / "experiment" / "locomo" / "data" / "locomo10.json"
LONGMEM_CATEGORIES = (
    "single_session_user",
    "single_session_assistant",
    "single_session_preference",
    "multi_session",
    "knowledge_update",
    "temporal_reasoning",
)
LOCOMO_CATEGORY_NAMES = {
    1: "Multi-hop",
    2: "Temporal",
    3: "Open-domain",
    4: "Single-hop",
    5: "Adversarial",
}
SKIP_FILES = {"all_answers.csv", "progress.csv"}


@dataclass(frozen=True)
class ScoredItem:
    category: str
    verdict: int
    gold: str
    generated: str


@dataclass(frozen=True)
class CategoryScore:
    correct: int
    total: int
    accuracy_percent: float


@dataclass(frozen=True)
class LexicalScore:
    total: int
    f1_percent: float | None
    bleu1_percent: float | None


@dataclass(frozen=True)
class RunScore:
    run: str
    benchmark: str
    protocol: str
    correct: int
    total: int
    accuracy_percent: float | None
    by_category: dict[str, CategoryScore]
    lexical: LexicalScore
    agent: dict[str, object] | None


def resolve_run(reference: str, benchmark: str = "auto") -> tuple[Path, str]:
    direct = Path(reference).expanduser()
    candidates: list[tuple[Path, str]] = []
    if direct.is_dir():
        candidates.append((direct.resolve(), detect_benchmark(direct)))
    else:
        if benchmark in ("auto", "longmem") and (LONGMEM_OUTPUT / reference).is_dir():
            candidates.append(((LONGMEM_OUTPUT / reference).resolve(), "longmem"))
        if benchmark in ("auto", "locomo") and (LOCOMO_OUTPUT / reference).is_dir():
            candidates.append(((LOCOMO_OUTPUT / reference).resolve(), "locomo"))
    if benchmark != "auto":
        candidates = [item for item in candidates if item[1] == benchmark]
    if not candidates:
        raise FileNotFoundError(f"Run not found: {reference}")
    if len(candidates) > 1:
        raise ValueError(f"Run tag is ambiguous; pass --benchmark: {reference}")
    return candidates[0]


def detect_benchmark(run_dir: Path) -> str:
    if any((run_dir / category).is_dir() for category in LONGMEM_CATEGORIES):
        return "longmem"
    if any(run_dir.glob("sample_*")):
        return "locomo"
    raise ValueError(f"Cannot detect benchmark layout: {run_dir}")


def _read_first(path: Path) -> pd.Series | None:
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return None
    return frame.iloc[0] if not frame.empty else None


def _value(row: pd.Series, names: Sequence[str]) -> str:
    for name in names:
        if name in row.index and pd.notna(row.get(name)):
            return str(row.get(name, ""))
    return ""


def load_longmem(run_dir: Path, column: str | None) -> list[ScoredItem]:
    items: list[ScoredItem] = []
    for category in LONGMEM_CATEGORIES:
        for path in sorted((run_dir / category).glob("*.csv")):
            if path.name in SKIP_FILES:
                continue
            row = _read_first(path)
            if row is None:
                continue
            target = column or (
                ABSTENTION_COLUMN if path.stem.endswith("_abs") else MAJORITY_VOTE_COLUMN
            )
            verdict = as_binary(row.get(target))
            if verdict is None:
                continue
            items.append(
                ScoredItem(
                    category=category,
                    verdict=verdict,
                    gold=_value(row, ("answer", "gold_answer")),
                    generated=_value(row, ("Generated_Answer", "generated_answer", "model_answer")),
                )
            )
    return items


def _locomo_category_map() -> dict[tuple[int, str], str]:
    if not LOCOMO_DATA.exists():
        return {}
    data = json.loads(LOCOMO_DATA.read_text(encoding="utf-8"))
    return {
        (sample_index, str(question.get("question", "")).strip()): LOCOMO_CATEGORY_NAMES.get(
            question.get("category"), str(question.get("category", "Unknown"))
        )
        for sample_index, sample in enumerate(data)
        for question in sample.get("qa", [])
    }


def _locomo_judge_file(sample_dir: Path) -> Path | None:
    preferred = sorted(sample_dir.glob("*_judge_4omini.csv"))
    if preferred:
        return preferred[0]
    legacy = sorted(sample_dir.glob("*_judge.csv"))
    return legacy[0] if legacy else None


def load_locomo(
    run_dir: Path,
    column: str | None,
    *,
    include_adversarial: bool,
) -> list[ScoredItem]:
    category_map = _locomo_category_map()
    target = column or MAJORITY_VOTE_COLUMN
    items: list[ScoredItem] = []
    for sample_dir in sorted(run_dir.glob("sample_*")):
        try:
            sample_index = int(sample_dir.name.removeprefix("sample_"))
        except ValueError:
            continue
        judge_path = _locomo_judge_file(sample_dir)
        if judge_path is None:
            continue
        frame = pd.read_csv(judge_path, encoding="utf-8-sig")
        for _, row in frame.iterrows():
            verdict = as_binary(row.get(target))
            if verdict is None:
                continue
            question = str(row.get("question", "")).strip()
            category = str(row.get("category_label", "")).strip()
            if not category or category.lower() == "nan":
                category = category_map.get((sample_index, question), "Unknown")
            if not include_adversarial and category.lower() == "adversarial":
                continue
            items.append(
                ScoredItem(
                    category=category,
                    verdict=verdict,
                    gold=_value(row, ("gold_answer", "answer", "gold")),
                    generated=_value(row, ("model_answer", "generated_answer", "Generated_Answer")),
                )
            )
    return items


def _count_value(value: object) -> int | None:
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _agent_records(run_dir: Path, benchmark: str) -> list[dict]:
    patterns = (
        ("*/_grep_agent_traces.jsonl",)
        if benchmark == "longmem"
        else ("sample_*/_grep_traces.jsonl", "sample_*/_grep_agent_traces.jsonl")
    )
    latest: dict[tuple[str, str], dict] = {}
    for pattern in patterns:
        for path in sorted(run_dir.glob(pattern)):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(record.get("sample") or record.get("question", "")[:100])
                latest[(str(path), key)] = record
    return list(latest.values())


def score_agent(run_dir: Path, benchmark: str) -> dict[str, object] | None:
    records = _agent_records(run_dir, benchmark)
    if not records:
        return None

    def average(values: Iterable[float | int | None]) -> float | None:
        present = [float(value) for value in values if value is not None]
        return round(sum(present) / len(present), 6) if present else None

    fallback_reasons = Counter(
        str(record.get("fallback")) for record in records if record.get("fallback")
    )
    return {
        "total": len(records),
        "fallback_percent": round(
            100 * sum(bool(record.get("fallback")) for record in records) / len(records), 2
        ),
        "kept_per_question": average(_count_value(record.get("kept")) for record in records),
        "added_per_question": average(_count_value(record.get("added")) for record in records),
        "dropped_per_question": average(_count_value(record.get("dropped")) for record in records),
        "fallback_reasons": dict(fallback_reasons.most_common()),
    }


def score_run(
    run_dir: Path,
    benchmark: str,
    *,
    column: str | None = None,
    include_adversarial: bool = False,
    include_agent: bool = False,
) -> RunScore:
    items = (
        load_longmem(run_dir, column)
        if benchmark == "longmem"
        else load_locomo(run_dir, column, include_adversarial=include_adversarial)
    )
    if not items:
        target = column or "final evaluation protocol"
        raise ValueError(f"No scored rows found for {target}: {run_dir}")

    grouped: dict[str, list[int]] = defaultdict(list)
    for item in items:
        grouped[item.category].append(item.verdict)
    by_category = {
        category: CategoryScore(
            correct=sum(verdicts),
            total=len(verdicts),
            accuracy_percent=round(100 * sum(verdicts) / len(verdicts), 2),
        )
        for category, verdicts in sorted(grouped.items())
    }

    lexical_values = [
        compute_f1_and_bleu1(item.gold, item.generated)
        for item in items
        if item.gold.strip() and item.generated.strip()
    ]
    lexical = LexicalScore(
        total=len(lexical_values),
        f1_percent=(
            round(100 * sum(value[0] for value in lexical_values) / len(lexical_values), 2)
            if lexical_values
            else None
        ),
        bleu1_percent=(
            round(100 * sum(value[1] for value in lexical_values) / len(lexical_values), 2)
            if lexical_values
            else None
        ),
    )
    correct = sum(item.verdict for item in items)
    protocol = column or (
        f"{MAJORITY_VOTE_COLUMN} + {ABSTENTION_COLUMN} for *_abs"
        if benchmark == "longmem"
        else MAJORITY_VOTE_COLUMN
    )
    return RunScore(
        run=str(run_dir),
        benchmark=benchmark,
        protocol=protocol,
        correct=correct,
        total=len(items),
        accuracy_percent=round(100 * correct / len(items), 2),
        by_category=by_category,
        lexical=lexical,
        agent=score_agent(run_dir, benchmark) if include_agent else None,
    )


def _print_run(result: RunScore) -> None:
    print(f"\n=== {Path(result.run).name} ({result.benchmark}) ===")
    print(f"protocol: {result.protocol}")
    print(f"{'category':28s} {'correct':>9s} {'total':>7s} {'accuracy':>10s}")
    for category, score in result.by_category.items():
        print(
            f"{category:28s} {score.correct:9d} {score.total:7d} "
            f"{score.accuracy_percent:9.2f}%"
        )
    print("-" * 58)
    print(
        f"{'OVERALL':28s} {result.correct:9d} {result.total:7d} "
        f"{result.accuracy_percent:9.2f}%"
    )
    if result.lexical.total:
        print(
            f"F1={result.lexical.f1_percent:.2f}% "
            f"BLEU-1={result.lexical.bleu1_percent:.2f}% n={result.lexical.total}"
        )
    if result.agent:
        print(
            "agent: "
            f"fallback={result.agent['fallback_percent']:.2f}% "
            f"kept={result.agent['kept_per_question']} "
            f"added={result.agent['added_per_question']} "
            f"dropped={result.agent['dropped_per_question']} "
            f"n={result.agent['total']}"
        )


def _print_multi_run(results: Sequence[RunScore]) -> None:
    if len(results) < 2:
        return
    accuracies = [result.accuracy_percent for result in results if result.accuracy_percent is not None]
    if not accuracies:
        return
    deviation = statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0
    print(
        f"\n{len(accuracies)}-run overall: "
        f"{statistics.mean(accuracies):.2f}% +/- {deviation:.2f}pp"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--benchmark", choices=("auto", "locomo", "longmem"), default="auto")
    parser.add_argument("--column", default=None, help="Score one custom correctness column")
    parser.add_argument("--include-adversarial", action="store_true")
    parser.add_argument("--agent", action="store_true", help="Include Agent Filter trace metrics")
    parser.add_argument("--json", dest="json_path", default=None, help="Write machine-readable results")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved = [resolve_run(reference, args.benchmark) for reference in args.runs]
    benchmarks = {benchmark for _, benchmark in resolved}
    if len(benchmarks) != 1:
        raise SystemExit("All runs must belong to the same benchmark")
    results = [
        score_run(
            path,
            benchmark,
            column=args.column,
            include_adversarial=args.include_adversarial,
            include_agent=args.agent,
        )
        for path, benchmark in resolved
    ]
    for result in results:
        _print_run(result)
    _print_multi_run(results)
    if args.json_path:
        output_path = Path(args.json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([asdict(result) for result in results], indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
