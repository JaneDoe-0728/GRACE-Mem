#!/usr/bin/env python3
"""Compare two LoCoMo sample runs and classify correctness flips."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

QUESTION_KEY = "\ufeffquestion"


@dataclass(frozen=True)
class QuestionRunData:
    """One question's outcome in one run, for cross-run comparison."""
    question: str
    correctness: str
    model_answer: str
    retrieved_context: str
    low_level_keywords: tuple[str, ...]
    high_level_keywords: tuple[str, ...]


@dataclass(frozen=True)
class FlipRecord:
    """A question whose verdict changed between two runs.

    Flips are the signal a raw accuracy delta hides: two runs can score
    identically while disagreeing on a third of the questions, which means the
    change moved behaviour without improving it. Direction is kept -- gained
    versus lost -- since a net-zero delta made of equal flips both ways is a very
    different result from no change at all.
    """
    question: str
    baseline_correctness: str
    candidate_correctness: str
    keyword_changed: bool
    retrieved_context_changed: bool
    final_answer_changed: bool
    baseline_low_keywords: tuple[str, ...]
    candidate_low_keywords: tuple[str, ...]
    baseline_high_keywords: tuple[str, ...]
    candidate_high_keywords: tuple[str, ...]
    baseline_answer: str
    candidate_answer: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two LoCoMo sample directories and list only the questions "
            "whose correctness label flipped, including whether the change "
            "showed up in keyword extraction, retrieved context, or final answer."
        )
    )
    parser.add_argument("baseline_sample_dir", type=Path, help="Path to the baseline sample_<n> directory")
    parser.add_argument("candidate_sample_dir", type=Path, help="Path to the candidate sample_<n> directory")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text output",
    )
    return parser.parse_args()


def _find_single_judge_csv(sample_dir: Path) -> Path:
    """Locate the one judge CSV in a run directory, or fail.

    Ambiguity raises rather than picking the first match: silently comparing
    against the wrong file produces a flip report that looks valid and is not.
    """
    matches = sorted(sample_dir.glob("*_judge.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_judge.csv under {sample_dir}, found {len(matches)}"
        )
    return matches[0]


def _load_judge_rows(sample_dir: Path) -> list[dict[str, str]]:
    judge_csv = _find_single_judge_csv(sample_dir)
    with judge_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_keyword_map(sample_dir: Path) -> dict[str, dict[str, tuple[str, ...]]]:
    """Load question -> extracted keywords, for annotating flips.

    Keywords are the cheapest explanation of a flip: a question whose verdict
    changed and whose keywords also changed flipped because retrieval was
    asked something different, not because the system got better at it.
    """
    log_path = sample_dir / "logs" / "kg_retriever.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"Retriever log not found: {log_path}")

    keyword_map: dict[str, dict[str, tuple[str, ...]]] = {}
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("event") != "assemble_context_from_query_start":
                continue
            question = str(payload.get("question") or "").strip()
            if not question:
                continue
            keyword_map[question] = {
                "low_level_keywords": tuple(payload.get("low_level_keywords") or ()),
                "high_level_keywords": tuple(payload.get("high_level_keywords") or ()),
            }
    return keyword_map


def _rows_by_question(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Index a run's judged rows by question text.

    Question text is the only key shared across runs -- row order and ids are
    run-local -- so it is what two runs are joined on.
    """
    by_question: dict[str, dict[str, str]] = {}
    for row in rows:
        question = (row.get(QUESTION_KEY) or row.get("question") or "").strip()
        if not question:
            continue
        if question in by_question:
            raise ValueError(f"Duplicate question in judge CSV: {question}")
        by_question[question] = row
    return by_question


def _build_question_run_data(sample_dir: Path) -> dict[str, QuestionRunData]:
    """Assemble one question's per-run outcomes for comparison."""
    rows = _rows_by_question(_load_judge_rows(sample_dir))
    keywords = _load_keyword_map(sample_dir)
    result: dict[str, QuestionRunData] = {}
    for question, row in rows.items():
        keyword_info = keywords.get(question, {})
        result[question] = QuestionRunData(
            question=question,
            correctness=(row.get("correctness") or "").strip(),
            model_answer=(row.get("model_answer") or "").strip(),
            retrieved_context=(row.get("retrieved_context") or "").strip(),
            low_level_keywords=tuple(keyword_info.get("low_level_keywords", ())),
            high_level_keywords=tuple(keyword_info.get("high_level_keywords", ())),
        )
    return result


def _compute_flips(
    baseline: dict[str, QuestionRunData],
    candidate: dict[str, QuestionRunData],
) -> list[FlipRecord]:
    """Find questions whose verdict changed between two runs.

    The measurement a raw accuracy delta hides: two runs can score identically
    while disagreeing on a third of their questions, which means the change
    moved behaviour without improving it. Direction is recorded, since a
    net-zero delta made of equal flips both ways is a very different result from
    no change at all.

    Questions judged in only one run are excluded -- an unjudged question is
    missing data, and counting it as a flip would manufacture one.
    """
    baseline_questions = set(baseline)
    candidate_questions = set(candidate)
    if baseline_questions != candidate_questions:
        missing_baseline = sorted(candidate_questions - baseline_questions)
        missing_candidate = sorted(baseline_questions - candidate_questions)
        raise ValueError(
            "Question sets do not match between runs. "
            f"Missing in baseline: {missing_baseline[:3]} "
            f"Missing in candidate: {missing_candidate[:3]}"
        )

    flips: list[FlipRecord] = []
    for question in sorted(baseline):
        left = baseline[question]
        right = candidate[question]
        if left.correctness == right.correctness:
            continue
        flips.append(
            FlipRecord(
                question=question,
                baseline_correctness=left.correctness,
                candidate_correctness=right.correctness,
                keyword_changed=(
                    left.low_level_keywords != right.low_level_keywords
                    or left.high_level_keywords != right.high_level_keywords
                ),
                retrieved_context_changed=left.retrieved_context != right.retrieved_context,
                final_answer_changed=left.model_answer != right.model_answer,
                baseline_low_keywords=left.low_level_keywords,
                candidate_low_keywords=right.low_level_keywords,
                baseline_high_keywords=left.high_level_keywords,
                candidate_high_keywords=right.high_level_keywords,
                baseline_answer=left.model_answer,
                candidate_answer=right.model_answer,
            )
        )
    return flips


def _summarize_flip(flip: FlipRecord) -> dict[str, Any]:
    """Describe one flip with the context needed to explain it.

    Carries both runs' answers and retrieved evidence, since a flip is only
    interpretable next to what changed underneath it.
    """
    return {
        "question": flip.question,
        "correctness": {
            "baseline": flip.baseline_correctness,
            "candidate": flip.candidate_correctness,
        },
        "changed_at": {
            "keyword_extraction": flip.keyword_changed,
            "retrieved_context": flip.retrieved_context_changed,
            "final_answer": flip.final_answer_changed,
        },
        "keywords": {
            "baseline": {
                "low_level": list(flip.baseline_low_keywords),
                "high_level": list(flip.baseline_high_keywords),
            },
            "candidate": {
                "low_level": list(flip.candidate_low_keywords),
                "high_level": list(flip.candidate_high_keywords),
            },
        },
        "answers": {
            "baseline": flip.baseline_answer,
            "candidate": flip.candidate_answer,
        },
    }


def _print_text_report(
    flips: list[FlipRecord],
    baseline_sample_dir: Path,
    candidate_sample_dir: Path,
) -> None:
    print(f"baseline:  {baseline_sample_dir}")
    print(f"candidate: {candidate_sample_dir}")
    print(f"correctness flips: {len(flips)}")
    for index, flip in enumerate(flips, start=1):
        changed_at = []
        if flip.keyword_changed:
            changed_at.append("keyword_extraction")
        if flip.retrieved_context_changed:
            changed_at.append("retrieved_context")
        if flip.final_answer_changed:
            changed_at.append("final_answer")
        print()
        print(f"{index}. {flip.question}")
        print(
            "   correctness: "
            f"{flip.baseline_correctness} -> {flip.candidate_correctness}"
        )
        print(f"   changed_at: {', '.join(changed_at) if changed_at else 'none'}")
        if flip.keyword_changed:
            if flip.baseline_low_keywords != flip.candidate_low_keywords:
                print(
                    "   low_level_keywords: "
                    f"{list(flip.baseline_low_keywords)} -> {list(flip.candidate_low_keywords)}"
                )
            if flip.baseline_high_keywords != flip.candidate_high_keywords:
                print(
                    "   high_level_keywords: "
                    f"{list(flip.baseline_high_keywords)} -> {list(flip.candidate_high_keywords)}"
                )
        if flip.final_answer_changed:
            print(f"   baseline_answer: {flip.baseline_answer}")
            print(f"   candidate_answer: {flip.candidate_answer}")


def main() -> None:
    args = _parse_args()
    baseline = _build_question_run_data(args.baseline_sample_dir)
    candidate = _build_question_run_data(args.candidate_sample_dir)
    flips = _compute_flips(baseline, candidate)

    if args.json:
        payload = {
            "baseline_sample_dir": str(args.baseline_sample_dir),
            "candidate_sample_dir": str(args.candidate_sample_dir),
            "flip_count": len(flips),
            "flips": [_summarize_flip(flip) for flip in flips],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    _print_text_report(flips, args.baseline_sample_dir, args.candidate_sample_dir)


if __name__ == "__main__":
    main()
