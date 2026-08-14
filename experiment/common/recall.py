"""Shared accumulation primitives for benchmark gold-recall reports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecallStats:
    """Accumulates the recall and accuracy figures a gold-recall report needs.

    Several denominators are tracked because they answer different questions and
    are easy to conflate. `gold_hit / gold_total` is turn-level recall -- how
    much of the evidence was found. `all_gold_hit / questions_with_gold` is the
    stricter question-level figure: how often *every* gold turn was found, which
    is what matters for a multi-hop question that needs all of them.

    `all_gold_hit_correct` is the interesting one. It counts questions that got
    perfect retrieval and were still answered wrong, which is the cleanest
    available measure of answer-generation error with retrieval held out.

    Questions with no gold annotation are excluded from the retrieval
    denominators but still counted for accuracy, so the two sets of numbers have
    different bases by design.
    """
    questions: int = 0
    correct: int = 0
    gold_total: int = 0
    gold_hit: int = 0
    questions_with_gold: int = 0
    all_gold_hit: int = 0
    all_gold_hit_correct: int = 0

    def add_accuracy(self, *, correct: bool) -> None:
        self.questions += 1
        self.correct += int(correct)

    def add_retrieval(self, *, gold: set[str], retrieved: set[str], correct: bool) -> None:
        if not gold:
            return
        hit = gold & retrieved
        self.gold_total += len(gold)
        self.gold_hit += len(hit)
        self.questions_with_gold += 1
        if hit == gold:
            self.all_gold_hit += 1
            self.all_gold_hit_correct += int(correct)


def format_ratio(numerator: int, denominator: int) -> str:
    if not denominator:
        return f"{numerator}/{denominator} = n/a"
    return f"{numerator}/{denominator} = {100 * numerator / denominator:.1f}%"


def metric_lines(stats: RecallStats, *, gold_label: str, indent: str = "") -> list[str]:
    return [
        f"{indent}overall accuracy            {format_ratio(stats.correct, stats.questions)}",
        f"{indent}{gold_label}       {format_ratio(stats.gold_hit, stats.gold_total)}",
        f"{indent}all-gold-hit rate           {format_ratio(stats.all_gold_hit, stats.questions_with_gold)}",
        f"{indent}accuracy when all gold hit  {format_ratio(stats.all_gold_hit_correct, stats.all_gold_hit)}",
    ]
