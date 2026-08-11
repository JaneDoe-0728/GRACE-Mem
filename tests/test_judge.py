from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from experiment.common.evaluation.judge import (
    MAJORITY_VOTE_COLUMN,
    SINGLE_VOTE_COLUMN,
    JudgeEngine,
    _judge_locomo_file,
    _score_longmem,
    normalize_temporal_gold,
    parse_locomo_verdict,
    parse_longmem_verdict,
)


class FakeLLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = iter(replies)
        self.temperatures: list[float] = []

    def chat(self, *, messages, temperature, max_tokens):
        self.temperatures.append(temperature)
        content = next(self.replies)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_judge_with_carry_skips_extra_votes_after_correct_first_pass() -> None:
    llm = FakeLLM(['{"label":"correct"}'])

    first, final = JudgeEngine(llm, "locomo").judge_with_carry(
        question="q",
        gold="g",
        generated="a",
        votes=3,
    )

    assert (first, final) == (1, 1)
    assert llm.temperatures == [0.0]


def test_judge_with_carry_rejudges_failed_first_pass_with_three_votes() -> None:
    llm = FakeLLM(
        [
            '{"label":"wrong"}',
            '{"label":"correct"}',
            '{"label":"wrong"}',
            '{"label":"correct"}',
        ]
    )

    first, final = JudgeEngine(llm, "locomo").judge_with_carry(
        question="q",
        gold="g",
        generated="a",
        votes=3,
    )

    assert (first, final) == (0, 1)
    assert llm.temperatures == [0.0, 0.0, 0.3, 0.6]


def test_longmem_abstention_always_uses_one_vote() -> None:
    llm = FakeLLM(['{"correct":true}'])

    first, final = JudgeEngine(llm, "longmem").judge_with_carry(
        question="q",
        gold="The information provided is not enough.",
        generated="I do not have that information.",
        category="single-session-user",
        is_abstention=True,
        votes=3,
    )

    assert (first, final) == (1, 1)
    assert llm.temperatures == [0.0]


def test_verdict_parsers_accept_expected_protocol_shapes() -> None:
    assert parse_locomo_verdict('{"label":"partial"}') == 0.5
    assert parse_locomo_verdict("WRONG") == 0.0
    assert parse_longmem_verdict('{"reasoning":"ok","correct":true}') == 1
    assert parse_longmem_verdict("No") == 0


def test_temporal_normalization_preserves_existing_protocol() -> None:
    assert normalize_temporal_gold("The week before 8 May 2023") == (
        "2023-05-01 to 2023-05-07 (the 7 days before 2023-05-08)"
    )


def test_locomo_csv_resume_carries_existing_correct_first_vote(tmp_path) -> None:
    source = tmp_path / "sample0_eval_run.csv"
    output = tmp_path / "sample0_eval_run_judge_4omini.csv"
    pd.DataFrame(
        [
            {
                "question": "q1",
                "gold_answer": "g1",
                "model_answer": "a1",
                SINGLE_VOTE_COLUMN: 1,
            },
            {
                "question": "q2",
                "gold_answer": "g2",
                "model_answer": "a2",
                SINGLE_VOTE_COLUMN: 0,
            },
        ]
    ).to_csv(source, index=False)
    llm = FakeLLM(
        [
            '{"label":"correct"}',
            '{"label":"wrong"}',
            '{"label":"correct"}',
        ]
    )

    judged, skipped, carried = _judge_locomo_file(
        source,
        output,
        client_factory=lambda: llm,
        votes=3,
        workers=1,
        dry_run=False,
    )

    result = pd.read_csv(output)
    assert (judged, skipped, carried) == (2, 0, 1)
    assert result[MAJORITY_VOTE_COLUMN].tolist() == [1, 1]
    assert llm.temperatures == [0.0, 0.3, 0.6]


def test_longmem_score_combines_general_and_abstention_columns(tmp_path) -> None:
    category = tmp_path / "temporal_reasoning"
    category.mkdir()
    general = category / "general.csv"
    abstention = category / "question_abs.csv"
    pd.DataFrame([{MAJORITY_VOTE_COLUMN: 1, "correctness_absrubric": ""}]).to_csv(
        general, index=False
    )
    pd.DataFrame([{MAJORITY_VOTE_COLUMN: 0, "correctness_absrubric": 1}]).to_csv(
        abstention, index=False
    )

    stats = _score_longmem([general, abstention], votes=3, column=None)

    assert stats["protocol"] == "longmem-final"
    assert stats["correct"] == 2
    assert stats["total"] == 2
    assert stats["accuracy_percent"] == 100.0
