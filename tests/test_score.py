from __future__ import annotations

import pandas as pd

from experiment.common.evaluation.score import score_run


def test_score_longmem_uses_final_protocol_columns(tmp_path) -> None:
    category = tmp_path / "temporal_reasoning"
    category.mkdir()
    pd.DataFrame(
        [{
            "question": "q1",
            "answer": "gold",
            "Generated_Answer": "gold",
            "correctness_3vote": 1,
            "correctness_absrubric": "",
        }]
    ).to_csv(category / "general.csv", index=False)
    pd.DataFrame(
        [{
            "question": "q2",
            "answer": "unknown",
            "Generated_Answer": "unknown",
            "correctness_3vote": 0,
            "correctness_absrubric": 1,
        }]
    ).to_csv(category / "missing_abs.csv", index=False)

    result = score_run(tmp_path, "longmem")

    assert result.correct == 2
    assert result.total == 2
    assert result.accuracy_percent == 100.0
    assert result.by_category["temporal_reasoning"].total == 2


def test_score_locomo_excludes_adversarial_by_default(tmp_path) -> None:
    sample_dir = tmp_path / "sample_0"
    sample_dir.mkdir()
    pd.DataFrame(
        [
            {
                "question": "q1",
                "gold_answer": "gold",
                "model_answer": "gold",
                "category_label": "Single-hop",
                "correctness_3vote": 1,
            },
            {
                "question": "q2",
                "gold_answer": "gold",
                "model_answer": "wrong",
                "category_label": "Adversarial",
                "correctness_3vote": 0,
            },
        ]
    ).to_csv(sample_dir / "sample0_eval_run_judge_4omini.csv", index=False)

    result = score_run(tmp_path, "locomo")

    assert result.correct == 1
    assert result.total == 1
    assert set(result.by_category) == {"Single-hop"}


def test_score_supports_custom_correctness_column(tmp_path) -> None:
    category = tmp_path / "multi_session"
    category.mkdir()
    pd.DataFrame(
        [{
            "question": "q",
            "answer": "gold",
            "Generated_Answer": "wrong",
            "correctness_custom": 0,
        }]
    ).to_csv(category / "question.csv", index=False)

    result = score_run(tmp_path, "longmem", column="correctness_custom")

    assert result.protocol == "correctness_custom"
    assert result.accuracy_percent == 0.0
