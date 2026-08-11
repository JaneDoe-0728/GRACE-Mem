from __future__ import annotations

from experiment.agent_filter.corpus import Corpus, Turn
from experiment.common.evaluation.oracle import build_locomo_context, expand_longmem_sids


def test_expand_longmem_sids_stays_within_session() -> None:
    corpus = Corpus(
        [
            Turn("s1:1:u", "s1", 0, 0, "user", "", "one"),
            Turn("s1:1:a", "s1", 1, 1, "assistant", "", "two"),
            Turn("s1:2:u", "s1", 2, 2, "user", "", "three"),
            Turn("s2:1:u", "s2", 0, 0, "user", "", "other"),
        ]
    )

    result = expand_longmem_sids(corpus, ["s1:1:a"], window=1)

    assert result == ["s1:1:u", "s1:1:a", "s1:2:u"]


def test_build_locomo_context_expands_window_and_controls_photo() -> None:
    sample = {
        "conversation": {
            "session_1_date_time": "2023-01-01",
            "session_1": [
                {"dia_id": "D1:1", "speaker": "A", "text": "one"},
                {
                    "dia_id": "D1:2",
                    "speaker": "B",
                    "text": "two",
                    "blip_caption": "a photo",
                },
                {"dia_id": "D1:3", "speaker": "A", "text": "three"},
            ],
            "session_2": [
                {"dia_id": "D2:1", "speaker": "A", "text": "other"},
            ],
        }
    }

    context, sids = build_locomo_context(
        sample,
        ["D1:2"],
        window=1,
        include_photo=True,
    )

    assert sids == ["D1:1", "D1:2", "D1:3"]
    assert "D2:1" not in context
    assert "[Image: a photo]" in context
