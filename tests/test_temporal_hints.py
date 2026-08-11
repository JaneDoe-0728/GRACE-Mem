from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from KG.utils.temporal import build_time_context, extract_temporal_hints, format_temporal_hints_for_prompt


def _ctx(ref: datetime | None = None):
    return build_time_context(
        reference_dt=ref or datetime(2023, 4, 12, 12, 0, 0),
        reference_time_str="2023/04/12 12:00",
        source="tests",
    )


def test_last_weekend_hint_uses_natural_display_and_normalized_range():
    hints = extract_temporal_hints(["We visited last weekend"], _ctx())
    h = hints[0]
    assert h["resolved_to"] == "2023-04-08 to 2023-04-09"
    assert h["normalized_start"] == "2023-04-08"
    assert h["normalized_end"] == "2023-04-09"
    assert h["granularity"] == "weekend"


def test_last_year_hint_uses_year_display():
    hints = extract_temporal_hints(["It happened last year"], _ctx())
    assert hints[0]["resolved_to"] == "2022"
    assert hints[0]["normalized_start"] == "2022-01-01"
    assert hints[0]["normalized_end"] == "2022-12-31"


def test_yesterday_hint_stays_exact_date():
    hints = extract_temporal_hints(["I went yesterday"], _ctx())
    assert hints[0]["resolved_to"] == "2023-04-11"
    assert hints[0]["normalized_start"] == "2023-04-11"
    assert hints[0]["normalized_end"] == "2023-04-11"


def test_format_temporal_hints_for_prompt_uses_display_value():
    result = format_temporal_hints_for_prompt([
        {"original": "last year", "resolved_to": "2022"},
        {"original": "last weekend", "resolved_to": "2023-04-08 to 2023-04-09"},
    ])
    assert '"last year" → 2022' in result
    assert '"last weekend" → 2023-04-08 to 2023-04-09' in result


def test_summary_rewrite_keeps_direct_replacement_without_anchor_block():
    from KG.pipeline.ingest_steps.compress import Compressor

    mock_vdb = MagicMock()
    mock_vdb.add_summary.return_value = "summary-id-001"

    compressor = Compressor(summaries_vdb=mock_vdb)
    with patch.object(compressor, "_get_compressor") as mock_get:
        mock_llmlingua = MagicMock()
        mock_llmlingua.compress_prompt_llmlingua2.return_value = {
            "compressed_prompt": "Yesterday took kids to museum",
            "rate": 0.8,
        }
        mock_get.return_value = mock_llmlingua

        _, summary_text = compressor.summarize_turn(
            session_id=1,
            message_id=1,
            user_text="Yesterday took kids to museum",
            assistant_text="",
            request_id="test-req",
            dialogue_datetime="2023/07/06 10:00",
            temporal_hints=[{"original": "Yesterday", "resolved_to": "2023-07-05"}],
            tctx=build_time_context(
                reference_dt=datetime(2023, 7, 6, 10, 0, 0),
                reference_time_str="2023/07/06 10:00",
                source="tests",
            ),
        )

    assert "2023-07-05" in summary_text
    assert "Yesterday" not in summary_text
    assert "[Temporal anchors:" not in summary_text
