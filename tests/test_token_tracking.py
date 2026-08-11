import json

from KG.llm import token_tracker as public_token_tracker
from KG.llm.client import token_tracker as client_token_tracker
from KG.llm.token_tracking import TokenTracker


def test_public_llm_exports_share_the_same_tracker():
    assert public_token_tracker is client_token_tracker


def test_token_tracker_preserves_context_logs_and_summary(tmp_path):
    global_log = tmp_path / "global.jsonl"
    dataset_log = tmp_path / "dataset.jsonl"
    tracker = TokenTracker(log_path=global_log)
    tracker.set_context(dataset="sample-1", stage="qa", log_path=dataset_log)

    tracker.record("chat", prompt_tokens=7, completion_tokens=3, elapsed=2.0)

    expected = {
        "dataset": "sample-1",
        "stage": "qa",
        "label": "qa_answer",
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
        "elapsed_s": 2.0,
        "tok_per_s": 5.0,
    }
    for path in (global_log, dataset_log):
        record = json.loads(path.read_text(encoding="utf-8"))
        assert {key: record[key] for key in expected} == expected

    summary = tracker.summary()
    assert "sample-1" in summary
    assert "qa_answer" in summary
    assert "10" in summary
