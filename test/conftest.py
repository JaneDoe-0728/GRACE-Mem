"""Pytest collection policy for the offline regression suite."""

MANUAL_SCRIPT_NAMES = (
    "test_api.py",
    "test_gemini.py",
    "test_gemini_new.py",
    "test_gpt.py",
    "test_gpt_new.py",
    "test_inference_keyword.py",
    "test_reranker.py",
    "test_sa_retrieve.py",
    "test_update.py",
)

# These are live network/model probes despite their historical test_ names.
collect_ignore = list(MANUAL_SCRIPT_NAMES)
