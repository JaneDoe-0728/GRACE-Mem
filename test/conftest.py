"""Pytest collection policy for the offline regression suite."""

# These are manual network/model scripts despite their historical test_ names.
# Keeping them out of collection makes the default suite deterministic and offline.
collect_ignore = [
    "test_api.py",
    "test_gemini.py",
    "test_gemini_new.py",
    "test_gpt.py",
    "test_gpt_new.py",
    "test_inference_keyword.py",
    "test_reranker.py",
    "test_sa_retrieve.py",
    "test_update.py",
    # This contract targets an analysis package absent from the tracked repository.
    "test_benchmark_analysis_imports.py",
]
