"""Benchmark infrastructure shared by more than one dataset.

`agent_filter.py` is where both pipelines mount `grace_mem.agent_filter`, and it
is the only module that knows the settings come from
`experiment_config.GREP_AGENT_PARAMS`. `post_retrieval/` holds one entry point
per benchmark -- dataset-specific, unlike the rest of this package, but kept
together because both replay a finished run through everything that happens
after retrieval.
"""
