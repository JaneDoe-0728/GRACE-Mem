"""Benchmark infrastructure shared by more than one dataset.

Most of it carries no dataset-specific semantics. `replay/` is the exception:
it holds one entry point per benchmark, kept together because both replay the
same mechanism (`grace_mem.agent_filter`) over a finished run.
"""
