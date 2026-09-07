"""Benchmark entry points that rerun everything after retrieval.

Both reuse a finished run's Retrieved_Context and replay only what follows it --
Agent Filter, then answering -- so the two arms get bit-identical retrieval
input and any difference in the score belongs to the stages under test.
"""
