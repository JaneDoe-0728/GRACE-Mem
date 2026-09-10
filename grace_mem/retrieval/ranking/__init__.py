"""Deciding which candidates survive.

    reranker   the scoring backend: Qwen3-Reranker in generative mode, API or local
    filter     the policy applied to those scores: intersection, threshold, top-K

The reranker *is* the filter in this pipeline -- there is no separate similarity
cut in front of it -- which is why the two live together but stay apart: one is
a model, the other is a decision.
"""
