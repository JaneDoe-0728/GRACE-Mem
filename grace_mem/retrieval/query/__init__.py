"""Turning the user's question into something the stores can be searched with.

    rewrite    relative time expressions -> absolute dates, before anything is looked up
    keywords   one LLM call -> high-level and low-level keyword lists, cached on disk

Both run before any store is touched, and neither changes the question the user
actually asked -- only the query derived from it.
"""
