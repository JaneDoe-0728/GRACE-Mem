"""Going round a second time when the first pass looks unconvincing.

    confidence   how good was pass 1, and what specifically failed
    controller   what to do about it: rewrite the query, re-search, merge additively

Off by default (`enable_adaptive_search`). Pass 2 may only add to pass 1's
result, never remove from it.
"""
