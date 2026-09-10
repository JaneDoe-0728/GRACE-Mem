"""What the agent is allowed to look at.

    corpus   the per-question turns, with GREP and READ over them
    vector   embedding search over the summaries VDB the run already built

Two deliberately unmerged backends: `corpus` is an in-memory structured text
store built per question, `vector` lazily loads an embedder and caches a Chroma
client. They answer different commands and fail in different ways.

Not to be confused with `grace_mem.retrieval`, the retrieval pipeline that
produced the context Agent Filter is handed. This package only re-reads the
corpus that pipeline drew from.
"""
