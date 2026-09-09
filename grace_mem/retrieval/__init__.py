"""Turning a Query into an Evidence block.

`pipeline.Retriever.build_kg_context()` is the only entry point. It reads as the
pipeline it is, and each stage lives in the subpackage named after it:

    query/          rewrite relative time, extract keywords
        |
    candidates/     hybrid search, graph expansion, temporal scoring
        |
    ranking/        rerank, then threshold and cut
        |
    evidence/       provenance -> snippets, top-K, narrow, render
        |
    answer context

    observability/  what each stage did, orthogonal to all of it

    config.py       every knob, plus the KG_ABLATION_* switches
    models.py       the types that cross a stage boundary
    prompts/        the prompt text those stages send
"""
