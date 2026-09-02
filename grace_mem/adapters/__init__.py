"""The one concrete implementation of each external technology.

FalkorDB for the graph, Chroma for the vector stores, rank_bm25 for the lexical
index, OpenAI for the LLM, a pickle file for the extraction cache. There is
exactly one of each, which is why there is no ports/ package -- see
the capability boundary tests for when that would change.
"""
