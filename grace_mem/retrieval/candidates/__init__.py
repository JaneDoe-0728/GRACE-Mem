"""Finding what might be relevant, before anything decides what survives.

    search           hybrid vector + BM25 over entities, vector over relationships
    graph_expansion  spreading activation from the seeds, for what only multi-hop reaches
    temporal         Weibull time-decay scoring, and coarse-range containment

Everything here widens the pool or scores it. Nothing here cuts it; that is
`ranking/`.
"""
