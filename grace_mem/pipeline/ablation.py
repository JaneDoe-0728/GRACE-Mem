"""The KG_ABLATION_* switches, in one place.

Each flag removes one retrieval channel so a run can be compared against the
same system without it. They were read in three modules with three slightly
different expressions; the names now live here so the set of ablations is
something you can look up rather than grep for.

The reader deliberately does not `.strip()`. Two of the three call sites it
replaces did not, and `grace_mem.temporal.normalizer` -- which does, and also
carries a legacy alias -- keeps its own reader rather than have this one
quietly start accepting `" 1 "` where it used to reject it.
"""

import os

#: Every ablation switch, and the channel it removes.
ABLATIONS = {
    "KG_ABLATION_NO_BM25": "lexical half of hybrid entity search",
    "KG_ABLATION_NO_DIRECT_VECTOR": "direct summary-vector retrieval",
    "KG_ABLATION_NO_GRAPH": "the graph channel entirely",
    "KG_ABLATION_NO_KEYWORDS": "LLM keyword extraction",
    "KG_ABLATION_NO_KG_TEXT": "entity/relationship text, keeping the graph",
    "KG_ABLATION_NO_TEMPORAL_BOOST": "temporal containment reranking",
    "KG_ABLATION_NO_TIME_REWRITE": "query-side temporal rewriting",
}


def flag_enabled(name: str) -> bool:
    """True when `name` is set to anything other than 0, empty, or false."""
    return os.getenv(name, "0").lower() not in ("0", "", "false")
