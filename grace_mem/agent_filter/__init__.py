"""Agent Filter: an optional post-retrieval evidence-refinement layer.

An existing run's retrieved context goes in; the agent inspects the question's
corpus with GREP, READ and VECTOR, and a refined context comes out. Any failure
along the way hands the original context back untouched.

Start at harness.py, which is the sequence the other modules run in:

    harness.py     prepare -> search -> finalize, and the fallback on any failure
    config.py      GREP_AGENT_PARAMS as one typed dataclass

    runtime/       how the agent runs: the loop, the reply protocol, the client
    retrieval/     what it may look at: the corpus, and vector search
    evidence/      what survives: context codec, selection policy, adjudication
    prompting/     every prompt, grouped by the call that sends it

`retrieval/` and `evidence/` name this package's own sub-capabilities, not
`grace_mem.retrieval` or `grace_mem.retrieval.evidence`. Those produced the
context; these refine one.
"""
