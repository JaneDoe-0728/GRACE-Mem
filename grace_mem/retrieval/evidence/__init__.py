"""From surviving candidates to the block the answering model reads.

    builder     provenance -> summary/raw events, scored and deduplicated
    ranking     the global top-K cut, with a per-source quota ahead of it
    source      raw conversation turns, read back out of the script CSVs
    narrowing   a last pass that drops evidence the question does not need
    rendering   entities and relationships as the text the LLM sees

`builder` is the sequence; the rest are the steps it delegates to or the
adapters it reads through.
"""
