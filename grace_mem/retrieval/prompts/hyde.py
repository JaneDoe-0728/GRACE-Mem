"""
HyDE-style prompt: generate hypothetical memory summary sentences for a question.

The generated sentences are embedded and compared against stored summary vectors
to improve summary retrieval, complementing the raw query embedding. They are
content sentences that look like real memories, not retrieval instructions.
"""

HYDE_SYSTEM = (
    """Generate 3 hypothetical memory summary sentences that could answer the question.

Rules:
- Use only entities, locations, and time expressions explicitly present in the question.
- Do not invent exact dates, names, numbers, durations, or specific attribute values
  (e.g. do not guess a car type, a city, or a count). Keep the unknown value general.
- Do not include any calendar date, year, or month in the output, unless that exact
  date appears verbatim in the question. Never echo the question date.
- If the answer value is unknown, still write a positive declarative sentence that states
  the relation, just without the value. Describe what happened, not what is missing.
- Never write uncertainty or refusal phrasing such as "I don't know", "I don't recall",
  "unclear", "unspecified", "not recorded", "remains unknown", or "hasn't been mentioned".
  Every sentence must read like a confidently recalled memory.
- Make each sentence look like a real memory summary, not a search instruction.
- Each sentence under 25 words.

Examples:
- Question: "How many times have I met up with Alex from Germany?"
  Good: "The user met up with Alex from Germany on several occasions."
  Bad:  "I don't recall how many times I met Alex from Germany."
- Question: "What type of vehicle model am I currently working on?"
  Good: "The user is currently working on a vehicle model."
  Bad:  "The user is developing a compact electric sedan with autonomous features."

Output the 3 sentences only, one per line, with no numbering or extra text."""
)

HYDE_USER = (
    "Question: {question}"
)
