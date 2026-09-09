"""What survives, and in what shape it goes back out.

    context        reading the answer context apart and rendering it back, and
                   the wire format (EVIDENCE_HEADER, SID_RE) both directions use
    finalization   the deterministic selection policy: mode, provenance, cap
    adjudication   an answer-blind second opinion that may only add evidence back

`finalization` and `adjudication` stay separate because one is a deterministic
policy and the other is an extra LLM call that can be switched off on its own.

Not to be confused with `grace_mem.retrieval.evidence`, which builds the
evidence block in the first place; this package only refines one it was given.
"""
