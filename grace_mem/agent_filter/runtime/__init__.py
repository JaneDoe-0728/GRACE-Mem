"""How the agent runs: one conversation, and the reply format it speaks.

    session    the search loop and its GREP/READ/VECTOR tools
    protocol   one model reply in, one Command out, whatever channel it arrived in
    llm        which endpoint the agent talks to

Nothing here decides what evidence survives -- that is `evidence/` -- or what
the agent is allowed to look at, which is `retrieval/`.
"""
