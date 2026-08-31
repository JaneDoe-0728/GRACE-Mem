"""Every prompt Agent Filter sends, grouped by the call that sends it.

    agent          the search loop's system and task prompts
    verification   the sufficiency verifier, and the gap hint its verdict feeds back
    adjudication   the answer-blind auditor of the seeds FINAL discarded
    skills         question-shape driven search tactics, injected into the task prompt
"""
