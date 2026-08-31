"""Agent Filter: an optional post-retrieval evidence-refinement layer.

An existing run's retrieved context goes in; the agent inspects the question's
corpus with GREP, READ and VECTOR, and a refined context comes out. Any failure
along the way hands the original context back untouched.

Start at harness.py, which is the sequence the other modules run in.
"""
