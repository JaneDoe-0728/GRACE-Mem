"""Judge and open-domain prompt message builders.

The `build_*` wrappers below forward to `helpers.llm` at call time rather than
importing it here. That indirection breaks a cycle: the helpers need the prompt
templates this package exports, so importing them at module scope would have
each waiting on the other.
"""

def build_judge_standard_messages(*args, **kwargs):
    from experiment.locomo.helpers.llm import build_judge_standard_messages as _impl

    return _impl(*args, **kwargs)


def build_open_domain_standard_messages(*args, **kwargs):
    from experiment.locomo.helpers.llm import (
        build_open_domain_standard_messages as _impl,
    )

    return _impl(*args, **kwargs)


__all__ = [
    "build_judge_standard_messages",
    "build_open_domain_standard_messages",
]
