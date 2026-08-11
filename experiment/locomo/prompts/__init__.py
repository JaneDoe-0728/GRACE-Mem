from .judge import PROMPT_TEMPLATES, SYSTEM_PROMPT_PLUS


def build_judge_plus_messages(*args, **kwargs):
    from experiment.locomo.helpers.llm import build_judge_plus_messages as _impl

    return _impl(*args, **kwargs)


def build_judge_standard_messages(*args, **kwargs):
    from experiment.locomo.helpers.llm import build_judge_standard_messages as _impl

    return _impl(*args, **kwargs)


def build_open_domain_plus_messages(*args, **kwargs):
    from experiment.locomo.helpers.llm import build_open_domain_plus_messages as _impl

    return _impl(*args, **kwargs)


def build_open_domain_standard_messages(*args, **kwargs):
    from experiment.locomo.helpers.llm import build_open_domain_standard_messages as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "PROMPT_TEMPLATES",
    "SYSTEM_PROMPT_PLUS",
    "build_judge_plus_messages",
    "build_judge_standard_messages",
    "build_open_domain_plus_messages",
    "build_open_domain_standard_messages",
]
