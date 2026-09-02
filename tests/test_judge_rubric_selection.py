"""Which judge rubric a call gets, and who decides.

An abstention question graded on the general rubric is marked wrong almost every
time -- the general prompt has no abstention exemption, which is the whole reason
ABS_JUDGE_PROMPT exists. So "did this call reach the right rubric" is not a
detail of the judge; it is the difference between two runs being comparable and
not.

Two sources decide it, in this order: the caller's explicit flag, which comes
from the dataset's `_abs` filename tag and is authoritative, and -- only when the
caller passes nothing -- the gold text.
"""

from __future__ import annotations

from experiment.common.evaluation.judge import JudgeEngine
from experiment.longmem.prompts import build_judge_messages, is_abstention_gold

ABS_GOLD = "The information provided is not enough to answer this question."
PLAIN_GOLD = "Three months."


def _rubric(messages: list[dict[str, str]]) -> str:
    text = "\n".join(m["content"] for m in messages)
    return "abstention" if "grading an ABSTENTION question" in text else "general"


def test_an_omitted_flag_falls_back_to_the_gold_text() -> None:
    """The default has to be "you decide", not False.

    A `bool = False` default reads as an explicit "this is not an abstention" by
    the time it reaches build_messages, which silently disables the fallback for
    every caller that simply did not pass the argument.
    """
    assert is_abstention_gold(ABS_GOLD)
    assert _rubric(build_judge_messages(question="q", gold=ABS_GOLD, generated="a")) == "abstention"


def test_an_explicit_flag_beats_the_gold_text_in_both_directions() -> None:
    """The filename tag is authoritative: a plain question in an `_abs` file is
    still an abstention, and abstention-sounding gold in a normal file is not."""
    assert _rubric(
        build_judge_messages(question="q", gold=PLAIN_GOLD, generated="a", is_abstention=True)
    ) == "abstention"
    assert _rubric(
        build_judge_messages(question="q", gold=ABS_GOLD, generated="a", is_abstention=False)
    ) == "general"


def test_the_engine_carries_the_omitted_flag_through_as_undecided() -> None:
    """JudgeEngine sits between every caller and the prompt builder; if it
    substitutes False for a missing flag, the fallback can never fire."""
    engine = JudgeEngine(llm=None, benchmark="longmem")

    assert engine._resolve_abstention(ABS_GOLD, None) is True
    assert engine._resolve_abstention(PLAIN_GOLD, None) is False
    assert engine._resolve_abstention(ABS_GOLD, False) is False
    assert engine._resolve_abstention(PLAIN_GOLD, True) is True


def test_locomo_never_takes_the_longmem_abstention_route() -> None:
    """LoCoMo has no abstention rubric; gold-shaped guessing must not leak in."""
    engine = JudgeEngine(llm=None, benchmark="locomo")

    assert engine._resolve_abstention(ABS_GOLD, None) is False


def test_the_rerun_and_replay_paths_pass_what_the_runner_passes() -> None:
    """The two call sites that were missed when the runner was updated.

    Re-judging `*_abs.csv` through pipeline.rerun (or `watchdog --rerun`), and
    replaying through analysis.fact_replay, graded clean abstentions on the general
    rubric -- so a re-judge disagreed with the run it was re-judging.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for module, callee in (
        ("experiment/longmem/pipeline/rerun.py", "judge_single"),
        ("experiment/longmem/analysis/fact_replay.py", "judge_single"),
    ):
        tree = ast.parse((root / module).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == callee
        ]
        assert calls, f"{module} no longer calls {callee}"
        for call in calls:
            keywords = {kw.arg for kw in call.keywords}
            assert "is_abstention" in keywords, f"{module}:{call.lineno} judges without a rubric flag"
            assert "category" in keywords, f"{module}:{call.lineno} judges without a category"
