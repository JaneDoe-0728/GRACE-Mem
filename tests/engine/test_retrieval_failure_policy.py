"""What a run records when retrieval raises.

The tolerant default is right for a service and wrong for a benchmark: a
FalkorDB timeout used to return the same empty context as a genuine miss, so
the answering model still produced an answer and the row was still scored.
These pin both policies and the marker that tells them apart afterwards.
"""

from __future__ import annotations

import pytest

from grace_mem.retrieval.config import RetrieverConfig
from grace_mem.retrieval.pipeline import (
    STRICT_ENV,
    RetrievalFailedError,
    Retriever,
    strict_retrieval_enabled,
)


class _Trace:
    """The one attribute build_kg_context sets before entering its try block."""

    last_evidence_trace: dict = {}


class _Exploding:
    """Stands in for a backend that is down once retrieval actually starts."""

    def __getattr__(self, name):
        raise RuntimeError("falkordb is down")


def _retriever() -> Retriever:
    """A Retriever wired to fail inside build_kg_context's try block.

    Built with `object.__new__` because `__init__` wants live services; only the
    attributes the failing path reaches are populated.
    """
    r = object.__new__(Retriever)
    r.cfg = RetrieverConfig()
    r.evidence_builder = _Trace()
    r.graph = _Exploding()
    r.llm = _Exploding()
    r.searcher = _Exploding()
    return r


def test_a_retrieval_failure_degrades_to_an_empty_context_by_default(monkeypatch) -> None:
    monkeypatch.delenv(STRICT_ENV, raising=False)

    context = _retriever().build_kg_context(question="anything")

    assert context == "(no KG context)"


def test_the_trace_marks_the_failure_even_on_the_tolerant_path(monkeypatch) -> None:
    # Without this a caller cannot tell "retrieval found nothing" from
    # "retrieval never ran", which is what made the two scoreable alike.
    monkeypatch.delenv(STRICT_ENV, raising=False)
    r = _retriever()

    r.build_kg_context(question="anything")

    assert r.last_retrieval_trace["retrieval_failed"] is True
    assert r.last_retrieval_trace["stop_reason"] == "build_kg_context_failed"


def test_strict_mode_raises_instead_of_answering_from_nothing(monkeypatch) -> None:
    monkeypatch.setenv(STRICT_ENV, "1")

    with pytest.raises(RetrievalFailedError) as excinfo:
        _retriever().build_kg_context(question="anything")

    assert isinstance(excinfo.value.__cause__, RuntimeError)


@pytest.mark.parametrize("value", ["0", "", "false", "FALSE"])
def test_the_strict_switch_is_off_for_every_falsy_spelling(monkeypatch, value: str) -> None:
    monkeypatch.setenv(STRICT_ENV, value)

    assert strict_retrieval_enabled() is False
