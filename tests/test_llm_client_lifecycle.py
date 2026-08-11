from unittest.mock import Mock

import pytest

from KG.llm.client import LLMClient


def _client_without_init() -> LLMClient:
    client = object.__new__(LLMClient)
    client.client = Mock()
    client._closed = False
    return client


def test_llm_client_close_is_idempotent():
    client = _client_without_init()

    client.close()
    client.close()

    client.client.close.assert_called_once_with()


def test_llm_client_context_manager_closes_on_error():
    client = _client_without_init()

    with pytest.raises(RuntimeError, match="request failed"):
        with client:
            raise RuntimeError("request failed")

    client.client.close.assert_called_once_with()
