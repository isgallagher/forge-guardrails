"""Tests for _get_usage extraction from LLMClient.last_usage."""

import pytest
from forge.proxy.handler import _get_usage
from forge.clients.base import TokenUsage


class TestGetUsage:
    """Test token usage extraction from various client states."""

    def test_no_last_usage_attr(self):
        """Client without last_usage returns None."""
        client = object()
        assert _get_usage(client) is None

    def test_last_usage_not_a_dict(self):
        """Client with non-dict last_usage returns None."""
        client = type("Client", (), {"last_usage": []})()
        assert _get_usage(client) is None

    def test_empty_last_usage(self):
        """Client with empty last_usage dict returns None."""
        client = type("Client", (), {"last_usage": {}})()
        assert _get_usage(client) is None

    def test_slot_id_key_found(self):
        """TokenUsage at slot_id key is converted to dict."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        client = type("Client", (), {
            "last_usage": {"0": usage},
            "_slot_id": 0,
        })()
        result = _get_usage(client)
        assert result == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    def test_slot_id_string_key(self):
        """OpenAICompatibleClient uses string keys."""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        client = type("Client", (), {
            "last_usage": {"0": usage},
            "_slot_id": 0,
        })()
        result = _get_usage(client)
        assert result is not None
        assert result["prompt_tokens"] == 10

    def test_default_slot_id_zero(self):
        """When _slot_id is missing, defaults to 0."""
        usage = TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        client = type("Client", (), {"last_usage": {"0": usage}})()
        result = _get_usage(client)
        assert result is not None
        assert result["total_tokens"] == 30

    def test_missing_slot_key_returns_none(self):
        """When slot_id key not in last_usage, returns None."""
        usage = TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        client = type("Client", (), {
            "last_usage": {"1": usage},
            "_slot_id": 0,
        })()
        assert _get_usage(client) is None

    def test_non_token_usage_value_returns_none(self):
        """Non-TokenUsage values are skipped."""
        client = type("Client", (), {
            "last_usage": {"0": "not a usage"},
            "_slot_id": 0,
        })()
        assert _get_usage(client) is None
