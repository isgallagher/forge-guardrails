"""Tests for AnthropicClient.send_http() and send_http_stream() methods.

Verifies the direct HTTP path works with mocked httpx responses.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from forge.clients.anthropic import AnthropicClient
from forge.clients.base import TokenUsage
from forge.core.workflow import TextResponse, ToolCall


class _AsyncIterator:
    """Mock async iterator over a list of values."""

    def __init__(self, values: list[str]):
        self._iter = iter(values)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def _make_client() -> AnthropicClient:
    """Create an AnthropicClient with a base_url (so _http is initialised)."""
    return AnthropicClient(
        model="claude-sonnet-4",
        api_key="none",
        base_url="http://localhost:9999",
    )


class TestBuildRequestBody:
    def test_basic_body(self):
        client = _make_client()
        body = client._build_request_body(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
        )
        assert body["model"] == "claude-sonnet-4"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["max_tokens"] == 8192
        assert "system" not in body
        assert "tools" not in body

    def test_system_extracted(self):
        client = _make_client()
        body = client._build_request_body(
            messages=[
                {"role": "system", "content": "Be nice."},
                {"role": "user", "content": "hi"},
            ],
            tools=None,
        )
        assert body["system"] == "Be nice."
        assert len(body["messages"]) == 1

    def test_max_tokens_override(self):
        client = _make_client()
        body = client._build_request_body(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=4096,
        )
        assert body["max_tokens"] == 4096


class TestParseHttpResponse:
    def test_text_response(self):
        data = {
            "id": "msg_123",
            "content": [{"type": "text", "text": "Hello!"}],
        }
        result = AnthropicClient._parse_http_response(data)
        assert isinstance(result, TextResponse)
        assert result.content == "Hello!"

    def test_tool_use_response(self):
        data = {
            "id": "msg_123",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                }
            ],
        }
        result = AnthropicClient._parse_http_response(data)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].tool == "get_weather"
        assert result[0].args == {"city": "Paris"}

    def test_tool_use_with_text_reasoning(self):
        data = {
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "search",
                    "input": {"q": "test"},
                },
            ],
        }
        result = AnthropicClient._parse_http_response(data)
        assert isinstance(result, list)
        assert result[0].reasoning == "Let me check."

    def test_empty_content(self):
        data = {"content": []}
        result = AnthropicClient._parse_http_response(data)
        assert isinstance(result, TextResponse)
        assert result.content == ""


class TestSendHttp:
    @pytest.mark.asyncio
    async def test_text_response(self):
        client = _make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_123",
            "model": "claude-sonnet-4",
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        with patch.object(client._http, "post", new=AsyncMock(return_value=mock_response)):
            result = await client.send_http([{"role": "user", "content": "hi"}])

        assert isinstance(result, TextResponse)
        assert result.content == "Hello!"
        usage = client.last_usage["0"]
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_tool_call_response(self):
        client = _make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_123",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "search",
                    "input": {"q": "test"},
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 15},
        }

        with patch.object(client._http, "post", new=AsyncMock(return_value=mock_response)):
            result = await client.send_http([{"role": "user", "content": "search"}])

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].tool == "search"

    @pytest.mark.asyncio
    async def test_non_200_raises_backend_error(self):
        client = _make_client()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch.object(client._http, "post", new=AsyncMock(return_value=mock_response)):
            from forge.errors import BackendError
            with pytest.raises(BackendError) as exc_info:
                await client.send_http([{"role": "user", "content": "hi"}])
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_request_body_correct(self):
        client = _make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

        with patch.object(client._http, "post", new=AsyncMock(return_value=mock_response)) as mock_post:
            await client.send_http(
                messages=[
                    {"role": "system", "content": "Be nice."},
                    {"role": "user", "content": "hi"},
                ],
                max_tokens=4096,
            )

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "/v1/messages"
        body = call_args[1]["json"]
        assert body["model"] == "claude-sonnet-4"
        assert body["system"] == "Be nice."
        assert body["max_tokens"] == 4096


class TestSendHttpStream:
    @pytest.mark.asyncio
    async def test_text_stream(self):
        client = _make_client()

        sse_lines = [
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"He"}}',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"llo"}}',
            'data: {"type":"content_block_stop","index":0}',
            'data: {"type":"message_delta","usage":{"output_tokens":5},"stop_reason":"end_turn"}',
            'data: {"type":"message_stop"}',
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = MagicMock(return_value=_AsyncIterator(sse_lines))

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        with patch.object(client._http, "stream", return_value=mock_stream):
            chunks = [chunk async for chunk in client.send_http_stream([{"role": "user", "content": "hi"}])]

        text_chunks = [c for c in chunks if c.type.value == "text_delta"]
        final_chunk = [c for c in chunks if c.type.value == "final"]
        assert len(text_chunks) == 2
        assert "".join(c.content for c in text_chunks) == "Hello"
        assert len(final_chunk) == 1
        assert isinstance(final_chunk[0].response, TextResponse)
        assert final_chunk[0].response.content == "Hello"

    @pytest.mark.asyncio
    async def test_tool_call_stream(self):
        client = _make_client()

        sse_lines = [
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","name":"search","id":"toolu_1","input":{}}}',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"q\\":"}}',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\\"test\\"}"}}',
            'data: {"type":"content_block_stop","index":0}',
            'data: {"type":"message_delta","usage":{"output_tokens":5},"stop_reason":"tool_use"}',
            'data: {"type":"message_stop"}',
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = MagicMock(return_value=_AsyncIterator(sse_lines))

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        with patch.object(client._http, "stream", return_value=mock_stream):
            chunks = [chunk async for chunk in client.send_http_stream([{"role": "user", "content": "search"}])]

        final_chunk = [c for c in chunks if c.type.value == "final"]
        assert len(final_chunk) == 1
        result = final_chunk[0].response
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].tool == "search"
        assert result[0].args == {"q": "test"}

    @pytest.mark.asyncio
    async def test_non_200_stream_raises_backend_error(self):
        client = _make_client()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.aread = AsyncMock(return_value=b"Internal Server Error")

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        with patch.object(client._http, "stream", return_value=mock_stream):
            from forge.errors import BackendError
            with pytest.raises(BackendError) as exc_info:
                async for _ in client.send_http_stream([{"role": "user", "content": "hi"}]):
                    pass
            assert exc_info.value.status_code == 500


class TestAclose:
    @pytest.mark.asyncio
    async def test_aclose_closes_http_client(self):
        client = _make_client()
        client._client.aclose = AsyncMock()
        client._http.aclose = AsyncMock()

        await client.aclose()

        client._client.aclose.assert_called_once()
        client._http.aclose.assert_called_once()
