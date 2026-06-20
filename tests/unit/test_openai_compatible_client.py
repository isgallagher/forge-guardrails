"""Tests for forge.clients.openai_compatible — OpenAI-compatible backend client."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.clients.openai_compatible import OpenAICompatibleClient
from forge.core.workflow import TextResponse, ToolCall


class _AsyncContextManager:
    """Minimal async context manager for mocking httpx.stream()."""

    def __init__(self, aiter_lines_result, raise_for_status_result=None):
        self._aiter_lines_result = aiter_lines_result
        self.raise_for_status = raise_for_status_result or MagicMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def aiter_lines(self):
        for line in self._aiter_lines_result:
            yield line


class TestOpenAICompatibleClientSend:
    """Test non-streaming send()."""

    @pytest.mark.asyncio
    async def test_text_response(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            MockClient.return_value = instance

            client = OpenAICompatibleClient(base_url="http://localhost:8080")
            # Patch the internal client
            client._client = instance

            result = await client.send([{"role": "user", "content": "hi"}])
            assert isinstance(result, TextResponse)
            assert result.content == "Hello!"
            assert client.last_usage["0"].prompt_tokens == 10
            assert client.last_usage["0"].completion_tokens == 5

    @pytest.mark.asyncio
    async def test_tool_call_response(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        }
                    }],
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            MockClient.return_value = instance

            client = OpenAICompatibleClient(base_url="http://localhost:8080")
            client._client = instance

            result = await client.send(
                [{"role": "user", "content": "Weather?"}],
                tools=[],
            )
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0].tool == "get_weather"
            assert result[0].args == {"city": "Paris"}

    @pytest.mark.asyncio
    async def test_sends_tools_in_openai_format(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        client = OpenAICompatibleClient(base_url="http://localhost:8080")
        client._client = mock_client

        from forge.core.workflow import ToolSpec
        from pydantic import BaseModel

        class SearchParams(BaseModel):
            q: str = "search query"

        spec = ToolSpec(name="search", description="Search", parameters=SearchParams)
        await client.send([{"role": "user", "content": "Go"}], tools=[spec])

        call_args = mock_client.post.call_args
        body = call_args[1]["json"]
        assert "tools" in body
        assert body["tools"][0]["type"] == "function"
        assert body["tools"][0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_sends_max_tokens(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            MockClient.return_value = instance

            client = OpenAICompatibleClient(base_url="http://localhost:8080")
            client._client = instance

            await client.send(
                [{"role": "user", "content": "hi"}],
                max_tokens=50,
            )
            body = instance.post.call_args.kwargs["json"]
            assert body["max_tokens"] == 50

    @pytest.mark.asyncio
    async def test_sends_sampling_params(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            MockClient.return_value = instance

            client = OpenAICompatibleClient(base_url="http://localhost:8080")
            client._client = instance

            await client.send(
                [{"role": "user", "content": "hi"}],
                sampling={"temperature": 0.7, "top_p": 0.9},
            )
            body = instance.post.call_args.kwargs["json"]
            assert body["temperature"] == 0.7
            assert body["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_empty_content_returns_empty_text(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": ""}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            MockClient.return_value = instance

            client = OpenAICompatibleClient(base_url="http://localhost:8080")
            client._client = instance

            result = await client.send([{"role": "user", "content": "hi"}])
            assert isinstance(result, TextResponse)
            assert result.content == ""

    @pytest.mark.asyncio
    async def test_args_json_decode_error_falls_back_to_empty(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {"name": "f", "arguments": "not json"}
                    }]
                }
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            MockClient.return_value = instance

            client = OpenAICompatibleClient(base_url="http://localhost:8080")
            client._client = instance

            result = await client.send([{"role": "user", "content": "go"}])
            assert isinstance(result, list)
            assert result[0].args == {}


class TestOpenAICompatibleClientStream:
    """Test streaming send_stream()."""

    @pytest.mark.asyncio
    async def test_streaming_text(self):
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"H"}}]}',
            'data: {"choices":[{"delta":{"content":"i"}}]}',
            'data: {"choices":[{"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]

        mock_stream = _AsyncContextManager(iter(sse_lines))
        mock_client = MagicMock()
        mock_client.stream.return_value = mock_stream

        client = OpenAICompatibleClient(base_url="http://localhost:8080")
        client._client = mock_client

        chunks = []
        async for chunk in client.send_stream([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert len(chunks) == 3  # 2 text deltas + 1 FINAL
        assert chunks[0].type.value == "text_delta"
        assert chunks[0].content == "H"
        assert chunks[1].type.value == "text_delta"
        assert chunks[1].content == "i"
        assert chunks[2].type.value == "final"

    @pytest.mark.asyncio
    async def test_streaming_tool_call(self):
        sse_lines = [
            'data: {"choices":[{"delta":{"content":null,"tool_calls":[{"function":{"name":"search","arguments":""}}]}}]}',
            'data: {"choices":[{"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]

        mock_stream = _AsyncContextManager(iter(sse_lines))
        mock_client = MagicMock()
        mock_client.stream.return_value = mock_stream

        client = OpenAICompatibleClient(base_url="http://localhost:8080")
        client._client = mock_client

        chunks = []
        async for chunk in client.send_stream([{"role": "user", "content": "search"}]):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].type.value == "tool_call_delta"
        assert chunks[1].type.value == "final"

    @pytest.mark.asyncio
    async def test_streaming_usage_extracted(self):
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}',
            'data: {"choices":[{"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]

        mock_stream = _AsyncContextManager(iter(sse_lines))
        mock_client = MagicMock()
        mock_client.stream.return_value = mock_stream

        client = OpenAICompatibleClient(base_url="http://localhost:8080")
        client._client = mock_client

        chunks = []
        async for chunk in client.send_stream([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert "0" in client.last_usage
        assert client.last_usage["0"].prompt_tokens == 10
        assert client.last_usage["0"].completion_tokens == 5


class TestOpenAICompatibleClientInit:
    """Test client construction."""

    def test_base_url_stripped(self):
        client = OpenAICompatibleClient(base_url="http://localhost:8080/")
        assert client.backend_url == "http://localhost:8080"

    def test_api_format_is_openai(self):
        client = OpenAICompatibleClient(base_url="http://localhost:8080")
        assert client.api_format == "openai"
        assert client._models_format == "openai"

    def test_default_timeout(self):
        with patch("httpx.AsyncClient") as MockClient:
            OpenAICompatibleClient(base_url="http://localhost:8080")
            MockClient.assert_called_once()
            assert MockClient.call_args.kwargs["timeout"] == 120.0

    def test_api_key_in_header(self):
        with patch("httpx.AsyncClient") as MockClient:
            OpenAICompatibleClient(
                base_url="http://localhost:8080",
                api_key="my-key",
            )
            headers = MockClient.call_args.kwargs["headers"]
            assert headers["Authorization"] == "Bearer my-key"

    def test_no_api_key(self):
        with patch("httpx.AsyncClient") as MockClient:
            OpenAICompatibleClient(base_url="http://localhost:8080")
            headers = MockClient.call_args.kwargs["headers"]
            assert "Authorization" not in headers
