"""Tests for format pass-through (backend supports client's native format).

Verifies that when the backend supports the client's request format,
requests without tools are forwarded directly without conversion.
Requests with tools always go through the guardrails pipeline.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.core.workflow import TextResponse, ToolCall
from forge.proxy.handler import (
    _PassthroughStream,
    handle_chat_completions,
    handle_messages,
)
from forge.proxy.server import HTTPServer


# ── Helpers ──────────────────────────────────────────────────


def _mock_client(response, streaming=False):
    from forge.clients.base import ChunkType, StreamChunk

    client = AsyncMock()
    client.api_format = "openai"
    client.backend_url = "http://localhost:8080"
    client._models_format = "openai"
    client.send_http = AsyncMock(return_value=response)

    async def fake_stream(*args, **kwargs):
        yield StreamChunk(type=ChunkType.FINAL, response=response)

    client.send_http_stream = fake_stream
    return client


def _make_fake_stream(response):
    """Create a fake streaming generator that yields a single FINAL chunk."""
    from forge.clients.base import ChunkType, StreamChunk

    async def fake_stream(*args, **kwargs):
        yield StreamChunk(type=ChunkType.FINAL, response=response)

    return fake_stream


def _ctx():
    return ContextManager(strategy=NoCompact(), budget_tokens=8192)


def _openai_body(messages=None, tools=None, stream=False, model="test-model"):
    b = {
        "messages": messages or [{"role": "user", "content": "hi"}],
        "model": model,
    }
    if tools is not None:
        b["tools"] = tools
    if stream:
        b["stream"] = True
    return b


def _anthropic_body(messages=None, tools=None, stream=False, model="claude-sonnet-4"):
    b = {
        "model": model,
        "messages": messages or [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
    }
    if tools is not None:
        b["tools"] = tools
    if stream:
        b["stream"] = True
    return b


def _tool_def(name="search", description="Search", parameters=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters or {"type": "object", "properties": {}},
        },
    }


# ── OpenAI pass-through ──────────────────────────────────────


class TestOpenaiPassThrough:
    """OpenAI-format requests to a backend that supports OpenAI."""

    @pytest.mark.asyncio
    async def test_no_tools_non_stream_returns_response(self):
        client = _mock_client(TextResponse(content="Hello!"))
        result = await handle_chat_completions(
            _openai_body(),
            client,
            _ctx(),
            backend_supports_openai=True,
        )
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_no_tools_stream_returns_passthrough_sentinel(self):
        """Streaming pass-through returns _PassthroughStream, not SSE events."""
        client = _mock_client(TextResponse(content="Hello!"))
        result = await handle_chat_completions(
            _openai_body(stream=True),
            client,
            _ctx(),
            backend_supports_openai=True,
        )
        assert isinstance(result, _PassthroughStream)
        assert result.path == "/v1/chat/completions"
        assert result.body["messages"] == [{"role": "user", "content": "hi"}]
        assert result.body["stream"] is True

    @pytest.mark.asyncio
    async def test_passthrough_body_preserves_sampling(self):
        """Sampling fields in the request body are forwarded in pass-through."""
        client = _mock_client(TextResponse(content="ok"))
        body = _openai_body(stream=True)
        body["temperature"] = 0.7
        body["top_p"] = 0.9
        body["seed"] = 42

        result = await handle_chat_completions(
            body, client, _ctx(), backend_supports_openai=True,
        )
        assert isinstance(result, _PassthroughStream)
        assert result.body["temperature"] == 0.7
        assert result.body["top_p"] == 0.9
        assert result.body["seed"] == 42

    @pytest.mark.asyncio
    async def test_with_tools_no_pass_through(self):
        """Requests with tools go through guardrails, not pass-through."""
        client = _mock_client([ToolCall(tool="search", args={"q": "test"})])
        result = await handle_chat_completions(
            _openai_body(tools=[_tool_def("search")]),
            client,
            _ctx(),
            backend_supports_openai=True,
        )
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_with_tools_stream_no_pass_through(self):
        """Streaming with tools goes through guardrails, not pass-through."""
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})], streaming=True)
        result = await handle_chat_completions(
            _openai_body(tools=[_tool_def("search")], stream=True),
            client,
            _ctx(),
            backend_supports_openai=True,
        )
        assert isinstance(result, list)
        assert not isinstance(result, _PassthroughStream)

    @pytest.mark.asyncio
    async def test_backend_does_not_support_openai_no_pass_through(self):
        """When backend doesn't support OpenAI, no pass-through."""
        client = _mock_client(TextResponse(content="ok"))
        result = await handle_chat_completions(
            _openai_body(),
            client,
            _ctx(),
            backend_supports_openai=False,
        )
        assert result["choices"][0]["message"]["content"] == "ok"
        # Should still work, just not pass-through


# ── Anthropic pass-through ───────────────────────────────────


class TestAnthropicPassThrough:
    """Anthropic-format requests to a backend that supports Anthropic."""

    @pytest.mark.asyncio
    async def test_no_tools_non_stream_passes_through(self):
        """Non-streaming Anthropic pass-through returns raw response."""
        client = _mock_client(TextResponse(content="ok"))
        client.send_raw = AsyncMock(
            return_value={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4",
                "content": [{"type": "text", "text": "Hello from backend!"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 10},
            }
        )

        result = await handle_messages(
            _anthropic_body(),
            client,
            _ctx(),
            backend_supports_anthropic=True,
        )
        assert result["type"] == "message"
        assert result["content"][0]["text"] == "Hello from backend!"
        assert result["stop_reason"] == "end_turn"
        client.send_raw.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_tools_stream_returns_passthrough_sentinel(self):
        """Streaming Anthropic pass-through returns _PassthroughStream."""
        client = _mock_client(TextResponse(content="ok"))
        result = await handle_messages(
            _anthropic_body(stream=True),
            client,
            _ctx(),
            backend_supports_anthropic=True,
        )
        assert isinstance(result, _PassthroughStream)
        assert result.path == "/v1/messages"
        assert result.body["messages"] == [{"role": "user", "content": "hi"}]
        assert result.body["stream"] is True
        assert result.headers is not None
        assert "anthropic-version" in (result.headers or {})

    @pytest.mark.asyncio
    async def test_with_tools_no_pass_through(self):
        """Anthropic requests with tools go through guardrails, not pass-through."""
        client = _mock_client([ToolCall(tool="get_weather", args={"city": "London"})])
        body = _anthropic_body(tools=[{
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }])
        result = await handle_messages(
            body, client, _ctx(), backend_supports_anthropic=True,
        )
        # handle_messages delegates to handle_chat_completions which returns
        # OpenAI format by default (anthropic_backend=False). The key point:
        # it went through guardrails, not pass-through.
        assert result["object"] == "chat.completion"
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_with_tools_stream_no_pass_through(self):
        """Streaming with tools goes through guardrails."""
        client = _mock_client([ToolCall(tool="get_weather", args={"city": "Paris"})], streaming=True)
        body = _anthropic_body(
            tools=[{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }],
            stream=True,
        )
        result = await handle_messages(
            body, client, _ctx(), backend_supports_anthropic=True,
        )
        assert isinstance(result, list)
        assert not isinstance(result, _PassthroughStream)

    @pytest.mark.asyncio
    async def test_backend_does_not_support_anthropic_no_pass_through(self):
        """When backend doesn't support Anthropic, convert and delegate.

        handle_messages converts to OpenAI format, delegates to
        handle_chat_completions, which returns OpenAI format by default.
        """
        client = _mock_client(TextResponse(content="ok"))
        result = await handle_messages(
            _anthropic_body(),
            client,
            _ctx(),
            backend_supports_anthropic=False,
        )
        # OpenAI format from handle_chat_completions (anthropic_backend=False)
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "ok"


# ── Server-level pass-through ────────────────────────────────


class TestServerPassThrough:
    """Integration tests for server detecting and handling pass-through."""

    @pytest.fixture
    async def passthrough_server_factory(self):
        """Factory that creates an HTTPServer with capabilities set."""
        servers = []

        async def _make(supports_anthropic=False, supports_openai=False):
            client = AsyncMock()
            client.api_format = "openai"
            client.backend_url = "http://localhost:8080"
            client._models_format = "openai"
            client.send_http = AsyncMock(return_value=TextResponse(content="ok"))
            client.send_http_stream = _make_fake_stream(TextResponse(content="ok"))
            client.send_raw = AsyncMock(return_value={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4",
                "content": [{"type": "text", "text": "pass-through response"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            })

            ctx = ContextManager(strategy=NoCompact(), budget_tokens=8192)
            srv = HTTPServer(
                client=client,
                context_manager=ctx,
                host="127.0.0.1",
                port=0,
                serialize_requests=False,
            )
            await srv.start()
            # Override capabilities to simulate a backend that supports both formats
            from forge.proxy.server import BackendCapabilities
            srv._capabilities = BackendCapabilities(
                supports_anthropic=supports_anthropic,
                supports_openai=supports_openai,
            )
            sock = srv._server.sockets[0]
            port = sock.getsockname()[1]
            servers.append(srv)
            return srv, port, client

        yield _make

        for srv in servers:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_anthropic_non_stream_pass_through(self, passthrough_server_factory):
        """Non-streaming Anthropic request with supports_anthropic=True uses pass-through."""
        srv, port, client = await passthrough_server_factory(supports_anthropic=True)

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            body = {
                "model": "claude-sonnet-4",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            }
            body_bytes = json.dumps(body).encode()
            request = (
                f"POST /v1/messages HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body_bytes)}\r\n"
                f"\r\n"
            ).encode() + body_bytes

            writer.write(request)
            await writer.drain()

            response_data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
            response_str = response_data.decode("utf-8", errors="replace")
            body_start = response_str.find("\r\n\r\n")
            json_str = response_str[body_start + 4:]
            data = json.loads(json_str)

            assert data["type"] == "message"
            assert data["content"][0]["text"] == "pass-through response"
            client.send_raw.assert_called_once()
        finally:
            writer.close()
            await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_openai_non_stream_pass_through(self, passthrough_server_factory):
        """Non-streaming OpenAI request with supports_openai=True uses pass-through."""
        srv, port, client = await passthrough_server_factory(supports_openai=True)

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            body = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            }
            body_bytes = json.dumps(body).encode()
            request = (
                f"POST /v1/chat/completions HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body_bytes)}\r\n"
                f"\r\n"
            ).encode() + body_bytes

            writer.write(request)
            await writer.drain()

            response_data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
            response_str = response_data.decode("utf-8", errors="replace")
            body_start = response_str.find("\r\n\r\n")
            json_str = response_str[body_start + 4:]
            data = json.loads(json_str)

            assert data["choices"][0]["message"]["content"] == "ok"
            assert data["choices"][0]["finish_reason"] == "stop"
        finally:
            writer.close()
            await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_tools_bypass_pass_through(self, passthrough_server_factory):
        """Requests with tools go through guardrails regardless of capabilities."""
        srv, port, client = await passthrough_server_factory(
            supports_anthropic=True,
            supports_openai=True,
        )
        # Override send to return a tool call for the tools path
        from forge.clients.base import ChunkType, StreamChunk

        tool_response = [ToolCall(tool="search", args={"q": "test"})]

        async def fake_stream(*args, **kwargs):
            yield StreamChunk(type=ChunkType.FINAL, response=tool_response)

        client.send_http = AsyncMock(return_value=tool_response)
        client.send_http_stream = fake_stream

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            body = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "search for test"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }],
            }
            body_bytes = json.dumps(body).encode()
            request = (
                f"POST /v1/chat/completions HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body_bytes)}\r\n"
                f"\r\n"
            ).encode() + body_bytes

            writer.write(request)
            await writer.drain()

            response_data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
            response_str = response_data.decode("utf-8", errors="replace")
            body_start = response_str.find("\r\n\r\n")
            json_str = response_str[body_start + 4:]
            data = json.loads(json_str)

            # Should have tool_calls in response, not pass-through
            assert "tool_calls" in data["choices"][0]["message"]
            assert data["choices"][0]["finish_reason"] == "tool_calls"
        finally:
            writer.close()
            await writer.wait_closed()
