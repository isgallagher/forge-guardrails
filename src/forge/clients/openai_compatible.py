"""Generic HTTP client for OpenAI-compatible backends.

Works with llama-server, vLLM, Ollama API, and any backend
implementing the OpenAI chat completions interface.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from forge.clients.base import ChunkType, LLMClient, StreamChunk, TokenUsage
from forge.core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec

logger = logging.getLogger("forge.clients.openai_compatible")


class OpenAICompatibleClient(LLMClient):
    """HTTP client for OpenAI-compatible chat completion backends.

    Sends messages in OpenAI format, parses standard OpenAI responses
    (both streaming and non-streaming) into forge's canonical types.
    """

    api_format = "openai"
    _models_format = "openai"

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_tokens: int = 8192,
    ) -> None:
        """
        Args:
            base_url: Backend URL (e.g. http://localhost:8080).
            api_key: Optional API key for backends that require auth.
            timeout: Request timeout in seconds.
            max_tokens: Default response token ceiling when the caller
                does not provide one. Prevents unbounded generation.
        """
        self.backend_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=self._headers(api_key),
            timeout=timeout,
        )
        self.last_usage: dict[str, Any] = {}
        self._slot_id: int = 0

    @staticmethod
    def _headers(api_key: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def send(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Send messages and return a parsed response."""
        body: dict[str, Any] = {}
        if model:
            body["model"] = model
        if messages:
            body["messages"] = messages
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.get_json_schema(),
                    },
                }
                for t in tools
            ]
        if sampling:
            for key in ("temperature", "top_p", "top_k", "min_p", "seed"):
                if key in sampling:
                    body[key] = sampling[key]
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        elif self.max_tokens:
            body["max_tokens"] = self.max_tokens

        resp = await self._client.post("/v1/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        return self._parse_response(data)

    async def send_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response chunks via SSE."""
        body: dict[str, Any] = {
            "messages": messages,
            "stream": True,
        }
        model_kwarg = sampling.get("model") if sampling else None
        if model_kwarg:
            body["model"] = model_kwarg
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.get_json_schema(),
                    },
                }
                for t in tools
            ]
        if sampling:
            for key in ("temperature", "top_p", "top_k", "min_p", "seed"):
                if key in sampling:
                    body[key] = sampling[key]
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        elif self.max_tokens:
            body["max_tokens"] = self.max_tokens

        async with self._client.stream("POST", "/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            # Accumulate for final response
            accumulated_text = ""
            tool_blocks: list[dict[str, str]] = []  # [{name, args}, ...]
            current_tool_idx: int = -1

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                # Track usage
                usage_data = data.get("usage")
                if usage_data:
                    self.last_usage[str(self._slot_id)] = TokenUsage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        total_tokens=usage_data.get("total_tokens", 0),
                    )

                choices = data.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                finish = choice.get("finish_reason")
                delta = choice.get("delta", {})

                if finish:
                    # Build final response from accumulated data
                    if tool_blocks:
                        final: LLMResponse = [
                            ToolCall(
                                tool=tb["name"],
                                args=json.loads(tb["args"]) if tb["args"] else {},
                            )
                            for tb in tool_blocks
                        ]
                    else:
                        final = TextResponse(content=accumulated_text)
                    yield StreamChunk(type=ChunkType.FINAL, response=final)
                    continue

                content = delta.get("content")
                tool_calls = delta.get("tool_calls")

                if tool_calls:
                    tc = tool_calls[0]
                    func = tc.get("function", {})
                    idx = tc.get("index", len(tool_blocks))
                    name = func.get("name", "")
                    args_str = func.get("arguments", "")

                    # Track new tool block
                    if idx >= len(tool_blocks):
                        tool_blocks.append({"name": name, "args": args_str})
                        current_tool_idx = idx
                    elif idx < len(tool_blocks):
                        tool_blocks[idx]["args"] += args_str
                        current_tool_idx = idx

                    yield StreamChunk(
                        type=ChunkType.TOOL_CALL_DELTA,
                        content=f"{name}({args_str})",
                    )
                elif content:
                    accumulated_text += content
                    yield StreamChunk(type=ChunkType.TEXT_DELTA, content=content)

    async def get_context_length(self) -> int | None:
        """Query the backend for its configured context window size."""
        try:
            resp = await self._client.get("/v1/models")
            resp.raise_for_status()
            return None
        except Exception:
            return None

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parse a non-streaming OpenAI response."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        # Extract usage — map OpenAI keys to Anthropic-style
        usage_data = data.get("usage", {})
        if usage_data:
            self.last_usage[str(self._slot_id)] = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        tool_calls = message.get("tool_calls")
        if tool_calls:
            result = []
            for tc in tool_calls:
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}
                result.append(ToolCall(tool=func.get("name", ""), args=args))
            return result

        content = message.get("content", "")
        return TextResponse(content=content or "")

    def _parse_stream_chunk(self, data: dict[str, Any]) -> StreamChunk | None:
        """Parse a streaming SSE chunk."""
        # Track usage even when choices is empty (usage-only chunk)
        usage_data = data.get("usage")
        if usage_data:
            self.last_usage[str(self._slot_id)] = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        choices = data.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta = choice.get("delta", {})

        finish = choice.get("finish_reason")
        if finish:
            # Build a FINAL chunk from accumulated content
            return StreamChunk(type=ChunkType.FINAL, response=TextResponse(content=""))

        content = delta.get("content")
        tool_calls = delta.get("tool_calls")

        if tool_calls:
            tc = tool_calls[0]
            func = tc.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "")
            return StreamChunk(
                type=ChunkType.TOOL_CALL_DELTA,
                content=f"{name}({args_str})",
            )

        if content:
            return StreamChunk(type=ChunkType.TEXT_DELTA, content=content)

        return None

    # ── HTTP aliases (SDK-free pass-through) ─────────────────────

    async def send_http(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Alias for send() — explicit HTTP method for proxy mode."""
        model_kwarg = sampling.get("model") if sampling else None
        return await self.send(messages, tools=tools, sampling=sampling, max_tokens=max_tokens, model=model_kwarg)

    def send_http_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Alias for send_stream() — explicit HTTP method for proxy mode."""
        return self.send_stream(messages, tools=tools, sampling=sampling, max_tokens=max_tokens)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
