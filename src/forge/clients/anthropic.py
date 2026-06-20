"""Anthropic API client adapter for frontier model baselines.

Translates between forge's OpenAI-style message format (what the runner
produces) and Anthropic's native Messages API format internally.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import anthropic
import httpx

from forge.clients.base import ChunkType, LLMClient, StreamChunk, TokenUsage
from forge.core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec
from forge.errors import BackendError

log = logging.getLogger(__name__)


class AnthropicClient:
    """Anthropic Messages API client for Claude models.

    Uses the official anthropic SDK.  The runner serializes messages in
    OpenAI format (``api_format = "openai"``); this client converts them
    to Anthropic format before each API call.
    """

    api_format: str = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 8192,
        timeout: float = 300.0,
        max_retries: int = 3,
        tool_choice: str | None = None,
        recommended_sampling: bool = False,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.backend_url = base_url
        self._models_format = "anthropic"
        self._tool_choice = tool_choice  # "auto", "any", or None (default=auto)
        # Accepted for API symmetry across clients but currently a no-op:
        # AnthropicClient does not expose sampling kwargs through forge today.
        # The Anthropic SDK manages sampling internally.
        if recommended_sampling:
            log.debug("AnthropicClient ignores recommended_sampling=True — no sampling kwargs are exposed.")
        self.last_usage: dict[int, TokenUsage] | None = None
        self._slot_id: int = 0

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url is not None:
            kwargs["base_url"] = base_url
            self._http = httpx.AsyncClient(
                base_url=base_url,
                headers={"anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                timeout=timeout,
            )
        else:
            # The Anthropic SDK reads ANTHROPIC_BASE_URL from env. When
            # the user does not provide an explicit base_url we must
            # suppress that env var so the SDK hits the real Anthropic
            # endpoint (not a local override).
            _saved = os.environ.pop("ANTHROPIC_BASE_URL", None)
            self._client = anthropic.AsyncAnthropic(**kwargs)
            if _saved is not None:
                os.environ["ANTHROPIC_BASE_URL"] = _saved
            self._http = None  # type: ignore[assignment]
            return
        self._client = anthropic.AsyncAnthropic(**kwargs)

    # ── Tool schema conversion ───────────────────────────────────

    @staticmethod
    def _convert_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        """ToolSpec list → Anthropic tool definitions."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.get_json_schema(),
            }
            for spec in tools
        ]

    # ── Message format conversion ────────────────────────────────

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """OpenAI-format dicts → (system_prompt, anthropic_messages).

        Handles:
        - System message extraction (→ separate ``system=`` kwarg)
        - assistant tool_calls → ``tool_use`` content blocks
        - role=tool → ``tool_result`` content blocks inside user messages
        - Unpaired tool_use (step/unknown-tool nudges) → synthetic error
          tool_results injected before the next user text
        - Consecutive same-role merging (Anthropic requires strict alternation)
        """
        system: str | None = None
        converted: list[dict[str, Any]] = []
        # Track tool_use IDs that haven't received a tool_result yet.
        pending_tool_use_ids: list[str] = []

        for msg in messages:
            role = msg["role"]

            if role == "system":
                system = msg["content"]
                continue

            if role == "assistant":
                if "tool_calls" in msg:
                    blocks: list[dict[str, Any]] = []
                    content = msg.get("content", "")
                    if content:
                        blocks.append({"type": "text", "text": content})
                    for tc in msg["tool_calls"]:
                        func = tc["function"]
                        args = func.get("arguments", "{}")
                        if isinstance(args, str):
                            args = json.loads(args)
                        tc_id = tc.get("id", f"toolu_{len(converted)}")
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc_id,
                                "name": func["name"],
                                "input": args,
                            }
                        )
                        pending_tool_use_ids.append(tc_id)
                    converted.append({"role": "assistant", "content": blocks})
                else:
                    converted.append(
                        {
                            "role": "assistant",
                            "content": msg.get("content", ""),
                        }
                    )
                continue

            if role == "tool":
                tc_id = msg.get("tool_call_id", "unknown")
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tc_id,
                    "content": msg.get("content", ""),
                }
                if tc_id in pending_tool_use_ids:
                    pending_tool_use_ids.remove(tc_id)
                converted.append({"role": "user", "content": [block]})
                continue

            if role == "user":
                # Inject error tool_results for any unpaired tool_use blocks
                # (e.g. step nudge or unknown-tool nudge — tool was never executed).
                if pending_tool_use_ids:
                    blocks = []
                    for tc_id in pending_tool_use_ids:
                        blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tc_id,
                                "content": "Not executed.",
                                "is_error": True,
                            }
                        )
                    pending_tool_use_ids.clear()
                    text = msg.get("content", "")
                    if text:
                        blocks.append({"type": "text", "text": text})
                    converted.append({"role": "user", "content": blocks})
                else:
                    converted.append(
                        {
                            "role": "user",
                            "content": msg.get("content", ""),
                        }
                    )
                continue

        # Merge consecutive same-role messages (Anthropic requires strict
        # user/assistant alternation).
        merged: list[dict[str, Any]] = []
        for msg in converted:
            if merged and merged[-1]["role"] == msg["role"]:
                prev_content = merged[-1]["content"]
                curr_content = msg["content"]
                # Normalise to list-of-blocks
                if isinstance(prev_content, str):
                    prev_blocks = [{"type": "text", "text": prev_content}]
                else:
                    prev_blocks = list(prev_content)
                if isinstance(curr_content, str):
                    curr_blocks = [{"type": "text", "text": curr_content}]
                else:
                    curr_blocks = list(curr_content)
                merged[-1] = {
                    "role": msg["role"],
                    "content": prev_blocks + curr_blocks,
                }
            else:
                merged.append(msg)

        return system, merged

    # ── Response parsing ─────────────────────────────────────────

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        """Anthropic Message → list[ToolCall] or TextResponse."""
        tool_uses: list[Any] = []
        text_parts: list[str] = []

        for block in response.content:
            if block.type == "tool_use":
                tool_uses.append(block)
            elif block.type == "text":
                text_parts.append(block.text)

        if tool_uses:
            reasoning = "\n".join(text_parts) if text_parts else None
            return [
                ToolCall(
                    tool=tu.name,
                    args=dict(tu.input),
                    reasoning=reasoning if i == 0 else None,
                )
                for i, tu in enumerate(tool_uses)
            ]
        return TextResponse(content="\n".join(text_parts))

    # ── API methods ──────────────────────────────────────────────

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Build kwargs dict for messages.create / messages.stream."""
        system, converted = self._convert_messages(messages)
        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": converted,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            if self._tool_choice:
                kwargs["tool_choice"] = {"type": self._tool_choice}
        return kwargs

    async def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send messages via the Anthropic Messages API.

        ``sampling`` is accepted for protocol symmetry but ignored —
        AnthropicClient does not currently expose sampling kwargs through
        forge.
        """
        model = sampling.get("model") if sampling else None
        kwargs = self._build_kwargs(messages, tools, max_tokens=max_tokens, model=model)
        try:
            response = await self._client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            raise BackendError(getattr(exc, "status_code", 0), str(exc)) from exc
        if response.usage is not None:
            self.last_usage = {
                self._slot_id: TokenUsage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                )
            }
        return self._parse_response(response)

    async def send_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream via the Anthropic Messages API.

        ``sampling`` is accepted for protocol symmetry but ignored.
        """
        model = sampling.get("model") if sampling else None
        kwargs = self._build_kwargs(messages, tools, max_tokens=max_tokens, model=model)

        accumulated_text = ""
        # Track multiple tool_use blocks by index.
        tool_blocks: list[dict[str, str]] = []  # [{name, args}, ...]
        _current_tool_idx: int = -1
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            tool_blocks.append(
                                {
                                    "name": event.content_block.name,
                                    "args": "",
                                }
                            )
                            _current_tool_idx = len(tool_blocks) - 1
                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            accumulated_text += event.delta.text
                            yield StreamChunk(
                                type=ChunkType.TEXT_DELTA,
                                content=event.delta.text,
                            )
                        elif event.delta.type == "input_json_delta" and _current_tool_idx >= 0:
                            tool_blocks[_current_tool_idx]["args"] += event.delta.partial_json
                            yield StreamChunk(
                                type=ChunkType.TOOL_CALL_DELTA,
                                content=event.delta.partial_json,
                            )
                    elif event.type == "content_block_stop":
                        # Reset current tool index when a block finishes
                        _current_tool_idx = -1
                    elif event.type == "message_stop":
                        if tool_blocks:
                            reasoning = accumulated_text or None
                            final: LLMResponse = [
                                ToolCall(
                                    tool=tb["name"],
                                    args=json.loads(tb["args"]) if tb["args"] else {},
                                    reasoning=reasoning if i == 0 else None,
                                )
                                for i, tb in enumerate(tool_blocks)
                            ]
                        else:
                            final = TextResponse(content=accumulated_text)
                        yield StreamChunk(type=ChunkType.FINAL, response=final)
                # Grab usage from the final accumulated message.
                final_message = await stream.get_final_message()
                if final_message.usage is not None:
                    self.last_usage = {
                        self._slot_id: TokenUsage(
                            prompt_tokens=final_message.usage.input_tokens,
                            completion_tokens=final_message.usage.output_tokens,
                            total_tokens=final_message.usage.input_tokens + final_message.usage.output_tokens,
                        )
                    }
        except anthropic.APIError as exc:
            raise BackendError(getattr(exc, "status_code", 0), str(exc)) from exc

    async def get_context_length(self) -> int | None:
        """Claude models have 200K context."""
        return 200_000

    # ── HTTP methods (SDK-free, for proxy mode) ──────────────────

    def _build_request_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Build a raw JSON body for POST /v1/messages."""
        system, converted = self._convert_messages(messages)
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": converted,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        elif self.max_tokens:
            body["max_tokens"] = self.max_tokens
        if system:
            body["system"] = system
        if tools:
            body["tools"] = self._convert_tools(tools)
            if self._tool_choice:
                body["tool_choice"] = {"type": self._tool_choice}
        return body

    @staticmethod
    def _parse_http_response(data: dict[str, Any]) -> LLMResponse:
        """Parse a raw Anthropic JSON response into LLMResponse."""
        tool_uses: list[dict[str, Any]] = []
        text_parts: list[str] = []

        for block in data.get("content", []):
            if block.get("type") == "tool_use":
                tool_uses.append(block)
            elif block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        if tool_uses:
            reasoning = "\n".join(text_parts) if text_parts else None
            return [
                ToolCall(
                    tool=tu["name"],
                    args=tu.get("input", {}),
                    reasoning=reasoning if i == 0 else None,
                )
                for i, tu in enumerate(tool_uses)
            ]
        return TextResponse(content="\n".join(text_parts))

    async def send_http(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send messages via direct HTTP to the Anthropic Messages API.

        Bypasses the SDK entirely — used in proxy mode where the backend
        handles authentication and no API key is available.
        """
        if sampling:
            log.debug(
                "AnthropicClient.send_http ignores per-call sampling overrides: %s",
                sorted(sampling.keys()),
            )
        model = sampling.get("model") if sampling else None
        body = self._build_request_body(messages, tools, max_tokens=max_tokens, model=model)

        resp = await self._http.post("/v1/messages", json=body)
        if resp.status_code != 200:
            raise BackendError(resp.status_code, resp.text)

        data = resp.json()
        usage = data.get("usage", {})
        if self.last_usage is None:
            self.last_usage = {}
        self.last_usage[self._slot_id] = TokenUsage(
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        )
        return self._parse_http_response(data)

    async def send_http_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream via direct HTTP to the Anthropic Messages API.

        Bypasses the SDK entirely — used in proxy mode where the backend
        handles authentication and no API key is available.
        """
        model = sampling.get("model") if sampling else None
        body = self._build_request_body(messages, tools, max_tokens=max_tokens, model=model)
        body["stream"] = True

        accumulated_text = ""
        tool_blocks: list[dict[str, str]] = []
        current_tool_idx: int = -1

        async with self._http.stream("POST", "/v1/messages", json=body) as resp:
            if resp.status_code != 200:
                error_text = await resp.aread()
                raise BackendError(resp.status_code, error_text.decode("utf-8", errors="replace"))

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if not payload:
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")

                if event_type == "content_block_start":
                    cb = event.get("content_block", {})
                    if cb.get("type") == "tool_use":
                        tool_blocks.append({
                            "name": cb.get("name", ""),
                            "args": "",
                        })
                        current_tool_idx = len(tool_blocks) - 1

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        accumulated_text += text
                        yield StreamChunk(
                            type=ChunkType.TEXT_DELTA,
                            content=text,
                        )
                    elif delta.get("type") == "input_json_delta" and current_tool_idx >= 0:
                        partial = delta.get("partial_json", "")
                        tool_blocks[current_tool_idx]["args"] += partial
                        yield StreamChunk(
                            type=ChunkType.TOOL_CALL_DELTA,
                            content=partial,
                        )

                elif event_type == "content_block_stop":
                    current_tool_idx = -1

                elif event_type == "message_delta":
                    usage = event.get("usage", {})
                    if usage:
                        if self.last_usage is None:
                            self.last_usage = {}
                        self.last_usage[self._slot_id] = TokenUsage(
                            prompt_tokens=usage.get("input_tokens", 0),
                            completion_tokens=usage.get("output_tokens", 0),
                            total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                        )

                elif event_type == "message_stop":
                    if tool_blocks:
                        reasoning = accumulated_text or None
                        final: LLMResponse = [
                            ToolCall(
                                tool=tb["name"],
                                args=json.loads(tb["args"]) if tb["args"] else {},
                                reasoning=reasoning if i == 0 else None,
                            )
                            for i, tb in enumerate(tool_blocks)
                        ]
                    else:
                        final = TextResponse(content=accumulated_text)
                    yield StreamChunk(type=ChunkType.FINAL, response=final)

    # ── Pass-through methods ─────────────────────────────────────

    async def send_raw(self, body: dict[str, Any]) -> dict[str, Any]:
        """Send a raw Anthropic API body directly to the backend.

        Used for pass-through when the client sends Anthropic format
        and the backend supports Anthropic natively.
        """
        try:
            response = await self._client.messages.create(**body)
        except anthropic.APIError as exc:
            raise BackendError(getattr(exc, "status_code", 0), str(exc)) from exc

        content_blocks: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "tool_use":
                content_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": dict(block.input),
                })
            elif block.type == "text":
                content_blocks.append({"type": "text", "text": block.text})

        return {
            "id": response.id,
            "type": "message",
            "role": "assistant",
            "model": response.model,
            "content": content_blocks,
            "stop_reason": response.stop_reason,
            "stop_sequence": response.stop_sequence,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    async def send_raw_stream(
        self,
        body: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a raw Anthropic API body directly to the backend.

        Yields Anthropic-format SSE events as they arrive.
        """
        message_id: str | None = None
        model_name: str | None = None
        text_started = False
        tool_blocks: list[dict[str, str]] = []
        current_tool_idx: int = -1

        try:
            async with self._client.messages.stream(**body) as stream:
                async for event in stream:
                    if event.type == "message_start":
                        message_id = event.id
                        model_name = event.model
                        yield {
                            "type": "message_start",
                            "id": message_id,
                            "model": model_name,
                            "message": {
                                "id": message_id,
                                "type": "message",
                                "role": "assistant",
                                "model": model_name,
                                "content": [],
                                "stop_reason": "end_turn",
                                "stop_sequence": None,
                                "usage": {"input_tokens": 0, "output_tokens": 0},
                            },
                        }

                    elif event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            tool_blocks.append({"name": event.content_block.name, "args": ""})
                            current_tool_idx = len(tool_blocks) - 1
                            yield {
                                "type": "content_block_start",
                                "index": current_tool_idx,
                                "content_block": {
                                    "type": "tool_use",
                                    "name": event.content_block.name,
                                    "id": "",
                                    "input": {},
                                },
                                "id": message_id,
                                "model": model_name,
                            }
                        else:
                            if not text_started:
                                yield {
                                    "type": "content_block_start",
                                    "index": 0,
                                    "content_block": {"type": "text", "text": ""},
                                    "id": message_id,
                                    "model": model_name,
                                }
                                text_started = True

                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": event.delta.text},
                                "id": message_id,
                                "model": model_name,
                            }
                        elif event.delta.type == "input_json_delta" and current_tool_idx >= 0:
                            tool_blocks[current_tool_idx]["args"] += event.delta.partial_json
                            yield {
                                "type": "content_block_delta",
                                "index": current_tool_idx,
                                "delta": {"type": "input_json_delta", "partial_json": event.delta.partial_json},
                                "id": message_id,
                                "model": model_name,
                            }

                    elif event.type == "content_block_stop":
                        current_tool_idx = -1
                        yield {
                            "type": "content_block_stop",
                            "index": 0,
                            "id": message_id,
                            "model": model_name,
                        }

                    elif event.type == "message_stop":
                        final_msg = await stream.get_final_message()
                        yield {
                            "type": "message_stop",
                            "stop_reason": final_msg.stop_reason,
                            "id": message_id,
                            "model": model_name,
                            "usage": {
                                "input_tokens": final_msg.usage.input_tokens,
                                "output_tokens": final_msg.usage.output_tokens,
                            },
                        }

        except anthropic.APIError as exc:
            raise BackendError(getattr(exc, "status_code", 0), str(exc)) from exc

    async def aclose(self) -> None:
        """Close the underlying HTTP clients."""
        await self._client.aclose()
        if self._http is not None:
            await self._http.aclose()
