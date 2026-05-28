"""Raw asyncio HTTP server for the proxy.

No framework dependencies — uses asyncio.start_server directly.
Handles routing, request queuing (single-GPU serialization), health
checks, SSE streaming, and client disconnect detection.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from forge.clients.base import LLMClient
from forge.context.manager import ContextManager
from forge.proxy.handler import handle_chat_completions, handle_messages
from forge.proxy.convert import (
    anthropic_models_to_openai,
    openai_models_to_anthropic,
    ollama_models_to_anthropic,
    ollama_models_to_openai,
)

logger = logging.getLogger("forge.proxy")

# Maximum request body size (16 MB)
_MAX_BODY = 16 * 1024 * 1024


@dataclass
class _QueueItem:
    """A request waiting to be processed by the inference worker."""

    body: dict[str, Any]
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())
    cancelled: bool = False
    handler_func: Any = None


class HTTPServer:
    """Raw asyncio HTTP server with OpenAI-compatible routing."""

    def __init__(
        self,
        client: LLMClient,
        context_manager: ContextManager,
        host: str = "127.0.0.1",
        port: int = 8081,
        serialize_requests: bool = True,
        max_retries: int = 3,
        rescue_enabled: bool = True,
    ) -> None:
        self._client = client
        self._context_manager = context_manager
        self._host = host
        self._port = port
        self._max_retries = max_retries
        self._rescue_enabled = rescue_enabled
        self._server: asyncio.Server | None = None
        self._serialize = serialize_requests
        self._queue: asyncio.Queue[_QueueItem | None] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._models_cache: dict[bool, bool] = {}
        self._http = httpx.AsyncClient(timeout=10.0)

    async def start(self) -> None:
        """Start listening for connections."""
        self._shutdown_event = asyncio.Event()
        if self._serialize:
            self._worker_task = asyncio.create_task(self._inference_worker())
        self._server = await asyncio.start_server(
            self._handle_connection,
            self._host,
            self._port,
        )
        logger.info("Proxy listening on %s:%d", self._host, self._port)

    async def stop(self) -> None:
        """Stop the server gracefully."""
        # 1. Signal shutdown to all components
        if self._shutdown_event is not None:
            self._shutdown_event.set()

        # 2. Stop accepting new connections
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # 3. Unblock worker with sentinel (works even if worker is blocked on queue.get)
        if self._serialize:
            self._queue.put_nowait(None)

        # 4. Cancel and await worker
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await asyncio.wait_for(self._worker_task, timeout=5)
            except (asyncio.CancelledError, TimeoutError):
                pass
            self._worker_task = None

        # 5. Close HTTP client used for models endpoint
        await self._http.aclose()

    async def _inference_worker(self) -> None:
        """Single worker that pulls requests off the queue and processes them.

        Ensures only one inference runs at a time (single-GPU constraint).
        Exits on sentinel (None) or cancellation.
        """
        while self._shutdown_event is None or not self._shutdown_event.is_set():
            item = await self._queue.get()
            try:
                # Sentinel value signals shutdown
                if item is None:
                    break
                if item.cancelled or item.future.cancelled():
                    logger.debug("   Skipping cancelled request")
                    continue
                if item.handler_func is not None:
                    result = await item.handler_func(item.body)
                else:
                    result = await self._run_handler(item.body)
                if not item.future.done():
                    item.future.set_result(result)
            except asyncio.CancelledError:
                raise  # finally will call task_done()
            except Exception as exc:
                if not item.future.done():
                    item.future.set_result(exc)
            finally:
                if item is not None:
                    self._queue.task_done()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single HTTP connection."""
        try:
            # Read request line
            request_line = await asyncio.wait_for(
                reader.readline(),
                timeout=30.0,
            )
            if not request_line:
                return

            request_str = request_line.decode("utf-8", errors="replace").strip()
            parts = request_str.split(" ", 2)
            if len(parts) < 2:
                await self._send_error(writer, 400, "Bad request")
                return

            method, path = parts[0], parts[1]
            pure_path = path.split("?")[0]
            logger.debug(">> %s %s", method, path)

            # Read headers
            headers = await self._read_headers(reader)
            content_length = int(headers.get("content-length", "0"))

            # Read body
            body_bytes = b""
            if content_length > 0:
                if content_length > _MAX_BODY:
                    await self._send_error(writer, 413, "Request too large")
                    return
                body_bytes = await asyncio.wait_for(
                    reader.readexactly(content_length),
                    timeout=60.0,
                )

            # Route (use pure_path to ignore query strings)
            if method == "GET" and pure_path == "/health":
                await self._handle_health(writer)
            elif method == "GET" and pure_path == "/v1/models":
                await self._handle_models(writer, headers)
            elif method == "POST" and pure_path == "/v1/chat/completions":
                await self._handle_completions(writer, body_bytes)
            elif method == "POST" and pure_path == "/v1/messages":
                await self._handle_messages(writer, body_bytes)
            elif method == "OPTIONS":
                await self._send_cors_preflight(writer)
            else:
                await self._send_error(writer, 404, "Not found")

        except (TimeoutError, asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception:
            logger.exception("Unhandled error in connection handler")
            try:
                await self._send_error(writer, 500, "Internal server error")
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _read_headers(self, reader: asyncio.StreamReader) -> dict[str, str]:
        """Read HTTP headers until blank line."""
        headers: dict[str, str] = {}
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=30.0)
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded:
                break
            if ":" in decoded:
                key, value = decoded.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        return headers

    async def _handle_health(self, writer: asyncio.StreamWriter) -> None:
        """GET /health — returns OK."""
        body = json.dumps({"status": "ok"})
        await self._send_json(writer, 200, body)

    async def _handle_models(self, writer: asyncio.StreamWriter, headers: dict[str, str]) -> None:
        """GET /v1/models — probes backend, caches result, converts format."""
        want_anthropic = "anthropic-version" in headers

        # Access client attributes via getattr for safety with mocks
        backend_url = getattr(self._client, "backend_url", None)
        native_format = getattr(self._client, "_models_format", None)

        if not backend_url or not native_format:
            # Fallback: return hardcoded forge model list
            body = json.dumps({
                "object": "list",
                "data": [{"id": "forge", "object": "model"}],
            })
            await self._send_json(writer, 200, body)
            return

        cache_key = want_anthropic
        try:
            supports_natively = self._models_cache.get(cache_key)
            if supports_natively is None:
                supports_natively = await self._probe_models_format(
                    backend_url, native_format, want_anthropic,
                )
                self._models_cache[cache_key] = supports_natively

            data = await self._fetch_models(
                backend_url, native_format, want_anthropic, supports_natively,
            )
            await self._send_json(writer, 200, json.dumps(data))
        except Exception:
            logger.exception("Models endpoint fallback")
            body = json.dumps({
                "object": "list",
                "data": [{"id": "forge", "object": "model"}],
            })
            await self._send_json(writer, 200, body)

    async def _probe_models_format(
        self,
        backend_url: str,
        native_format: str,
        want_anthropic: bool,
    ) -> bool:
        """Probe the backend to see if it supports the requested format.

        Returns True if the backend supports the format natively, False otherwise.
        """
        url = self._models_endpoint(backend_url)
        headers = {}
        if want_anthropic:
            headers["anthropic-version"] = "2023-06-01"

        try:
            resp = await self._http.get(url, headers=headers or None)
            if resp.status_code == 200:
                return True
            if resp.status_code == 404:
                return False
        except Exception:
            pass
        # On error, assume it doesn't support the format → will try native
        return False

    async def _fetch_models(
        self,
        backend_url: str,
        native_format: str,
        want_anthropic: bool,
        supports_natively: bool,
    ) -> dict[str, Any]:
        """Fetch models from the backend, converting format if needed."""
        if supports_natively:
            # Fetch in requested format, passthrough
            url = self._models_endpoint(backend_url)
            headers = {}
            if want_anthropic:
                headers["anthropic-version"] = "2023-06-01"
            resp = await self._http.get(url, headers=headers or None)
            resp.raise_for_status()
            return resp.json()

        # Fetch in native format, convert
        url = self._native_models_endpoint(backend_url, native_format)
        resp = await self._http.get(url)
        resp.raise_for_status()
        native_data = resp.json()

        if native_format == "ollama":
            if want_anthropic:
                return ollama_models_to_anthropic(native_data)
            return ollama_models_to_openai(native_data)
        elif native_format == "openai":
            if want_anthropic:
                return openai_models_to_anthropic(native_data)
            return native_data
        elif native_format == "anthropic":
            if want_anthropic:
                return native_data
            return anthropic_models_to_openai(native_data)

        # Unknown format, return fallback
        return {"object": "list", "data": [{"id": "forge", "object": "model"}]}

    @staticmethod
    def _models_endpoint(backend_url: str) -> str:
        """Build the /v1/models endpoint URL for probing.

        Both Anthropic and OpenAI use /v1/models, so the same URL applies.
        """
        base = backend_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/models"
        return f"{base}/v1/models"

    @staticmethod
    def _native_models_endpoint(backend_url: str, native_format: str) -> str:
        """Build the native models endpoint URL for a given backend format."""
        base = backend_url.rstrip("/")
        if native_format == "ollama":
            return f"{base}/api/tags"
        elif native_format == "openai":
            if base.endswith("/v1"):
                return f"{base}/models"
            return f"{base}/v1/models"
        elif native_format == "anthropic":
            if base.endswith("/v1"):
                return f"{base}/models"
            return f"{base}/v1/models"
        # Default: OpenAI-style
        if base.endswith("/v1"):
            return f"{base}/models"
        return f"{base}/v1/models"

    async def _handle_completions(
        self,
        writer: asyncio.StreamWriter,
        body_bytes: bytes,
    ) -> None:
        """POST /v1/chat/completions — the main proxy endpoint."""
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            await self._send_error(writer, 400, "Invalid JSON")
            return

        is_stream = body.get("stream", False)
        msg_count = len(body.get("messages", []))
        tool_count = len(body.get("tools", []))
        logger.debug(
            "   stream=%s messages=%d tools=%d model=%s",
            is_stream,
            msg_count,
            tool_count,
            body.get("model", "?"),
        )

        if self._serialize:
            # Queue the request and wait for the worker to process it
            item = _QueueItem(body=body)
            queue_depth = self._queue.qsize()
            if queue_depth > 0:
                logger.debug("   Queued (depth=%d)", queue_depth + 1)

            # For streaming requests, send SSE headers immediately so the
            # client knows we're alive while waiting in the queue
            if is_stream:
                await self._send_sse_header(writer)

            self._queue.put_nowait(item)

            # Wait for result, monitoring for client disconnect
            result = await self._await_with_disconnect(item, writer)
        else:
            if is_stream:
                await self._send_sse_header(writer)
            result = await self._run_handler(body)

        if result is None:
            # Client disconnected
            logger.debug("<< Client disconnected, discarding result")
            return

        if isinstance(result, Exception):
            error_msg = str(result)
            logger.debug("<< ERROR: %s", error_msg[:120])
            if is_stream:
                await self._send_sse_body(writer, [{"error": error_msg}])
            else:
                await self._send_error(writer, 502, error_msg)
            return

        if is_stream:
            logger.debug("<< SSE %d events", len(result))
            await self._send_sse_body(writer, result)
        else:
            logger.debug("<< JSON 200")
            await self._send_json(writer, 200, json.dumps(result))

    async def _handle_messages(
        self,
        writer: asyncio.StreamWriter,
        body_bytes: bytes,
    ) -> None:
        """POST /v1/messages — Anthropic-compatible endpoint."""
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            await self._send_error(writer, 400, "Invalid JSON")
            return

        is_stream = body.get("stream", False)
        msg_count = len(body.get("messages", []))
        tool_count = len(body.get("tools", []))
        logger.debug(
            "   [anthropic] stream=%s messages=%d tools=%d model=%s",
            is_stream,
            msg_count,
            tool_count,
            body.get("model", "?"),
        )

        if self._serialize:
            item = _QueueItem(body=body, handler_func=self._run_anthropic_handler)
            queue_depth = self._queue.qsize()
            if queue_depth > 0:
                logger.debug("   Queued (depth=%d)", queue_depth + 1)

            if is_stream:
                await self._send_sse_header(writer)

            self._queue.put_nowait(item)
            result = await self._await_with_disconnect(item, writer)
        else:
            if is_stream:
                await self._send_sse_header(writer)
            result = await self._run_anthropic_handler(body)

        if result is None:
            logger.debug("<< Client disconnected, discarding result")
            return

        if isinstance(result, Exception):
            error_msg = str(result)
            logger.debug("<< ERROR: %s", error_msg[:120])
            if is_stream:
                await self._send_sse_body(writer, [{"error": error_msg}])
            else:
                await self._send_error(writer, 502, error_msg)
            return

        if is_stream:
            logger.debug("<< SSE %d events", len(result))
            await self._send_sse_body(writer, result)
        else:
            logger.debug("<< JSON 200")
            await self._send_json(writer, 200, json.dumps(result))

    async def _run_anthropic_handler(
        self,
        body: dict[str, Any],
    ) -> dict[str, Any] | list[dict[str, Any]] | Exception:
        """Run the Anthropic handler, catching errors."""
        try:
            return await handle_messages(
                body=body,
                client=self._client,
                context_manager=self._context_manager,
                max_retries=self._max_retries,
                rescue_enabled=self._rescue_enabled,
                anthropic_backend=True,
            )
        except Exception as exc:
            logger.exception("Handler error")
            return exc

    async def _await_with_disconnect(
        self,
        item: _QueueItem,
        writer: asyncio.StreamWriter,
    ) -> dict[str, Any] | list[dict[str, Any]] | Exception | None:
        """Wait for a queued item's result, checking for client disconnect.

        Returns None if the client disconnected or server is shutting down.
        """
        while not item.future.done():
            if writer.is_closing():
                item.cancelled = True
                logger.debug("   Client disconnected, cancelling queued request")
                return None
            # Exit early if server is shutting down
            if self._shutdown_event is not None and self._shutdown_event.is_set():
                logger.debug("   Server shutting down, discarding in-flight result")
                return None
            done, _ = await asyncio.wait(
                [item.future],
                timeout=1.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if done:
                break
        return item.future.result()

    async def _run_handler(
        self,
        body: dict[str, Any],
    ) -> dict[str, Any] | list[dict[str, Any]] | Exception:
        """Run the handler, catching errors."""
        try:
            return await handle_chat_completions(
                body=body,
                client=self._client,
                context_manager=self._context_manager,
                max_retries=self._max_retries,
                rescue_enabled=self._rescue_enabled,
                anthropic_backend=False,
            )
        except Exception as exc:
            logger.exception("Handler error")
            return exc

    async def _send_json(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        body: str,
    ) -> None:
        """Send a JSON HTTP response."""
        response = (
            f"HTTP/1.1 {status} {_status_text(status)}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body.encode())}\r\n"
            f"Connection: close\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"\r\n"
            f"{body}"
        )
        writer.write(response.encode())
        await writer.drain()

    async def _send_sse_header(self, writer: asyncio.StreamWriter) -> None:
        """Send SSE response headers immediately (before body is ready)."""
        header = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Transfer-Encoding: chunked\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: keep-alive\r\n"
            "\r\n"
        )
        writer.write(header.encode())
        await writer.drain()

    async def _send_sse_body(
        self,
        writer: asyncio.StreamWriter,
        events: list[dict[str, Any]],
    ) -> None:
        """Send SSE event data and terminator. Headers must already be sent."""
        for event in events:
            if writer.is_closing():
                return
            event_type = event.get("type", "")
            data_line = f"data: {json.dumps(event)}\n"
            event_line = f"event: {event_type}\n" if event_type else ""
            body = f"{event_line}{data_line}\n".encode()
            writer.write(f"{len(body):x}\r\n".encode() + body + b"\r\n")
            await writer.drain()

        done = b"data: [DONE]\n\n"
        writer.write(f"{len(done):x}\r\n".encode() + done + b"\r\n")
        # Terminating zero-length chunk
        writer.write(b"0\r\n\r\n")
        await writer.drain()
        logger.debug("<< SSE complete, [DONE] sent")

    async def _send_error(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        message: str,
    ) -> None:
        """Send an error JSON response."""
        body = json.dumps({"error": {"message": message, "type": "proxy_error"}})
        await self._send_json(writer, status, body)

    async def _send_cors_preflight(self, writer: asyncio.StreamWriter) -> None:
        """Handle CORS preflight."""
        response = (
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        writer.write(response.encode())
        await writer.drain()


def _status_text(code: int) -> str:
    """HTTP status code to text."""
    return {
        200: "OK",
        204: "No Content",
        400: "Bad Request",
        404: "Not Found",
        413: "Payload Too Large",
        500: "Internal Server Error",
        502: "Bad Gateway",
    }.get(code, "Error")
