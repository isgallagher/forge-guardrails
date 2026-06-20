"""ProxyServer — programmatic API for the forge proxy.

Connects to an external backend and applies forge guardrails transparently.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Literal

from forge.clients.anthropic import AnthropicClient
from forge.clients.base import LLMClient
from forge.clients.openai_compatible import OpenAICompatibleClient
from forge.context.manager import ContextManager
from forge.context.strategies import TieredCompact
from forge.proxy.server import HTTPServer

logger = logging.getLogger("forge.proxy")


class ProxyServer:
    """OpenAI-compatible proxy that applies forge guardrails transparently.

    External mode — connect to any OpenAI-compatible or Anthropic backend::

        proxy = ProxyServer(backend_url="http://localhost:8080")
        proxy.start()   # starts proxy on :8081
        proxy.stop()

    """

    def __init__(
        self,
        # Backend connection
        backend_url: str,
        backend_type: Literal["anthropic", "openai"] | None = None,
        # Proxy settings
        host: str = "127.0.0.1",
        port: int = 8081,
        serialize: bool | None = None,
        max_retries: int = 3,
        rescue_enabled: bool = True,
    ) -> None:
        """
        Args:
            backend_url: URL of the backend to proxy (e.g. http://localhost:8080).
            backend_type: Override backend API format — "anthropic" or "openai".
                When None, defaults to "openai" (the proxy speaks OpenAI to the
                backend, which works for llama.cpp, vLLM, Ollama API, etc.).
                Set to "anthropic" when the backend requires Anthropic format.
            host: Proxy listen host.
            port: Proxy listen port.
            serialize: Serialize requests via lock. None = default (False,
                allows concurrency; set True to enforce serial execution).
            max_retries: Max consecutive retries for bad LLM responses.
            rescue_enabled: Attempt rescue parsing of text responses.
        """
        self._backend_url = backend_url
        self._backend_type = backend_type or "openai"
        self._host = host
        self._port = port
        self._max_retries = max_retries
        self._rescue_enabled = rescue_enabled

        if serialize is None:
            self._serialize = False
        else:
            self._serialize = serialize

        self._http_server: HTTPServer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = False

    @property
    def url(self) -> str:
        """The proxy's base URL."""
        return f"http://{self._host}:{self._port}"

    def start(self) -> None:
        """Start the proxy.

        Blocks until the proxy is ready to accept connections.
        """
        if self._started:
            return

        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(ready,),
            daemon=True,
        )
        self._thread.start()
        ready.wait(timeout=120)

        if not self._started:
            raise RuntimeError("Proxy failed to start")

        logger.info("Proxy ready at %s", self.url)

    def stop(self) -> None:
        """Stop the proxy."""
        if not self._started or self._loop is None:
            return

        try:
            asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop).result(timeout=10)
        except Exception as e:
            logger.debug("Async stop error (ignored during shutdown): %s", e)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._started = False
        logger.info("Proxy stopped")

    def _run_loop(self, ready: threading.Event) -> None:
        """Event loop thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_start(ready))
            self._loop.run_forever()
        finally:
            self._loop.close()

    async def _async_start(self, ready: threading.Event) -> None:
        """Async startup: create client + HTTP server."""
        client: LLMClient
        context_manager: ContextManager

        if self._backend_type == "anthropic":
            client = AnthropicClient(
                model="forge",
                api_key="none",
                base_url=self._backend_url,
            )
            context_manager = ContextManager(
                strategy=TieredCompact(),
                budget_tokens=8192,
            )
        else:
            client = OpenAICompatibleClient(base_url=self._backend_url.rstrip("/"))
            context_manager = ContextManager(
                strategy=TieredCompact(),
                budget_tokens=8192,
            )

        self._http_server = HTTPServer(
            client=client,
            context_manager=context_manager,
            host=self._host,
            port=self._port,
            serialize_requests=self._serialize,
            max_retries=self._max_retries,
            rescue_enabled=self._rescue_enabled,
        )
        await self._http_server.start()
        self._started = True
        ready.set()

    async def _async_stop(self) -> None:
        """Async shutdown."""
        if self._http_server is not None:
            await self._http_server.stop()
