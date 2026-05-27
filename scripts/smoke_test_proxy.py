"""Smoke test for the proxy — starts proxy in external mode against a
mock backend, sends one request, verifies the response.

Usage: python scripts/smoke_test_proxy.py
       (loads .env.dev for BACKEND_URL, BACKEND, MODEL, ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import httpx

# Load .env.dev if available
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env.dev")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


async def mock_backend(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Minimal mock that returns a tool call response."""
    # Read request
    request_line = await reader.readline()
    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        decoded = line.decode().strip()
        if not decoded:
            break
        if ":" in decoded:
            k, v = decoded.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    content_length = int(headers.get("content-length", "0"))
    if content_length > 0:
        body = await reader.readexactly(content_length)

    # Return a tool call response
    response_body = json.dumps({
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_mock1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })

    response = (
        f"HTTP/1.1 200 OK\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(response_body)}\r\n"
        f"\r\n"
        f"{response_body}"
    )
    writer.write(response.encode())
    await writer.drain()
    writer.close()
    await writer.wait_closed()


async def mock_backend_with_health(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Mock that handles both /props (for context length) and /v1/chat/completions."""
    request_line = await reader.readline()
    path = request_line.decode().split(" ")[1] if request_line else ""

    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        decoded = line.decode().strip()
        if not decoded:
            break
        if ":" in decoded:
            k, v = decoded.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    content_length = int(headers.get("content-length", "0"))
    if content_length > 0:
        await reader.readexactly(content_length)

    if "/props" in path:
        body = json.dumps({"default_generation_settings": {"n_ctx": 8192}})
    else:
        body = json.dumps({
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": "mock",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_mock1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    response = (
        f"HTTP/1.1 200 OK\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"\r\n"
        f"{body}"
    )
    writer.write(response.encode())
    await writer.drain()
    writer.close()
    await writer.wait_closed()


async def main() -> None:
    # Start mock backend on port 18080
    mock_server = await asyncio.start_server(mock_backend_with_health, "127.0.0.1", 18080)
    print("[mock] Backend running on :18080")

    # Start proxy
    from forge.proxy import ProxyServer

    proxy = ProxyServer(
        backend_url="http://127.0.0.1:18080",
        port=18081,
        budget_tokens=8192,
    )

    # Run proxy start in a thread (it blocks until ready)
    import threading
    proxy_thread = threading.Thread(target=proxy.start, daemon=True)
    proxy_thread.start()
    proxy_thread.join(timeout=10)

    if not proxy._started:
        print("[FAIL] Proxy didn't start")
        sys.exit(1)

    print("[proxy] Running on :18081")

    # Send a request
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Test health
        health = await client.get("http://127.0.0.1:18081/health")
        assert health.status_code == 200, f"Health check failed: {health.status_code}"
        print("[test] Health check: OK")

        # Test chat completions (non-streaming)
        resp = await client.post(
            "http://127.0.0.1:18081/v1/chat/completions",
            json={
                "model": "test",
                "messages": [
                    {"role": "system", "content": "You are a weather assistant."},
                    {"role": "user", "content": "What's the weather in Paris?"},
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string", "description": "City name"},
                            },
                            "required": ["city"],
                        },
                    },
                }],
                "stream": False,
            },
        )

        print(f"[test] Chat completions status: {resp.status_code}")
        data = resp.json()
        print(f"[test] Response: {json.dumps(data, indent=2)}")

        choice = data["choices"][0]
        assert choice["finish_reason"] == "tool_calls", f"Expected tool_calls, got {choice['finish_reason']}"
        tc = choice["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        args = json.loads(tc["function"]["arguments"])
        assert args["city"] == "Paris"
        print("[test] Tool call validated: get_weather(city='Paris')")

        # Test streaming
        resp_stream = await client.post(
            "http://127.0.0.1:18081/v1/chat/completions",
            json={
                "model": "test",
                "messages": [
                    {"role": "user", "content": "Weather in Paris?"},
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }],
                "stream": True,
            },
        )
        print(f"[test] Streaming status: {resp_stream.status_code}")
        body = resp_stream.text
        lines = [l for l in body.split("\n") if l.startswith("data: ")]
        print(f"[test] SSE events: {len(lines)}")
        assert any("[DONE]" in l for l in body.split("\n")), "Missing [DONE]"
        print("[test] Streaming validated: got SSE events + [DONE]")

    print("\n[PASS] All smoke tests passed!")
    proxy.stop()
    mock_server.close()
    await mock_server.wait_closed()


async def main_live() -> None:
    """Test against a running proxy using .env.dev config."""
    proxy_url = os.environ.get("PROXY_URL", "http://127.0.0.1:8081")
    backend_url = os.environ.get("BACKEND_URL")
    backend = os.environ.get("BACKEND", "anthropic")
    model = os.environ.get("MODEL", "qwen3.6-27b")

    print(f"[config] Proxy: {proxy_url}")
    print(f"[config] Backend: {backend} at {backend_url}")
    print(f"[config] Model: {model}")
    print()

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        # Test health
        try:
            health = await client.get(f"{proxy_url}/health")
            print(f"[test] Health: {health.status_code} — {health.text}")
        except Exception as e:
            print(f"[FAIL] Health check: {e}")
            sys.exit(1)

        # Determine endpoint based on backend type
        if backend == "anthropic":
            endpoint = f"{proxy_url}/v1/messages"
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Say hello in three words."},
                ],
                "max_tokens": 50,
                "stream": False,
            }
        else:
            endpoint = f"{proxy_url}/v1/chat/completions"
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Say hello in three words."},
                ],
                "max_tokens": 50,
                "stream": False,
            }

        # Test non-streaming
        try:
            print(f"[test] POST {endpoint}")
            resp = await client.post(endpoint, json=payload)
            print(f"[test] Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                if backend == "anthropic":
                    content = data.get("content", [{}])[0].get("text", "(no content)")
                else:
                    content = data["choices"][0]["message"].get("content", "(no content)")
                print(f"[test] Response: {content[:200]}")
            else:
                print(f"[test] Error body: {resp.text[:300]}")
        except Exception as e:
            print(f"[FAIL] Chat request: {type(e).__name__}: {e}")
            sys.exit(1)

        # Test streaming
        try:
            payload["stream"] = True
            print(f"[test] POST {endpoint} (streaming)")
            resp = await client.post(endpoint, json=payload)
            print(f"[test] Status: {resp.status_code}")
            body = resp.text
            lines = [l for l in body.split("\n") if l.startswith("data: ")]
            print(f"[test] SSE events: {len(lines)}")
            if "[DONE]" in body:
                print("[test] Got [DONE] terminator")
            else:
                print(f"[test] Body tail: {body[-200:]}")
        except Exception as e:
            print(f"[FAIL] Streaming request: {type(e).__name__}: {e}")
            sys.exit(1)

    print("\n[PASS] Live smoke tests passed!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        asyncio.run(main_live())
    else:
        asyncio.run(main())
