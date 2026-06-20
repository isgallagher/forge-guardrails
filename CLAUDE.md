# Forge Guardrails Proxy

## Architecture

Forge is a guardrails proxy that sits between LLM clients (Claude Code, Continue, etc.) and LLM backends (Anthropic API, llama-server, vLLM). It uses **direct HTTP pass-through** (no SDK required) so it works without API keys — the backend handles authentication.

### Request Flow

- **Same format** (OpenAI → OpenAI backend, Anthropic → Anthropic backend): direct pass-through
- **Different format**: translate using `src/forge/proxy/convert.py`
- **With tools**: always go through guardrails (validate, retry, nudge, step enforcement)
- **Without tools**: pass through directly, no guardrails

### Client Methods

- `send_http()` / `send_http_stream()`: HTTP methods used by `run_inference()` — bypass SDK entirely
- `send()` / `send_stream()`: SDK-based methods, kept for backward compatibility
- `send_raw()` / `send_raw_stream()`: Anthropic pass-through for native format

### Key Files

| File | Purpose |
|------|---------|
| `src/forge/proxy/handler.py` | HTTP request → inference pipeline bridge |
| `src/forge/core/inference.py` | `run_inference()` — compact, fold, serialize, send, validate, retry |
| `src/forge/clients/anthropic.py` | AnthropicClient with both SDK and httpx paths |
| `src/forge/clients/openai_compatible.py` | OpenAICompatibleClient (httpx only) |
| `src/forge/clients/base.py` | LLMClient protocol with send_http methods |
| `src/forge/proxy/proxy.py` | ProxyServer programmatic API |
| `src/forge/proxy/server.py` | Raw asyncio HTTP server, SSE streaming |
| `src/forge/proxy/convert.py` | OpenAI ↔ Anthropic format conversion |
| `src/forge/guardrails/` | Validation, retry nudges, step enforcement |

### Guardrails

Guardrails (validation, retry, nudges, step enforcement) always apply to requests **with tools**. Requests without tools pass through directly. The `respond` synthetic tool is injected when tools are present so the model calls `respond(message="...")` instead of producing bare text.

### Format Conversion (Encoder/Decoder Pattern)

The client adapter (`src/forge/clients/`) is the encoder boundary — it converts
backend wire format into forge's canonical types (`ToolCall`, `TextResponse`,
`TokenUsage`). The conversion module (`src/forge/proxy/convert.py`) is the
decoder boundary — it converts canonical types to the appropriate wire format.
Everything in between (inference, guardrails, context) uses canonical types only.

### Testing

Run: `python3 -m pytest --tb=short -q`
All tests mock `client.send_http()` (not `send()`) since `run_inference()` calls `send_http()`.
