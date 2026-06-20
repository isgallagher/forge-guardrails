# User Guide

Practical usage for the forge guardrails proxy.

---

## Starting the Proxy

### CLI

```bash
forge-proxy --backend-url http://localhost:8080 --port 8081
```

### Environment variables

```bash
export BACKEND_URL=http://localhost:8080
export BACKEND_TYPE=openai
export PORT=8081

forge-proxy
```

### Docker

```bash
docker run -p 8081:8081 \
  -e BACKEND_URL=http://host.docker.internal:8080 \
  -e BACKEND_TYPE=openai \
  ghcr.io/isgallagher/forge-guardrails:main
```

### Programmatic

```python
from forge.proxy import ProxyServer

proxy = ProxyServer(
    backend_url="http://localhost:8080",
    backend_type="openai",
    port=8081,
)
proxy.start()
# proxy.stop() when done
```

---

## Connecting a Client

Point your client at the proxy's URL:

```
http://localhost:8081/v1/chat/completions  (OpenAI format)
http://localhost:8081/v1/messages          (Anthropic format)
```

Forge accepts both OpenAI and Anthropic request formats. It probes the backend on startup to determine which formats it supports, then routes and translates automatically.

### OpenAI-compatible clients

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8081/v1",
    api_key="not-needed",  # backend handles auth
)

response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Anthropic clients

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8081/v1",
    api_key="not-needed",
)

message = client.messages.create(
    model="your-model",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=1024,
)
```

---

## Guardrails

Forge applies guardrails automatically to requests **with tools**:

| Guardrail | What it does |
|---|---|
| **Response validation** | Checks tool calls against the request's `tools` array |
| **Rescue parsing** | Extracts tool calls from malformed output (JSON fences, XML, etc.) |
| **Retry nudges** | Retries when the model returns bare text instead of a tool call |
| **Synthetic `respond` tool** | Injected so conversational turns become `respond(message="...")` calls |
| **Context compaction** | Tiered compaction when conversation approaches the context budget |

Requests **without tools** pass through directly — no guardrails, no overhead.

### The Synthetic `respond` Tool

When tools are present but the user sends a conversational message, small models must choose between calling a tool and responding with text. They frequently choose wrong.

Forge injects a `respond` tool so the model calls `respond(message="...")` instead of producing bare text. The `respond` call is stripped from the outbound response — the client sees a normal text response and never knows the tool exists.

**Why this works:** small models struggle with open-ended decisions ("tools or chat?") but are good at structured choices ("which tool?"). The respond tool converts an open-ended decision into a structured one. The model stays in tool-calling grammar at all times, which is where it performs best.

### Retry loop

When validation fails, forge retries inference up to `--max-retries` (default 3) with a corrective nudge. After exhausting retries, the last text response is returned to the client rather than an error — the client's own agentic loop can decide what to do.

### Rescue parsing

Forge recognizes tool calls in multiple formats:
- Standard OpenAI `tool_calls` schema
- JSON in code fences
- Mistral `[TOOL_CALLS]name{args}` format
- Qwen `<tool_call>...</tool_call>` XML tags

---

## Context Management

Forge manages context automatically. When the conversation approaches the configured budget, tiered compaction fires:

| Priority | Type | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| Cut first | Nudges (step, prereq, retry) | Drop | Drop | Drop |
| Cut second | Older tool results | Truncate ~200 chars | Drop | Drop |
| Cut third | Text responses | Preserved | Preserved | Drop |
| Cut fourth | Reasoning | Preserved | Preserved | Drop |
| Preserved | Tool calls | Preserved | Preserved | Preserved |
| Never cut | System, user input | Preserved | Preserved | Preserved |

**Key design choice — reasoning survives through Phase 2.** The model's chain-of-thought from step 3 is what informs decisions at step 5+. Losing raw tool results is recoverable; losing the model's interpretation of those results is not.

**Phase 3 is the emergency cutoff** — should only fire under extreme context pressure.

All compaction is deterministic text manipulation — no LLM calls, sub-millisecond.

Three built-in strategies:

- **`NoCompact`** — passthrough. Use when context is abundant or workflows are short.
- **`SlidingWindowCompact`** — keeps system prompt, original user input, and the last N iterations.
- **`TieredCompact`** (default) — three-phase escalating compaction with the priority order above.

---

## Proxy Design Boundaries

The proxy is intentionally focused: it applies response-quality guardrails without requiring workflow knowledge. The following features require workflow structure that doesn't exist in the OpenAI chat completions API:

- **Step enforcement and prerequisites.** These require workflow structure (required steps, terminal tool, tool dependencies) that doesn't exist in the OpenAI chat completions API. The proxy receives tool definitions per request but has no concept of workflow progression.

- **Max iterations.** The proxy calls `run_inference` once per request. Each call is bounded at `max_retries + 1` LLM attempts (default 4). There is no outer loop — a runaway model cannot loop indefinitely.

- **Real streaming.** The proxy accepts `stream=true` and returns SSE events, but the full inference completes before SSE conversion. Token-by-token streaming during inference would require validating partial responses, which is incompatible with guardrails that need complete responses (rescue parsing, retry nudges). The guardrail-first design is the proxy's value proposition.

- **Context threshold warnings.** The proxy is stateless — the client sends the full conversation history in every request and decides what to include. Context pressure is the client's concern. Compaction still fires when the budget is exceeded.

- **Cancellation on disconnect.** Client disconnects are detected but do not cancel in-flight inference. The worst case is `max_retries + 1` wasted calls (default 4) for a disconnected client.

---

## Proxy Options

| Flag | Default | Purpose |
|---|---|---|
| `--backend-url` | (required) | URL of the backend (clean base URL, no /v1) |
| `--backend-type` | `anthropic` | Backend format: `anthropic` or `openai` |
| `--max-retries N` | 3 | Retry budget per validation failure |
| `--no-rescue` | (on) | Disable rescue parsing |
| `--serialize` / `--no-serialize` | auto | Force request serialization |
| `--host` | `127.0.0.1` | Proxy listen host |
| `--port` | `8081` | Proxy listen port |
| `--log-level` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Health Check

```bash
curl http://localhost:8081/health
# {"status": "ok"}
```

## Models Endpoint

```bash
curl http://localhost:8081/v1/models
```

Returns the backend's available models. Forge fetches from the backend and converts the format if needed (e.g., OpenAI → Anthropic format when the client requests Anthropic).
