# forge

[![PyPI](https://img.shields.io/pypi/v/forge-guardrails.svg)](https://pypi.org/project/forge-guardrails/)
[![Tests](https://github.com/isgallagher/forge-guardrails/actions/workflows/tests.yml/badge.svg)](https://github.com/isgallagher/forge-guardrails/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/isgallagher/forge-guardrails/branch/main/graph/badge.svg)](https://codecov.io/gh/isgallagher/forge-guardrails)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A transparent guardrails proxy for LLM tool-calling. Forge sits between any client (Claude Code, Continue, opencode, etc.) and any backend (Anthropic API, llama-server, vLLM, Ollama) and applies response validation, rescue parsing, retry nudges, and context compaction — so the client thinks it's talking to a more reliable model.

Forge takes an 8B local model from single digits to 84% across forge's eval suite — and even lifts Sonnet 4.6 from 85% to 98% on the same workload.

**What forge isn't:**
- **Not an agent orchestrator.** Forge sits inside one agentic loop and makes its tool calls reliable. Multi-agent graphs, DAG planners, and cross-agent coordination are out of scope.
- **Not a coding harness.** Forge is domain-agnostic. It lifts your existing agent harness with guardrails — no rewrite.

**Architecture:**

Forge is a guardrails layer that sits between clients and LLM backends. It uses direct HTTP pass-through (no SDK required) so it works without API keys — the backend handles authentication.

- **Same format** (OpenAI client → OpenAI backend, Anthropic → Anthropic): requests pass through directly
- **Different format**: forge translates using conversion functions in `convert.py`
- **With tools**: all requests go through guardrails (validate, retry, nudge, step enforcement)
- **Without tools**: requests pass through to the backend directly, skipping guardrails

No API keys needed — the backend handles authentication. Forge uses `httpx` for all HTTP communication.

**How it works:**

On every `POST /v1/chat/completions`, forge applies (in order):

1. **Response validation** — each tool call is checked against the `tools` array in the request. Calls to unknown tool names or with malformed shapes are caught.
2. **Rescue parsing** — when the model emits tool calls in the wrong format (JSON in a code fence, Mistral's `[TOOL_CALLS]name{args}`, Qwen's `<tool_call>...</tool_call>` XML), forge extracts the structured call.
3. **Retry loop** — if validation fails, forge retries inference up to `--max-retries` (default 3) with a corrective tool-result message.
4. **Synthetic `respond` tool injection** — when tools are present, forge injects a `respond` tool the model calls instead of producing bare text. The call is stripped from the outbound response.

## Install

```bash
pip install forge-guardrails
```

## Quick Start

Start your backend (Anthropic API, llama-server, vLLM, etc.):

```bash
# Example: llama-server
llama-server -m path/to/model.gguf --jinja -ngl 999 --port 8080
```

Start the forge proxy:

```bash
forge-proxy --backend-url http://localhost:8080/v1 --port 8081
```

Then configure your client to use `http://localhost:8081/v1` as the API base URL.

### Programmatic usage

```python
from forge.proxy import ProxyServer

proxy = ProxyServer(
    backend_url="http://localhost:8080/v1",
    backend_type="openai",  # or "anthropic"
    port=8081,
)
proxy.start()
# proxy.stop() when done
```

### Proxy options

| Flag | Default | Purpose |
|---|---|---|
| `--backend-url` | (required) | URL of the backend to proxy |
| `--backend-type` | `anthropic` | Backend API format: `anthropic` or `openai` |
| `--max-retries N` | 3 | Retry budget per validation failure |
| `--no-rescue` | (rescue on) | Disable rescue parsing |
| `--serialize` / `--no-serialize` | auto | Force request serialization |
| `--host` | `127.0.0.1` | Proxy listen host |
| `--port` | `8081` | Proxy listen port |

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | — | Backend URL (required) |
| `BACKEND_TYPE` | `anthropic` | `anthropic` or `openai` |
| `HOST` | `127.0.0.1` | Proxy bind host |
| `PORT` | `8081` | Proxy listen port |
| `MAX_RETRIES` | `3` | Retry budget per validation failure |
| `RESCUE` | `true` | Set to `false` to disable rescue parsing |
| `SERIALIZE` | — | Set to `true` to force request serialization |
| `VERBOSE` | `false` | Set to `true` for debug logging |

### Docker

```bash
docker run -p 8081:8081 \
  -e BACKEND_URL=http://host.docker.internal:8080/v1 \
  -e BACKEND_TYPE=openai \
  ghcr.io/isgallagher/forge-guardrails:main
```

## Project Structure

```
src/forge/
  errors.py            # ForgeError hierarchy
  clients/
    base.py            # ChunkType, StreamChunk, LLMClient protocol
    anthropic.py       # AnthropicClient
    openai_compatible.py  # OpenAICompatibleClient (llama-server, vLLM, etc.)
  context/
    manager.py         # ContextManager, CompactEvent
    strategies.py      # CompactStrategy, NoCompact, TieredCompact, SlidingWindowCompact
  guardrails/
    guardrails.py      # Guardrails facade
    nudge.py           # Nudge dataclass
    response_validator.py  # ResponseValidator, ValidationResult
    step_enforcer.py   # StepEnforcer, StepCheck
    error_tracker.py   # ErrorTracker
  proxy/
    __main__.py        # CLI entry point: forge-proxy
    proxy.py           # ProxyServer — programmatic start/stop API
    server.py          # Raw asyncio HTTP server, SSE streaming
    handler.py         # Request handler — bridge between HTTP and run_inference
    convert.py         # OpenAI ↔ Anthropic message/response conversion
  core/
    messages.py        # Message, MessageRole, MessageType, MessageMeta
    workflow.py        # ToolSpec, ToolDef, ToolCall, TextResponse, Workflow
    inference.py       # run_inference() — compact, fold, validate, retry
    steps.py           # StepTracker
  prompts/
    templates.py       # Tool prompt builders
    nudges.py          # Retry and step-enforcement nudge templates
  tools/
    respond.py         # Synthetic respond tool
tests/
  unit/                # Deterministic tests — no LLM backend required
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — Full design document
- [User Guide](docs/USER_GUIDE.md) — Usage patterns, context management, guardrails
- [Contributing](CONTRIBUTING.md) — How to set up, test, and contribute

## Paper

The forge guardrail framework and ablation study are published as:

> Zambelli, A. *Forge: A Reliability Layer for Self-Hosted LLM Tool-Calling.*
> [https://doi.org/10.1145/3786335.3813193](https://doi.org/10.1145/3786335.3813193)

A pre-publication preprint is available at [docs/forge_ieee_preprint.pdf](docs/forge_ieee_preprint.pdf).

## License

[MIT](LICENSE) — Copyright (c) 2025-2026 Antoine Zambelli
