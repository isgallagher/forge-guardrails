# forge-guardrails

[![PyPI](https://img.shields.io/pypi/v/forge-guardrails.svg)](https://pypi.org/project/forge-guardrails/)
[![Tests](https://github.com/isgallagher/forge-guardrails/actions/workflows/tests.yml/badge.svg)](https://github.com/isgallagher/forge-guardrails/actions/workflows/tests.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A transparent guardrails proxy for LLM tool-calling. Forge sits between any client (Claude Code, Continue, opencode, etc.) and any backend (Anthropic API, llama-server, vLLM, Ollama) and applies response validation, rescue parsing, retry nudges, and context compaction — so the client thinks it's talking to a more reliable model.

**What forge isn't:**
- **Not an inference engine.** Forge does not run models. All inference is handled by a separate backend that Forge proxies to.
- **Not an agent orchestrator.** Forge sits inside one agentic loop and makes its tool calls reliable. Multi-agent graphs, DAG planners, and cross-agent coordination are out of scope.
- **Not a coding harness.** Forge is domain-agnostic. It lifts your existing agent harness with guardrails — no rewrite.

## How It Works

Forge is a drop-in proxy that speaks OpenAI-compatible and Anthropic APIs. It automatically detects which formats your backend supports and translates between client and backend formats as needed.

```
Client (Claude Code, Continue, etc.)
  │
  ▼
Forge Proxy (port 8081)
  │  • Validates tool calls against request schema
  │  • Rescues malformed tool calls (JSON fences, XML, etc.)
  │  • Retries with corrective nudges on validation failure
  │  • Injects synthetic `respond` tool for conversational turns
  │  • Compacts context when approaching budget
  │  • Translates OpenAI ↔ Anthropic format automatically
  ▼
Backend (llama-server, vLLM, Anthropic API, etc.)
```

### Request Flow

- **Same format** (OpenAI client → OpenAI backend, Anthropic → Anthropic): requests pass through directly
- **Different format**: forge translates using conversion functions
- **With tools**: all requests go through guardrails (validate, retry, nudge, step enforcement)
- **Without tools**: requests pass through to the backend directly, skipping guardrails

### Backend Detection

Forge probes the backend on startup to determine which API formats it supports (OpenAI, Anthropic, or both). It then routes and translates requests accordingly — no configuration needed.

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
forge-proxy --backend-url http://localhost:8080 --port 8081
```

Then configure your client to use `http://localhost:8081/v1` as the API base URL.

### Programmatic usage

```python
from forge.proxy import ProxyServer

proxy = ProxyServer(
    backend_url="http://localhost:8080",
    backend_type="openai",  # or "anthropic"
    port=8081,
)
proxy.start()
# proxy.stop() when done
```

### Proxy options

| Flag | Default | Purpose |
|---|---|---|
| `--backend-url` | (required) | URL of the backend to proxy (clean base URL, no /v1 suffix) |
| `--backend-type` | `anthropic` | Backend API format: `anthropic` or `openai` |
| `--max-retries N` | 3 | Retry budget per validation failure |
| `--no-rescue` | (rescue on) | Disable rescue parsing |
| `--serialize` / `--no-serialize` | auto | Force request serialization |
| `--host` | `127.0.0.1` | Proxy listen host |
| `--port` | `8081` | Proxy listen port |

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | — | Backend URL (clean base URL, no /v1 suffix) |
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
  -e BACKEND_URL=http://host.docker.internal:8080 \
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

- [Architecture](docs/ARCHITECTURE.md) — Design principles and guardrail rationale
- [Contributing](CONTRIBUTING.md) — How to set up, test, and contribute

## Paper

The forge guardrail framework and ablation study are published as:

> Zambelli, A. *Forge: A Reliability Layer for Self-Hosted LLM Tool-Calling.*
> [https://doi.org/10.1145/3786335.3813193](https://doi.org/10.1145/3786335.3813193)

## License

[MIT](LICENSE) — Copyright (c) 2025-2026 Antoine Zambelli
