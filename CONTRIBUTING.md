# Contributing to forge-guardrails

Thanks for your interest in contributing. This guide covers how to get set up, run tests, and where to look when adding new functionality.

## Setup

```bash
git clone https://github.com/isgallagher/forge-guardrails.git
cd forge-guardrails
python -m venv .venv
pip install -e ".[dev]"
```

## Running Tests

Unit tests are fully deterministic — no LLM backend required.

```bash
# Full suite
python -m pytest tests/unit/ -v --tb=short

# With coverage
python -m pytest tests/unit/ --cov=forge --cov-report=term-missing

# Single file
python -m pytest tests/unit/test_proxy_handler.py -v
```

## Project Layout

```
src/forge/           # Library source
  clients/           # LLM backend adapters (Anthropic, OpenAI-compatible)
  core/              # Workflow, messages, inference pipeline
  context/           # Context management and compaction
  guardrails/        # Validation, retry nudges, step enforcement
  proxy/             # HTTP server, handler, format conversion
  prompts/           # Prompt templates and nudges
  tools/             # Synthetic respond tool
tests/
  unit/              # Deterministic tests
docs/                # User-facing documentation
  decisions/         # Architecture Decision Records (ADRs)
```

## Common Contribution Areas

### Adding or modifying guardrails

Guardrails live in `src/forge/guardrails/` and nudge templates in `src/forge/prompts/nudges.py`. The inference pipeline in `src/forge/core/inference.py` composes them.

### Format conversion

OpenAI ↔ Anthropic conversion lives in `src/forge/proxy/convert.py`. Add new format pairs here.

### Client adapters

The `LLMClient` protocol is defined in `src/forge/clients/base.py`. New backends implement `send_http()` and `send_http_stream()`.

## Architecture Decision Records

Design decisions are documented in `docs/decisions/`. If you're proposing a significant change, consider writing an ADR first. See existing ones for the format.

## Code Style

- Python 3.12+ — use modern syntax (type unions with `|`, etc.)
- `asyncio` throughout — all client methods and the proxy are async
- Pydantic for tool parameter schemas
- No external formatting/linting tools enforced yet — match the style of surrounding code

## Questions

Open an issue on GitHub if something is unclear.
