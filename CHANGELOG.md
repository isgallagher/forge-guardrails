# Changelog

All notable changes to forge-guardrails are documented here.

## [0.8.0] — Unreleased

### Changed
- **Proxy-only scope** — removed eval harness, WorkflowRunner, LlamafileClient, OllamaClient, sampling defaults, SlotWorker, and all inference-specific infrastructure. Forge is now focused exclusively on the guardrails proxy.
- **Remove /v1 auto-append from backend_url** — `BACKEND_URL` is now a clean base URL with `/v1` prepended to paths where needed. Consistent between OpenAI and Anthropic client paths.
- **Backend detection** — forge probes the backend on startup to determine which API formats it supports (OpenAI, Anthropic, or both) and routes/transforms automatically.

### Retained
- Guardrails stack (response validation, rescue parsing, retry nudges, synthetic `respond` tool)
- Context management (TieredCompact, SlidingWindowCompact, NoCompact strategies)
- OpenAI ↔ Anthropic format conversion
- Direct HTTP pass-through (no SDK required)
- Token usage pass-through in proxy responses
- Client disconnect detection and backend request cancellation

### Removed
- Eval framework (`tests/eval/`, `scripts/run_ablation.py`, `eval_*.jsonl`)
- Llamafile and Ollama clients (proxy-only, backend handles auth)
- Hardware detection, slot workers, runner, sampling defaults
- Decision logs for removed infrastructure (ADR-003, ADR-011)
- Flake.nix (Docker-only build)
- Eval dashboard and model guide docs

## [0.7.0] — 2026-05-22

### Added
- Real token counting from backends for compaction decisions
- Proxy unit tests covering handler, convert, and server modules

### Changed
- Step enforcement and prerequisite violations surface as tool-error responses on the tool channel
- Unknown-tool retry on the tool channel instead of user nudges

## [0.6.0] — 2026-04-29

### Added
- Proxy sampling pass-through — proxy plumbs OpenAI-compatible body fields through to the backend per request

## [0.5.0] — 2026-04-19

### Added
- Three-screen dashboard for ablation results
- Statistical significance script

## [0.4.3] — 2026-04-17

### Added
- Qwen Coder XML rescue parsing

## [0.4.2] — 2026-04-10

### Added
- `forge-proxy` CLI entry point
- codecov integration

## [0.4.0] — 2026-04-02

### Added
- SlotWorker for priority-queued shared slot access
- Tool prerequisites with arg-matched enforcement
- Workflow cancellation via `cancel_event`
- Multiple terminal tools support
- Custom retry nudges

### Changed
- Removed `trust_text_intent` and `TextResponse.intentional` — respond tool pattern supersedes

## [0.3.0] — 2026-03-12

### Added
- Proxy server — OpenAI-compatible drop-in proxy with automatic respond tool injection
- Guardrails middleware — composable middleware for foreign orchestration loops
- Anthropic client — frontier baseline backend
- Context thresholds — configurable warning callbacks at budget percentages
- TieredCompact — three-phase compaction strategy

## [0.2.0] — 2026-02-15

### Added
- WorkflowRunner — agentic tool-calling loop with retry logic
- ResponseValidator — rescue parsing for malformed tool calls
- StepEnforcer — required step and terminal tool enforcement
- OllamaClient and LlamafileClient — local model backends

## [0.1.0] — 2026-01-20

- Initial release — core framework with tool-calling loop, basic guardrails, Ollama backend
