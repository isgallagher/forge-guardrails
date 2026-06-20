# Architecture

The forge guardrails proxy — how it works and why it's designed this way.

## What forge is

A transparent HTTP proxy that sits between LLM clients (Claude Code, Continue, opencode, etc.) and LLM backends (Anthropic API, llama-server, vLLM). All inference is handled by the backend — forge does not run models.

Forge applies guardrails to requests with tools:

- **Response validation** — each tool call is checked against the `tools` array in the request
- **Rescue parsing** — malformed tool calls (JSON in code fences, Mistral `[TOOL_CALLS]`, Qwen XML) are extracted into structured form
- **Retry nudges** — bare text responses when a tool call is expected trigger a corrective retry
- **Synthetic `respond` tool** — injected when tools are present so the model calls `respond(message="...")` instead of producing bare text
- **Context compaction** — deterministic tiered compaction when the conversation approaches the context budget

Requests without tools pass through directly to the backend, skipping guardrails.

---

## Design Principles

### 1. Pass-Through by Default

Requests without tools bypass guardrails entirely. No overhead for simple chat — the proxy is transparent. Only when tools are present does forge intervene, and even then only for response quality (validation, rescue, retry).

### 2. Fail Fast, Fail Loud

No defensive coding. No silent `try/except`, no swallowed errors. If the model returns garbage, the retry loop handles it explicitly. If retries are exhausted, forge raises a typed exception with full context (attempt count, last error, last raw response). Silent failures in agentic loops are devastating — a swallowed error at step 3 corrupts every subsequent step.

```python
# BAD — defensive
try:
    tool_call = parse_tool_call(response)
except Exception:
    tool_call = ToolCall(tool="fallback", args={})  # Silent corruption

# GOOD — fail fast with context
try:
    tool_call = parse_tool_call(response)
except ParseError as e:
    raise ToolCallError(
        f"Failed to parse tool call on attempt {attempt}/{max_retries}",
        raw_response=response,
        cause=e,
    )
```

### 3. Control Flow Is Not Memory

Forge separates *what the model remembers* (message history, subject to compaction) from *what the runner enforces* (step completion, iteration count, terminal conditions). The model's context is a resource to be managed. Control-flow state is authoritative and lives outside the message history.

Concrete consequence: step completion is tracked in a `StepTracker` on the runner. Compaction may aggressively drop a tool result, but `StepEnforcer` checks `completed_steps` from the tracker, not from what the model "remembers."

**Tradeoff:** The model may redundantly re-call a tool whose result was compacted. This wastes an iteration but doesn't corrupt the workflow. Tools that are expensive to re-run should be idempotent.

### 4. The Client Adapter Is the Abstraction Boundary

Forge doesn't know whether the LLM supports native function calling, prompt-injected tool calling, or some future protocol. The `LLMClient` adapter translates between forge's internal `ToolCall` representation and whatever the backend expects. The tool-calling loop receives validated `ToolCall` objects and never parses raw text.

This means rescue parsing (Mistral `[TOOL_CALLS]`, Qwen `<tool_call>` XML, fenced JSON, etc.) lives in the *client* — the inference pipeline doesn't grow special cases per model family.

The client adapter is the abstraction boundary: it owns the wire format, the serialization, and the response parsing. Everything above it works with forge's canonical types (`ToolCall`, `TextResponse`).

#### Encoder / Hidden-Layers / Decoder Pattern

All transformation happens at the boundaries. Internals never transform data.

- **Encoder** (`src/forge/clients/`): Converts backend wire format → forge canonical types (`ToolCall`, `TextResponse`, `TokenUsage`). For example, `AnthropicClient` maps `{"input_tokens": N, "output_tokens": M}` to `TokenUsage(prompt_tokens=N, completion_tokens=M, total_tokens=N+M)`.
- **Hidden layers** (`src/forge/core/`, `src/forge/guardrails/`): Inference loop, guardrails, context management — all use canonical types only. No format transformation.
- **Decoder** (`src/forge/proxy/convert.py`): Converts canonical types → wire format (OpenAI/Anthropic) for the client.

Data flow for token usage:
```
Backend raw usage → Encoder (AnthropicClient) → TokenUsage(prompt_tokens, completion_tokens, total_tokens)
  → Hidden layers (inference, _sync_token_count, guardrails) — unchanged
    → Decoder (convert.py) → Wire format:
       OpenAI client: {"prompt_tokens": N, "completion_tokens": M, "total_tokens": T}
       Anthropic client: {"input_tokens": N, "output_tokens": M}
```

**Key rule:** All transformation happens at the boundaries. Internals never transform.

### 5. Context Is a First-Class Resource

Forge budgets context proactively. A long conversation with many tool calls can easily exceed the model's context window. Context management is not optional — it's load-bearing infrastructure.

The compaction strategy is owned by the strategy object (not the manager), so swapping `TieredCompact` for `SlidingWindowCompact` or a custom strategy is a constructor change.

---

## Request Flow

```
Client ──► Forge Proxy ──► Backend
              │
         ┌────┴────┐
         │  Guard  │  (only when tools present)
         └─────────┘
```

### Format Handling

Forge automatically detects which API formats the backend supports (OpenAI, Anthropic, or both) by probing on startup. The routing logic:

- **Same format** (OpenAI client → OpenAI backend, Anthropic → Anthropic): direct pass-through
- **Different format**: forge translates using conversion functions in `src/forge/proxy/convert.py`
- **With tools**: always goes through guardrails (validate, retry, nudge)
- **Without tools**: passes through to the backend directly

### Backend Detection

On startup, forge probes the backend's `/v1/models`, `/v1/messages`, and `/v1/chat/completions` endpoints to determine supported formats. Results are cached and used for all subsequent routing decisions.

---

## Key Components

| File | Purpose |
|---|---|
| `src/forge/proxy/server.py` | Raw asyncio HTTP server, SSE streaming, routing |
| `src/forge/proxy/handler.py` | HTTP request → inference pipeline bridge |
| `src/forge/proxy/proxy.py` | ProxyServer — programmatic API for start/stop |
| `src/forge/proxy/convert.py` | OpenAI ↔ Anthropic format conversion |
| `src/forge/clients/base.py` | LLMClient protocol with `send_http` methods |
| `src/forge/clients/anthropic.py` | AnthropicClient (direct HTTP pass-through) |
| `src/forge/clients/openai_compatible.py` | OpenAICompatibleClient (httpx, no SDK) |
| `src/forge/guardrails/` | Validation, retry nudges, step enforcement |
| `src/forge/core/inference.py` | `run_inference()` — compact, fold, validate, send, retry |

---

## Guardrails

| Guardrail | What it catches | Wire shape |
|---|---|---|
| **Rescue parsing** | Model emits tool call in wrong format (JSON fence, XML, etc.) | Extracted to structured tool call |
| **Response validation** | Unknown tool name, malformed args | Retry nudge on user channel |
| **Retry nudges** | Bare text instead of tool call | User nudge: "you must call a tool" |
| **Step enforcement** | Premature terminal tool call | Tool-error reply on tool channel |
| **Compaction** | Conversation approaching context budget | Deterministic text manipulation, no LLM calls |

---

## The Synthetic `respond` Tool

When tools are present but the user sends a conversational message, small models must choose between calling a tool and responding with text. They frequently choose wrong. Eval testing showed that trusting the model's finish reason dropped workflow completion from 100% to as low as 4%.

The respond tool eliminates the open-ended choice. The model calls `respond(message="...")` instead of producing bare text. From forge's perspective, every response is a valid tool call — no retries wasted on conversational turns, no accuracy loss on tool-calling turns.

**Why this works for small models:** small models struggle with open-ended decisions ("tools or chat?") but are good at structured choices ("which tool?"). The respond tool converts an open-ended decision into a structured one. The model stays in tool-calling grammar at all times, which is where it performs best.

The `respond` call is stripped from the outbound response — the client sees a normal text response and never knows the tool exists.

---

## Context Management

Forge budgets context proactively. When the conversation approaches the budget limit, tiered compaction fires:

| Priority | Type | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| Cut first | `step_nudge`, `prerequisite_nudge`, `retry_nudge` | Drop | Drop | Drop |
| Cut second | Older `tool_result` | Truncate ~200 chars | Drop | Drop |
| Cut third | `text_response` | Preserved | Preserved | Drop |
| Cut fourth | `reasoning` | Preserved | Preserved | Drop |
| Preserved | Older `tool_call` | Preserved | Preserved | Preserved (full) |
| Never cut | `system_prompt`, `user_input` | Preserved | Preserved | Preserved |
| Never cut | Recent iterations (`keep_recent`) | Preserved | Preserved | Preserved |

**Key design choice — reasoning survives through Phase 2.** The model's chain-of-thought from step 3 ("price below web but above historical") is what informs decisions at step 5+. Losing raw tool results is recoverable; losing the model's interpretation of those results is not. `text_response` (a failed tool call attempt) is expendable after the retry nudge corrects the model.

**Phase 3 is the emergency cutoff** — should only fire under extreme context pressure.

All three phases are deterministic text manipulation — no LLM calls, sub-millisecond.

Three built-in strategies; consumers can supply their own by implementing the `CompactStrategy` interface:

- **`NoCompact`** — passthrough. Use when context is abundant or workflows are short.
- **`SlidingWindowCompact`** — keeps system prompt, original user input, and the last N iterations. Simple, predictable.
- **`TieredCompact`** (default) — three-phase escalating compaction with the priority order above.

---

## Where to find things

- **How to use the proxy** — [USER_GUIDE.md](USER_GUIDE.md)
- **Past decisions and rationale** — [decisions/](decisions/) (ADRs)
- **Class signatures and exact APIs** — source (`src/forge/`) is authoritative
