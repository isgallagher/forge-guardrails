"""forge — a transparent proxy with guardrails for LLM tool-calling."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("forge-guardrails")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from forge.clients.anthropic import AnthropicClient
from forge.clients.base import ChunkType, LLMClient, StreamChunk, TokenUsage
from forge.context import (
    CompactEvent,
    CompactStrategy,
    ContextManager,
    NoCompact,
    SlidingWindowCompact,
    TieredCompact,
    default_context_warning,
)
from forge.core.inference import InferenceResult, fold_and_serialize, run_inference
from forge.core.messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo
from forge.core.steps import StepTracker
from forge.core.workflow import (
    LLMResponse,
    TextResponse,
    ToolCall,
    ToolDef,
    ToolSpec,
    Workflow,
)
from forge.errors import (
    ContextBudgetExceeded,
    ForgeError,
    StreamError,
    ToolCallError,
    ToolResolutionError,
)
from forge.guardrails import (
    CheckResult,
    ErrorTracker,
    Guardrails,
    Nudge,
    ResponseValidator,
    StepCheck,
    StepEnforcer,
    ValidationResult,
)
from forge.prompts import build_tool_prompt, extract_tool_call, rescue_tool_call, retry_nudge, step_nudge
from forge.tools import RESPOND_TOOL_NAME, respond_spec, respond_tool

__all__ = [
    # Version
    "__version__",
    # Messages
    "Message",
    "MessageMeta",
    "MessageRole",
    "MessageType",
    "ToolCallInfo",
    # Tools & Workflow
    "LLMResponse",
    "TextResponse",
    "ToolCall",
    "ToolDef",
    "ToolSpec",
    "Workflow",
    # Steps
    "StepTracker",
    # Inference (front half — shared by runner and proxy)
    "InferenceResult",
    "fold_and_serialize",
    "run_inference",
    # Client
    "AnthropicClient",
    "ChunkType",
    "LLMClient",
    "StreamChunk",
    "TokenUsage",
    # Context
    "CompactEvent",
    "CompactStrategy",
    "ContextManager",
    "default_context_warning",
    "NoCompact",
    "SlidingWindowCompact",
    "TieredCompact",
    # Prompts
    "build_tool_prompt",
    "extract_tool_call",
    "rescue_tool_call",
    "retry_nudge",
    "step_nudge",
    # Built-in tools
    "RESPOND_TOOL_NAME",
    "respond_spec",
    "respond_tool",
    # Guardrails
    "CheckResult",
    "Guardrails",
    # Guardrails (granular middleware)
    "ErrorTracker",
    "Nudge",
    "ResponseValidator",
    "StepCheck",
    "StepEnforcer",
    "ValidationResult",
    # Errors
    "ContextBudgetExceeded",
    "ForgeError",
    "StreamError",
    "ToolCallError",
    "ToolResolutionError",
]
