"""Exception hierarchy for the forge library."""


class ForgeError(Exception):
    """Base exception for the forge library."""

    pass


class ToolCallError(ForgeError):
    """LLM failed to produce a valid tool call after retries."""

    def __init__(
        self,
        message: str,
        raw_response: str | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.raw_response = raw_response
        self.cause = cause


class ToolResolutionError(Exception):
    """Tool arguments were valid but the data didn't resolve.

    The tool equivalent of HTTP 4xx — the call was well-formed and the
    schema was satisfied, but the arguments couldn't be resolved against
    the underlying data (wrong key, empty result set, unrecognized ID,
    etc.).

    Raise this from a tool callable to signal "try again with different
    arguments" without counting toward consecutive_tool_errors.  The
    runner feeds the message back to the model and does NOT mark the
    step as completed.

    Not a ForgeError — this is a tool-author exception, not a framework
    error.  The runner catches it explicitly.
    """

    def __init__(self, message: str, tool_name: str | None = None):
        super().__init__(message)
        self.tool_name = tool_name


class ContextBudgetExceeded(ForgeError):
    """Context exceeded budget even after compaction. Unrecoverable."""

    def __init__(self, estimated_tokens: int, budget_tokens: int):
        super().__init__(
            f"Context budget exceeded: {estimated_tokens} tokens "
            f"estimated, budget is {budget_tokens}"
        )
        self.estimated_tokens = estimated_tokens
        self.budget_tokens = budget_tokens


class BackendError(ForgeError):
    """Unexpected HTTP error from the LLM backend."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"Backend returned {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class StreamError(ForgeError):
    """Stream ended without producing a FINAL chunk."""

    def __init__(self, message: str = "Stream ended without FINAL chunk"):
        super().__init__(message)
