"""Context management for the forge library.

Provides compaction strategies, context budget management, and
default context warnings.
"""

from forge.context.manager import CompactEvent, ContextManager, default_context_warning
from forge.context.strategies import (
    CompactStrategy,
    NoCompact,
    SlidingWindowCompact,
    TieredCompact,
)

__all__ = [
    "CompactEvent",
    "CompactStrategy",
    "ContextManager",
    "default_context_warning",
    "NoCompact",
    "SlidingWindowCompact",
    "TieredCompact",
]
