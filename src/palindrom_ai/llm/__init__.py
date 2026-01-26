"""
Palindrom AI LLM SDK

Unified Python SDK for LLM operations including:
- Multi-provider completions (OpenAI, Anthropic, Google)
- Structured output with Pydantic validation
- LLM observability with Langfuse
- Simple RAG with vector search
- LLM evaluation and testing
"""

from typing import TYPE_CHECKING

__version__ = "0.1.0"

from palindrom_ai.llm.completion import complete, get_cost, get_usage, stream
from palindrom_ai.llm.config import LLMSettings, configure, get_settings
from palindrom_ai.llm.structured import extract, extract_stream

# Lazy imports for observability (langfuse may not be compatible with all Python versions)
if TYPE_CHECKING:
    from palindrom_ai.llm.metrics import (
        MetricsBridge,
        collect_metrics_once,
        init_metrics_bridge,
        stop_metrics_bridge,
    )
    from palindrom_ai.llm.observability import (
        add_trace_metadata,
        flush_traces,
        init_observability,
        set_trace_session,
        set_trace_user,
        trace,
    )
    from palindrom_ai.llm.prompts import (
        get_chat_prompt,
        get_langfuse,
        get_prompt,
        get_prompt_with_vars,
    )
    from palindrom_ai.llm.rag import (
        SearchResult,
        VectorStore,
        embed,
        embed_single,
        retrieve_and_generate,
    )

__all__ = [
    # Version
    "__version__",
    # Completion
    "complete",
    "stream",
    "get_cost",
    "get_usage",
    # Structured output
    "extract",
    "extract_stream",
    # Observability
    "init_observability",
    "trace",
    "add_trace_metadata",
    "set_trace_user",
    "set_trace_session",
    "flush_traces",
    # Metrics
    "init_metrics_bridge",
    "collect_metrics_once",
    "stop_metrics_bridge",
    "MetricsBridge",
    # Prompts
    "get_prompt",
    "get_prompt_with_vars",
    "get_chat_prompt",
    "get_langfuse",
    # Config
    "get_settings",
    "configure",
    "LLMSettings",
    # RAG
    "VectorStore",
    "SearchResult",
    "embed",
    "embed_single",
    "retrieve_and_generate",
]


def __getattr__(name: str):
    """Lazy import for observability, metrics, and prompt modules."""
    if name in (
        "init_observability",
        "trace",
        "add_trace_metadata",
        "set_trace_user",
        "set_trace_session",
        "flush_traces",
    ):
        from palindrom_ai.llm import observability

        return getattr(observability, name)
    if name in (
        "init_metrics_bridge",
        "collect_metrics_once",
        "stop_metrics_bridge",
        "MetricsBridge",
    ):
        from palindrom_ai.llm import metrics

        return getattr(metrics, name)
    if name in ("get_prompt", "get_prompt_with_vars", "get_chat_prompt", "get_langfuse"):
        from palindrom_ai.llm import prompts

        return getattr(prompts, name)
    if name in (
        "VectorStore",
        "SearchResult",
        "embed",
        "embed_single",
        "retrieve_and_generate",
    ):
        from palindrom_ai.llm import rag

        return getattr(rag, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
