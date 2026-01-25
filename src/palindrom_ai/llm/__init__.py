"""
Palindrom AI LLM SDK

Unified Python SDK for LLM operations including:
- Multi-provider completions (OpenAI, Anthropic, Google)
- Structured output with Pydantic validation
- Simple RAG with vector search
- LLM evaluation and testing
"""

__version__ = "0.1.0"

from palindrom_ai.llm.completion import complete, get_cost, get_usage, stream
from palindrom_ai.llm.config import LLMSettings, configure, get_settings
from palindrom_ai.llm.structured import extract, extract_stream

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
    # Config
    "get_settings",
    "configure",
    "LLMSettings",
]
