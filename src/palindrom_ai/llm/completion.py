"""
Unified LLM completion wrapper using LiteLLM.

Provides async-first interface for OpenAI, Anthropic, and Google models
with configurable retry logic and exponential backoff.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import litellm
from litellm import ModelResponse
from litellm import acompletion as litellm_acompletion
from pydantic import BaseModel

from palindrom_ai.llm.config import LLMSettings, get_settings


class UsageStats(BaseModel):
    """Token usage statistics from a completion response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff."""

    max_retries: int = 3
    min_wait: float = 1.0  # Minimum wait between retries (seconds)
    max_wait: float = 60.0  # Maximum wait between retries (seconds)
    multiplier: float = 2.0  # Exponential backoff multiplier

    @classmethod
    def from_settings(cls, settings: LLMSettings | None = None) -> "RetryConfig":
        """Create RetryConfig from LLMSettings."""
        s = settings or get_settings()
        return cls(
            max_retries=s.default_max_retries,
            min_wait=s.retry_min_wait,
            max_wait=s.retry_max_wait,
            multiplier=s.retry_multiplier,
        )


async def complete(
    model: str,
    messages: list[dict[str, str]],
    fallbacks: list[str] | None = None,
    max_retries: int | None = None,
    timeout: float | None = None,
    retry_config: RetryConfig | None = None,
    settings: LLMSettings | None = None,
    **kwargs: Any,
) -> ModelResponse:
    """
    Unified async completion across all providers.

    Supports configurable retry logic with exponential backoff.

    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514",
               "gemini/gemini-2.0-flash")
        messages: List of message dicts with role and content
        fallbacks: Optional list of fallback models if primary fails
        max_retries: Number of retries on transient failures (overrides retry_config)
        timeout: Request timeout in seconds (uses settings default if not provided)
        retry_config: Optional RetryConfig for fine-grained retry control
        settings: Optional custom LLMSettings instance (uses global if not provided)
        **kwargs: Additional args passed to LiteLLM

    Returns:
        ModelResponse with completion and usage info

    Example:
        >>> response = await complete(
        ...     model="gpt-4o",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... )
        >>> print(response.choices[0].message.content)

        >>> # With custom retry config
        >>> config = RetryConfig(max_retries=5, min_wait=2.0, multiplier=3.0)
        >>> response = await complete(
        ...     model="gpt-4o",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     retry_config=config,
        ... )
    """
    s = settings or get_settings()
    rc = retry_config or RetryConfig.from_settings(s)

    # Allow max_retries to override retry_config for backwards compatibility
    effective_retries = max_retries if max_retries is not None else rc.max_retries
    effective_timeout = timeout if timeout is not None else s.default_timeout

    return await litellm_acompletion(
        model=model,
        messages=messages,
        fallbacks=fallbacks,
        num_retries=effective_retries,
        timeout=effective_timeout,
        **kwargs,
    )


async def stream(
    model: str,
    messages: list[dict[str, str]],
    timeout: float | None = None,
    settings: LLMSettings | None = None,
    **kwargs: Any,
) -> AsyncIterator[str]:
    """
    Stream completion tokens.

    Args:
        model: Model identifier
        messages: List of message dicts
        timeout: Request timeout in seconds (uses settings default if not provided)
        settings: Optional custom LLMSettings instance (uses global if not provided)
        **kwargs: Additional args passed to LiteLLM

    Yields:
        Individual tokens as they arrive

    Example:
        >>> async for token in stream(
        ...     model="gpt-4o",
        ...     messages=[{"role": "user", "content": "Tell me a story"}],
        ... ):
        ...     print(token, end="", flush=True)
    """
    s = settings or get_settings()
    effective_timeout = timeout if timeout is not None else s.default_timeout

    response = await litellm_acompletion(
        model=model,
        messages=messages,
        stream=True,
        timeout=effective_timeout,
        **kwargs,
    )
    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def get_cost(response: ModelResponse) -> float:
    """
    Get the cost in USD for a completion response.

    Args:
        response: The ModelResponse from a completion call

    Returns:
        Cost in USD

    Example:
        >>> response = await complete(model="gpt-4o", messages=[...])
        >>> cost = get_cost(response)
        >>> print(f"Cost: ${cost:.6f}")
    """
    return litellm.completion_cost(response)


def get_usage(response: ModelResponse) -> UsageStats:
    """
    Get token usage from a completion response.

    Args:
        response: The ModelResponse from a completion call

    Returns:
        UsageStats with prompt_tokens, completion_tokens, total_tokens

    Raises:
        ValueError: If usage information is not available in the response

    Example:
        >>> response = await complete(model="gpt-4o", messages=[...])
        >>> usage = get_usage(response)
        >>> print(f"Total tokens: {usage.total_tokens}")
    """
    if response.usage is None:
        raise ValueError("Usage information not available in response")
    return UsageStats(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
    )
