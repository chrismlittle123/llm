"""
Unified LLM completion wrapper using LiteLLM.

Provides async-first interface for OpenAI, Anthropic, and Google models.
"""

from collections.abc import AsyncIterator
from typing import Any

import litellm
from litellm import ModelResponse
from litellm import acompletion as litellm_acompletion


async def complete(
    model: str,
    messages: list[dict[str, str]],
    fallbacks: list[str] | None = None,
    max_retries: int = 3,
    timeout: float = 60.0,
    **kwargs: Any,
) -> ModelResponse:
    """
    Unified async completion across all providers.

    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514",
               "gemini/gemini-2.0-flash")
        messages: List of message dicts with role and content
        fallbacks: Optional list of fallback models if primary fails
        max_retries: Number of retries on transient failures
        timeout: Request timeout in seconds
        **kwargs: Additional args passed to LiteLLM

    Returns:
        ModelResponse with completion and usage info

    Example:
        >>> response = await complete(
        ...     model="gpt-4o",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... )
        >>> print(response.choices[0].message.content)
    """
    return await litellm_acompletion(
        model=model,
        messages=messages,
        fallbacks=fallbacks,
        num_retries=max_retries,
        timeout=timeout,
        **kwargs,
    )


async def stream(
    model: str,
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> AsyncIterator[str]:
    """
    Stream completion tokens.

    Args:
        model: Model identifier
        messages: List of message dicts
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
    response = await litellm_acompletion(
        model=model,
        messages=messages,
        stream=True,
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


def get_usage(response: ModelResponse) -> dict[str, int]:
    """
    Get token usage from a completion response.

    Args:
        response: The ModelResponse from a completion call

    Returns:
        Dict with prompt_tokens, completion_tokens, total_tokens

    Raises:
        ValueError: If usage information is not available in the response

    Example:
        >>> response = await complete(model="gpt-4o", messages=[...])
        >>> usage = get_usage(response)
        >>> print(f"Total tokens: {usage['total_tokens']}")
    """
    if response.usage is None:
        raise ValueError("Usage information not available in response")
    return {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
