"""
Embedding utilities for RAG using LiteLLM.

Provides functions to generate embeddings for texts using various providers.
"""

import litellm


async def embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
) -> list[list[float]]:
    """
    Generate embeddings for texts using LiteLLM.

    Args:
        texts: List of texts to embed
        model: Embedding model (default: OpenAI text-embedding-3-small)

    Returns:
        List of embedding vectors

    Example:
        >>> embeddings = await embed(["Hello world", "Goodbye world"])
        >>> len(embeddings)
        2
    """
    response = await litellm.aembedding(
        model=model,
        input=texts,
    )
    return [item["embedding"] for item in response.data]


async def embed_single(
    text: str,
    model: str = "text-embedding-3-small",
) -> list[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Text to embed
        model: Embedding model (default: OpenAI text-embedding-3-small)

    Returns:
        Embedding vector

    Example:
        >>> embedding = await embed_single("Hello world")
        >>> len(embedding) > 0
        True
    """
    embeddings = await embed([text], model=model)
    return embeddings[0]
