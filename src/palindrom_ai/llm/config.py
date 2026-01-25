"""
Configuration for the LLM package.

Uses pydantic-settings to load from environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys (loaded from environment)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    # Defaults
    default_model: str = "gpt-4o"
    default_timeout: float = 60.0
    default_max_retries: int = 3

    # Langfuse (for observability)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # OTLP Configuration (for metrics export)
    otlp_endpoint: str = "http://localhost:4317"
    otlp_insecure: bool = True
    service_name: str = "palindrom-llm"
    service_environment: str = "development"


# Global settings instance
_settings: LLMSettings | None = None


def get_settings() -> LLMSettings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = LLMSettings()
    return _settings


def configure(**kwargs: str | float | int | None) -> None:
    """
    Configure LLM settings programmatically.

    Args:
        **kwargs: Settings to override

    Example:
        >>> configure(default_model="claude-sonnet-4-20250514", default_timeout=30.0)
    """
    global _settings
    current = get_settings().model_dump()
    current.update({k: v for k, v in kwargs.items() if v is not None})
    _settings = LLMSettings(**current)
