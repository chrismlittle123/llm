"""Pydantic request/response models for the HTTP gateway."""

from pydantic import BaseModel


class CompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    fallbacks: list[str] | None = None
    max_retries: int | None = None
    timeout: float | None = None


class CompletionResponse(BaseModel):
    content: str
    usage: dict


class ExtractionRequest(BaseModel):
    model: str
    messages: list[dict[str, str]] | None = None
    prompt: str | None = None
    response_schema: dict  # JSON Schema
    max_retries: int = 3


class ExtractionResponse(BaseModel):
    data: dict
    usage: dict


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None
