"""API route handlers for the HTTP gateway."""

from typing import Any

from fastapi import APIRouter, Request
from pydantic import create_model

from palindrom_ai.llm.completion import complete
from palindrom_ai.llm.server.models import (
    CompletionRequest,
    CompletionResponse,
    ExtractionRequest,
    ExtractionResponse,
)
from palindrom_ai.llm.structured import extract

router = APIRouter()

# JSON Schema type → Python type mapping
_JSON_SCHEMA_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _json_schema_to_fields(
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    """Convert JSON Schema properties to pydantic create_model field definitions."""
    required = required or []
    fields: dict[str, Any] = {}
    for name, prop in properties.items():
        python_type = _resolve_type(prop)
        if name in required:
            fields[name] = (python_type, ...)
        else:
            fields[name] = (python_type | None, None)
    return fields


def _resolve_type(prop: dict[str, Any]) -> type:
    """Resolve a single JSON Schema property to a Python type."""
    schema_type = prop.get("type", "string")

    if schema_type == "array":
        item_type = _resolve_type(prop.get("items", {"type": "string"}))
        return list[item_type]  # type: ignore[valid-type]

    if schema_type == "object":
        nested_props = prop.get("properties", {})
        nested_required = prop.get("required", [])
        nested_fields = _json_schema_to_fields(nested_props, nested_required)
        return create_model("NestedModel", **nested_fields)  # type: ignore[call-overload]

    return _JSON_SCHEMA_TYPES.get(schema_type, str)


def _build_response_model(schema: dict[str, Any]) -> type:
    """Build a dynamic Pydantic model from a JSON Schema dict."""
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    fields = _json_schema_to_fields(properties, required)
    return create_model("DynamicResponseModel", **fields)  # type: ignore[call-overload]


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/v1/complete")
async def complete_endpoint(body: CompletionRequest) -> CompletionResponse:
    response = await complete(
        model=body.model,
        messages=body.messages,
        fallbacks=body.fallbacks,
        max_retries=body.max_retries,
        timeout=body.timeout,
    )
    content = response.choices[0].message.content or ""  # ty: ignore[possibly-missing-attribute]
    usage = {}
    if response.usage:  # ty: ignore[unresolved-attribute]
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,  # ty: ignore[unresolved-attribute]
            "completion_tokens": response.usage.completion_tokens,  # ty: ignore[unresolved-attribute]
            "total_tokens": response.usage.total_tokens,  # ty: ignore[unresolved-attribute]
        }
    return CompletionResponse(content=content, usage=usage)


@router.post("/v1/extract")
async def extract_endpoint(body: ExtractionRequest, request: Request) -> ExtractionResponse:
    response_model = _build_response_model(body.response_schema)
    result = await extract(
        response_model=response_model,  # ty: ignore[invalid-argument-type]
        model=body.model,
        messages=body.messages,
        prompt=body.prompt,
        max_retries=body.max_retries,
    )
    data = result.model_dump()
    # Usage is tracked via observability; return empty dict as placeholder
    return ExtractionResponse(data=data, usage={})
