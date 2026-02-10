# @palindrom-ai/llm

Unified Python LLM SDK for Palindrom services.

## Installation

```bash
pip install palindrom-ai-llm
```

With the HTTP gateway:

```bash
pip install "palindrom-ai-llm[server]"
```

For development:

```bash
pip install -e ".[dev]"
```

## Features

- **Unified API**: Single interface for OpenAI, Anthropic, and Google models
- **Structured Output**: All LLM calls return validated Pydantic models
- **HTTP Gateway**: FastAPI server exposing the SDK over HTTP for non-Python clients
- **Simple RAG**: Vector search with ChromaDB
- **Observability**: Built-in Langfuse integration
- **Evaluation**: DeepEval integration for testing
- **Retry Logic**: Configurable exponential backoff for transient failures
- **Type Safety**: Full type hints with PEP 561 py.typed marker

## Important: Langfuse Version Constraint

This SDK requires **Langfuse v2.x** (`langfuse>=2.0.0,<3.0.0`). This constraint exists because LiteLLM's Langfuse callback integration is not yet compatible with Langfuse v3.

When Langfuse v3 support is added to LiteLLM, this constraint will be relaxed. Track progress at the [LiteLLM repository](https://github.com/BerriAI/litellm).

## Quick Start

### Python SDK

```python
from palindrom_ai.llm import complete, extract
from pydantic import BaseModel

# Simple completion
response = await complete(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Structured extraction
class User(BaseModel):
    name: str
    age: int

user = await extract(
    response_model=User,
    model="gpt-4o",
    prompt="John is 25 years old",
)
print(user.name)  # "John"
```

### HTTP Gateway

Start the server:

```bash
uvicorn palindrom_ai.llm.server.app:app --port 8000
```

Call from any language:

```bash
# Health check
curl http://localhost:8000/health

# Completion
curl -X POST http://localhost:8000/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Structured extraction
curl -X POST http://localhost:8000/v1/extract \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "prompt": "John is 25 years old",
    "response_schema": {
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
      },
      "required": ["name", "age"]
    }
  }'
```

## HTTP Gateway

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check, returns `{"status": "ok"}` |
| `POST` | `/v1/complete` | LLM completion with model routing and fallbacks |
| `POST` | `/v1/extract` | Structured data extraction via JSON Schema |

### `POST /v1/complete`

Request body:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | `string` | yes | Model identifier (e.g. `gpt-4o`, `claude-sonnet-4-20250514`) |
| `messages` | `array` | yes | List of `{"role": "...", "content": "..."}` message objects |
| `fallbacks` | `string[]` | no | Fallback models if primary fails |
| `max_retries` | `integer` | no | Number of retries on transient failures |
| `timeout` | `number` | no | Request timeout in seconds |

Response:

```json
{
  "content": "Hello! How can I help?",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

### `POST /v1/extract`

Request body:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | `string` | yes | Model identifier |
| `messages` | `array` | no | Message list (mutually exclusive with `prompt`) |
| `prompt` | `string` | no | Simple prompt string |
| `response_schema` | `object` | yes | JSON Schema defining the output structure |
| `max_retries` | `integer` | no | Retries on validation failure (default: 3) |

The `response_schema` supports `string`, `integer`, `number`, `boolean`, `array`, and nested `object` types.

Response:

```json
{
  "data": {"name": "John", "age": 25},
  "usage": {}
}
```

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `LLM_CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse public key (optional, enables observability) |
| `LANGFUSE_SECRET_KEY` | — | Langfuse secret key (optional, enables observability) |
| `LANGFUSE_HOST` | Langfuse cloud | Langfuse host URL |

Observability is best-effort: if Langfuse env vars are not set, the server starts normally with a warning.

### Interactive API Docs

When the server is running, interactive documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

### Middleware

Every response includes an `X-Request-ID` header for tracing. Error responses return a consistent JSON structure:

```json
{
  "error": "Bad Request",
  "detail": "description of the error",
  "request_id": "uuid"
}
```

## Documentation

- [Research Docs](./docs/research/)
- [Implementation Plans](./docs/plans/)

## Development

```bash
# Install dev dependencies
uv sync --extra all --extra dev

# Run unit tests
uv run pytest -m "not integration" -v

# Run integration tests (requires API keys)
uv run pytest tests/integration/ -v

# Run all tests
uv run pytest -v

# Lint and type check
uv run ruff check . && uv run ruff format --check . && uv run ty check
```
