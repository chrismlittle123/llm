"""Tests for the FastAPI HTTP gateway."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from palindrom_ai.llm.server.app import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_has_request_id():
    resp = client.get("/health")
    assert "x-request-id" in resp.headers


# ---------------------------------------------------------------------------
# POST /v1/complete
# ---------------------------------------------------------------------------


def _mock_model_response(content: str = "Hello!") -> MagicMock:
    """Build a fake ModelResponse-like object."""
    resp = MagicMock()
    resp.model_dump.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    return resp


@patch("palindrom_ai.llm.server.routes.complete", new_callable=AsyncMock)
def test_complete(mock_complete: AsyncMock):
    mock_complete.return_value = _mock_model_response("Hi there")
    resp = client.post(
        "/v1/complete",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "Hi there"
    assert data["usage"]["totalTokens"] == 15


@patch("palindrom_ai.llm.server.routes.complete", new_callable=AsyncMock)
def test_complete_has_request_id(mock_complete: AsyncMock):
    mock_complete.return_value = _mock_model_response()
    resp = client.post(
        "/v1/complete",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert "x-request-id" in resp.headers


def test_complete_invalid_body():
    """Missing required 'model' field returns 422."""
    resp = client.post("/v1/complete", json={"messages": []})
    assert resp.status_code == 422


@patch("palindrom_ai.llm.server.routes.complete", new_callable=AsyncMock)
def test_complete_llm_error(mock_complete: AsyncMock):
    """LLM exception returns 500 with structured error."""
    mock_complete.side_effect = RuntimeError("provider down")
    resp = client.post(
        "/v1/complete",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 500
    data = resp.json()
    assert data["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert "provider down" in data["error"]["message"]
    assert data["error"]["requestId"] is not None


# ---------------------------------------------------------------------------
# POST /v1/extract
# ---------------------------------------------------------------------------


@patch("palindrom_ai.llm.server.routes.extract", new_callable=AsyncMock)
def test_extract(mock_extract: AsyncMock):
    fake_result = MagicMock()
    fake_result.model_dump.return_value = {"name": "Alice", "age": 30}
    mock_extract.return_value = fake_result

    resp = client.post(
        "/v1/extract",
        json={
            "model": "gpt-4o",
            "prompt": "Alice is 30",
            "response_schema": {
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["data"] == {"name": "Alice", "age": 30}


@patch("palindrom_ai.llm.server.routes.extract", new_callable=AsyncMock)
def test_extract_has_request_id(mock_extract: AsyncMock):
    fake_result = MagicMock()
    fake_result.model_dump.return_value = {"x": 1}
    mock_extract.return_value = fake_result

    resp = client.post(
        "/v1/extract",
        json={
            "model": "gpt-4o",
            "prompt": "x is 1",
            "response_schema": {
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        },
    )
    assert "x-request-id" in resp.headers


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


def test_cors_headers():
    resp = client.options(
        "/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-origin" in resp.headers


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@patch("palindrom_ai.llm.server.routes.complete", new_callable=AsyncMock)
def test_value_error_returns_400(mock_complete: AsyncMock):
    mock_complete.side_effect = ValueError("bad model name")
    resp = client.post(
        "/v1/complete",
        json={"model": "bad", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"]["code"] == "BAD_REQUEST"
    assert "bad model name" in data["error"]["message"]
