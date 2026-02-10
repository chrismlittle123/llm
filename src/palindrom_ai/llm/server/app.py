"""FastAPI HTTP gateway for the Palindrom AI LLM SDK."""

import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from palindrom_ai.llm.server.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: init observability (best-effort). Shutdown: flush traces."""
    try:
        from palindrom_ai.llm.observability import init_observability

        init_observability()
        logger.info("Langfuse observability initialized")
    except (ValueError, Exception) as exc:
        logger.warning("Observability not initialized: %s", exc)
    yield
    try:
        from palindrom_ai.llm.observability import flush_traces

        flush_traces()
    except Exception:
        pass


app = FastAPI(title="Palindrom AI LLM Gateway", lifespan=lifespan)

# --- CORS ---
cors_origins = os.getenv("LLM_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,  # ty: ignore[invalid-argument-type]
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request ID middleware (raw ASGI for performance) ---
class RequestIDMiddleware:
    def __init__(self, app):  # ty: ignore[unresolved-attribute]
        self.app = app

    async def __call__(self, scope, receive, send):  # ty: ignore[unresolved-attribute]
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_request_id)


app.add_middleware(RequestIDMiddleware)  # ty: ignore[invalid-argument-type]


# --- Exception handlers ---
def _get_request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": "Bad Request",
            "detail": str(exc),
            "request_id": _get_request_id(request),
        },
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "request_id": _get_request_id(request),
        },
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "request_id": _get_request_id(request),
        },
    )


# --- Routes ---
app.include_router(router)
