"""FastAPI HTTP gateway for the Palindrom AI LLM SDK."""

import importlib
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.types import ASGIApp, Receive, Scope, Send

from palindrom_ai.llm.server.routes import router

logger = logging.getLogger(__name__)


def _resolve_secrets() -> None:
    """Resolve *_SECRET_NAME env vars from GCP Secret Manager into actual values."""
    secret_name_vars = {k: v for k, v in os.environ.items() if k.endswith("_SECRET_NAME")}
    if not secret_name_vars:
        return
    try:
        secretmanager = importlib.import_module("google.cloud.secretmanager")
        client = secretmanager.SecretManagerServiceClient()
        for env_var, secret_resource in secret_name_vars.items():
            target_var = env_var.removesuffix("_SECRET_NAME")
            if os.environ.get(target_var):
                continue  # Already set directly, don't override
            try:
                response = client.access_secret_version(name=f"{secret_resource}/versions/latest")
                os.environ[target_var] = response.payload.data.decode("utf-8")
                logger.info("Resolved secret %s -> %s", env_var, target_var)
            except Exception as exc:
                logger.warning("Failed to resolve secret %s: %s", env_var, exc)
    except ImportError:
        logger.warning("google-cloud-secret-manager not installed, skipping secret resolution")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: resolve secrets, init observability. Shutdown: flush traces."""
    _resolve_secrets()
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
# cast() needed because ty cannot match Starlette middleware classes to the
# _MiddlewareFactory[P] ParamSpec protocol used by add_middleware.
app.add_middleware(
    cast(Any, CORSMiddleware),
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request ID middleware (raw ASGI for performance) ---
class RequestIDMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: Any) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_request_id)


app.add_middleware(cast(Any, RequestIDMiddleware))


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
