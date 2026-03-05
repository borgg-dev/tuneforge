"""
FastAPI application for the TuneForge Subnet API.

Runs on validator machines. Provides the organic generation endpoint
that the platform API proxies to. No SaaS logic — just Bittensor
primitives and the organic query router.
"""

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

import bittensor as bt
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from tuneforge import VERSION
from tuneforge.api.organic_router import OrganicQueryRouter
from tuneforge.settings import Settings, get_settings
from tuneforge.utils.logging import setup_logging


# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    """Container for objects shared across request handlers."""

    settings: Settings = field(default_factory=get_settings)
    dendrite: bt.Dendrite | None = None
    metagraph: bt.Metagraph | None = None
    organic_router: OrganicQueryRouter | None = None


app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialise Bittensor primitives on startup."""
    s = app_state.settings

    # Logging
    setup_logging(level=s.log_level, log_dir=s.log_dir, component_name="subnet-api")
    logger.info("TuneForge Subnet API v{} starting on {}:{}", VERSION, s.api_host, s.api_port)

    # Bittensor primitives
    try:
        app_state.dendrite = s.dendrite
        app_state.metagraph = s.metagraph
        logger.info(
            "Connected to network netuid={} block={}",
            s.netuid,
            app_state.metagraph.block if app_state.metagraph else "?",
        )
    except Exception as exc:
        logger.warning("Could not connect to Bittensor network: {} — running in offline mode", exc)

    # Organic query router
    if app_state.dendrite is not None and app_state.metagraph is not None:
        app_state.organic_router = OrganicQueryRouter(
            dendrite=app_state.dendrite,
            metagraph=app_state.metagraph,
            settings=s,
        )
        logger.info("Organic query router initialized")

    logger.info("Startup complete")
    yield

    # --- Shutdown ---
    logger.info("Shutting down TuneForge Subnet API…")
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TuneForge Subnet API",
    description="Bittensor subnet API — organic music generation via miners",
    version=VERSION,
    lifespan=lifespan,
)

# CORS
_allowed_origins = [
    get_settings().frontend_url,
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a unique request ID to every request/response."""
    req_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.opt(exception=exc).error("Unhandled exception on {}", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------

from tuneforge.api.routes.generate import router as generate_router  # noqa: E402
from tuneforge.api.routes.health import router as health_router  # noqa: E402

app.include_router(generate_router)
app.include_router(health_router)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the API server with uvicorn."""
    import uvicorn

    s = get_settings()
    uvicorn.run(
        "tuneforge.api.server:app",
        host=s.api_host,
        port=s.api_port,
        log_level=s.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
