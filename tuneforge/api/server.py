"""
FastAPI application for the TuneForge marketplace API.

Provides REST endpoints for music generation, track browsing,
user auth, credits, and network health monitoring.
"""

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

import bittensor as bt
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from tuneforge import VERSION
from tuneforge.api import rate_limiter
from tuneforge.api.billing import UsageTracker
from tuneforge.api.credits import CreditService
from tuneforge.api.database import Database
from tuneforge.api.organic_router import OrganicQueryRouter
from tuneforge.api.redis_manager import RedisManager
from tuneforge.api.sse import SSEManager
from tuneforge.api.storage import StorageBackend, get_storage_backend
from tuneforge.settings import Settings, get_settings
from tuneforge.utils.logging import setup_logging


# ---------------------------------------------------------------------------
# Shared application state — initialised during lifespan startup
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    """Container for objects shared across request handlers."""

    settings: Settings = field(default_factory=get_settings)
    dendrite: bt.Dendrite | None = None
    metagraph: bt.Metagraph | None = None
    storage: StorageBackend | None = None
    db: Database | None = None
    billing: UsageTracker = field(default_factory=UsageTracker)
    # SaaS additions
    redis: RedisManager | None = None
    organic_router: OrganicQueryRouter | None = None
    credit_service: CreditService | None = None
    sse_manager: SSEManager | None = None


app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialise resources on startup, tear down on shutdown."""
    s = app_state.settings

    # Logging
    setup_logging(level=s.log_level, log_dir=s.log_dir, component_name="api")
    logger.info("TuneForge API v{} starting on {}:{}", VERSION, s.api_host, s.api_port)

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

    # Storage backend
    app_state.storage = get_storage_backend(
        backend=s.storage_backend,
        storage_path=s.storage_path,
        s3_bucket=s.s3_bucket,
        s3_region=s.s3_region,
        s3_access_key=s.s3_access_key,
        s3_secret_key=s.s3_secret_key,
    )

    # Database
    app_state.db = Database(url=s.db_url)
    await app_state.db.init_db()

    # Redis
    try:
        rm = RedisManager(url=s.redis_url)
        await rm.ping()
        app_state.redis = rm
        app_state.sse_manager = SSEManager(app_state.redis)
        logger.info("Redis connected")
    except Exception as exc:
        logger.warning("Could not connect to Redis: {} — SSE and Redis rate-limiting unavailable", exc)
        app_state.redis = None

    # Credit service
    if app_state.db is not None:
        app_state.credit_service = CreditService(app_state.db)

    # Organic query router
    if app_state.dendrite is not None and app_state.metagraph is not None:
        app_state.organic_router = OrganicQueryRouter(
            dendrite=app_state.dendrite,
            metagraph=app_state.metagraph,
            settings=s,
        )
        logger.info("Organic query router initialized")

    # Rate limiter
    rate_limiter.configure(max_requests=s.api_rate_limit, window_seconds=60)

    # Mount static files for local storage
    if s.storage_backend == "local":
        from pathlib import Path

        static_dir = Path(s.storage_path)
        static_dir.mkdir(parents=True, exist_ok=True)
        app.mount("/static/audio", StaticFiles(directory=str(static_dir)), name="audio")

    logger.info("Startup complete")
    yield

    # --- Shutdown ---
    logger.info("Shutting down TuneForge API…")
    if app_state.redis:
        await app_state.redis.close()
    if app_state.db:
        await app_state.db.close()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TuneForge API",
    description="AI music generation marketplace powered by Bittensor",
    version=VERSION,
    lifespan=lifespan,
)

# CORS — configurable origins
_allowed_origins = [
    get_settings().frontend_url,
    "http://localhost:3000",
    "http://192.168.1.83:3000",
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
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.opt(exception=exc).error("Unhandled exception on {}", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------

from tuneforge.api.routes.auth import router as auth_router  # noqa: E402
from tuneforge.api.routes.browse import router as browse_router  # noqa: E402
from tuneforge.api.routes.credits import router as credits_router  # noqa: E402
from tuneforge.api.routes.generate import router as generate_router  # noqa: E402
from tuneforge.api.routes.health import router as health_router  # noqa: E402
from tuneforge.api.routes.keys import router as keys_router  # noqa: E402

app.include_router(auth_router)
app.include_router(generate_router)
app.include_router(browse_router)
app.include_router(credits_router)
app.include_router(keys_router)
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
