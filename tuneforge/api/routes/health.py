"""
Health and status endpoints for the TuneForge API.

GET /health           — lightweight liveness probe
GET /api/v1/status    — detailed network status
"""

import time

from fastapi import APIRouter
from loguru import logger

from tuneforge import VERSION
from tuneforge.api.models import HealthResponse

router = APIRouter(tags=["health"])

_start_time: float = time.monotonic()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Lightweight health probe for load balancers and orchestrators."""
    from tuneforge.api.server import app_state

    block_height = 0
    connected = 0
    try:
        if app_state.metagraph is not None:
            block_height = int(app_state.metagraph.block)
            connected = sum(
                1 for ax in app_state.metagraph.axons if ax.is_serving
            )
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        version=VERSION,
        block_height=block_height,
        connected_miners=connected,
        uptime_seconds=round(time.monotonic() - _start_time, 1),
    )


@router.get("/api/v1/status", response_model=HealthResponse)
async def detailed_status() -> HealthResponse:
    """Detailed network status for dashboards and monitoring."""
    from tuneforge.api.server import app_state

    block_height = 0
    connected = 0
    metagraph = app_state.metagraph

    if metagraph is not None:
        try:
            block_height = int(metagraph.block)
            connected = sum(1 for ax in metagraph.axons if ax.is_serving)
        except Exception as exc:
            logger.warning("Failed to read metagraph stats: {}", exc)

    return HealthResponse(
        status="ok",
        version=VERSION,
        block_height=block_height,
        connected_miners=connected,
        uptime_seconds=round(time.monotonic() - _start_time, 1),
    )
