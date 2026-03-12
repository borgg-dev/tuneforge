"""Tests for the TuneForge Subnet API server and models."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tuneforge.api.models import (
    GenerateRequest,
    HealthResponse,
    TrackInfo,
)


# ---------------------------------------------------------------------------
# Health endpoint (standalone test app with just health router)
# ---------------------------------------------------------------------------
class TestHealthEndpoint:

    @pytest.mark.asyncio
    async def test_health_returns_200(self):
        """GET /health returns 200 with status field."""
        import httpx
        from fastapi import FastAPI
        from tuneforge.api.routes.health import router as health_router

        test_app = FastAPI()
        test_app.include_router(health_router)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=test_app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "status" in data
            assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_status_endpoint(self):
        """GET /api/v1/status returns 200 with version."""
        import httpx
        from fastapi import FastAPI
        from tuneforge.api.routes.health import router as health_router

        test_app = FastAPI()
        test_app.include_router(health_router)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=test_app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/api/v1/status")
            assert resp.status_code == 200
            data = resp.json()
            assert "version" in data


# ---------------------------------------------------------------------------
# API Pydantic models
# ---------------------------------------------------------------------------
class TestAPIModels:

    def test_generate_request_defaults(self):
        req = GenerateRequest(prompt="test music")
        assert req.prompt == "test music"
        assert req.duration_seconds == 15.0
        assert req.num_variations == 1
        assert req.format == "wav"

    def test_generate_request_full(self):
        req = GenerateRequest(
            prompt="epic orchestral",
            genre="cinematic",
            mood="epic",
            tempo_bpm=100,
            duration_seconds=20.0,
            instruments=["orchestra", "choir"],
            num_variations=3,
            format="wav",
        )
        assert req.genre == "cinematic"
        assert req.num_variations == 3
        assert req.format == "wav"

    def test_generate_request_rejects_empty_prompt(self):
        with pytest.raises(Exception):
            GenerateRequest(prompt="")

    def test_track_info_model(self):
        track = TrackInfo(
            track_id="abc123",
            audio_url="http://localhost/audio/abc123.mp3",
            duration_seconds=15.0,
            sample_rate=32000,
            format="mp3",
            generation_time_ms=5000,
            miner_hotkey="5Hot0000",
            scores={"clap": 0.8, "quality": 0.7},
        )
        assert track.track_id == "abc123"
        assert track.scores["clap"] == 0.8

    def test_health_response_model(self):
        hr = HealthResponse(
            status="healthy",
            version="1.0.0",
            block_height=12345,
            connected_miners=10,
            uptime_seconds=3600.0,
        )
        assert hr.status == "healthy"
        assert hr.block_height == 12345
