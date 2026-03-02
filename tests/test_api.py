"""Tests for the FastAPI API server, rate limiter, auth, and models."""

import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tuneforge.api.rate_limiter import RateLimiter
from tuneforge.api.models import (
    BrowseResponse,
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
# Auth
# ---------------------------------------------------------------------------
class TestAuth:

    def test_no_keys_allows_all(self):
        """When TF_API_KEYS is empty, auth is disabled."""
        from tuneforge.api import auth

        auth._cached_keys = None
        with patch.dict(os.environ, {"TF_API_KEYS": ""}, clear=False):
            auth._cached_keys = None
            keys = auth._load_env_keys()
            assert len(keys) == 0

    def test_verify_valid_key(self):
        from tuneforge.api import auth

        auth._cached_keys = {"test-key-123"}
        assert auth.verify_api_key("test-key-123") is True

    def test_verify_invalid_key(self):
        from tuneforge.api import auth

        auth._cached_keys = {"test-key-123"}
        assert auth.verify_api_key("wrong-key") is False

    def test_generate_key_format(self):
        from tuneforge.api.auth import generate_api_key

        key = generate_api_key()
        assert key.startswith("tf_")
        assert len(key) > 10

    def test_reload_keys(self):
        from tuneforge.api import auth

        auth._cached_keys = {"old-key"}
        with patch.dict(os.environ, {"TF_API_KEYS": "new-key-1,new-key-2"}, clear=False):
            auth.reload_keys()
            assert "new-key-1" in auth._cached_keys
            assert "new-key-2" in auth._cached_keys


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
class TestRateLimiter:

    def test_creation(self):
        rl = RateLimiter(max_requests=10, window_seconds=60)
        assert rl is not None

    def test_allows_within_limit(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert rl.is_allowed("test_key")

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            assert rl.is_allowed("test_key")
        assert not rl.is_allowed("test_key")

    def test_separate_keys(self):
        rl = RateLimiter(max_requests=2, window_seconds=60)
        assert rl.is_allowed("key_a")
        assert rl.is_allowed("key_a")
        assert not rl.is_allowed("key_a")
        assert rl.is_allowed("key_b")

    def test_remaining(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        assert rl.remaining("key_a") == 5
        rl.is_allowed("key_a")
        assert rl.remaining("key_a") == 4

    def test_reset_time_no_requests(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        assert rl.reset_time("fresh_key") == 0.0

    def test_reset_time_after_request(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        rl.is_allowed("key_x")
        assert rl.reset_time("key_x") > 0


# ---------------------------------------------------------------------------
# API Pydantic models
# ---------------------------------------------------------------------------
class TestAPIModels:

    def test_generate_request_defaults(self):
        req = GenerateRequest(prompt="test music")
        assert req.prompt == "test music"
        assert req.duration_seconds == 15.0
        assert req.num_variations == 1
        assert req.format == "mp3"

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

    def test_browse_response_model(self):
        br = BrowseResponse(tracks=[], total=0, page=1, pages=0)
        assert br.total == 0
        assert br.tracks == []
