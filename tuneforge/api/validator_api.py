"""
Validator organic generation API.

Lightweight FastAPI server that runs inside the validator process.
Receives organic generation requests from the SaaS backend, fans out
to all serving miners, scores responses with ProductionRewardModel,
and returns the top N results ranked by composite score.

This replaces the old standalone OrganicQueryRouter which picked a
single miner by EMA and returned unscored results.
"""

import base64
import io
import time
import uuid
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from tuneforge.core.validator import TuneForgeValidator


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class OrganicGenerateRequest(BaseModel):
    """Organic generation request from the SaaS backend."""

    prompt: str = Field(..., min_length=1, max_length=2000)
    genre: str | None = Field(default=None)
    mood: str | None = Field(default=None)
    tempo_bpm: int | None = Field(default=None, ge=20, le=300)
    duration_seconds: float = Field(default=15.0, ge=1.0, le=60.0)
    key_signature: str | None = Field(default=None)
    instruments: list[str] | None = Field(default=None)
    num_variations: int = Field(default=1, ge=1, le=5)
    format: str = Field(default="wav", pattern=r"^(mp3|wav|ogg|flac)$")


class OrganicTrack(BaseModel):
    """A single scored track from an organic generation round."""

    track_id: str
    audio_b64: str
    duration_seconds: float
    sample_rate: int
    format: str
    generation_time_ms: int
    miner_uid: int
    miner_hotkey: str
    composite_score: float
    model_id: str | None = None


class OrganicGenerateResponse(BaseModel):
    """Response for organic generation endpoint."""

    request_id: str
    tracks: list[OrganicTrack]
    total_miners_queried: int
    total_valid_responses: int
    total_time_ms: int
    variations_requested: int = 1
    variations_delivered: int = 0


# ---------------------------------------------------------------------------
# Audio format conversion
# ---------------------------------------------------------------------------


def _convert_audio(raw_bytes: bytes, target_format: str) -> bytes:
    """Convert raw WAV bytes to the requested output format."""
    if target_format == "wav":
        return raw_bytes
    try:
        from pydub import AudioSegment

        segment = AudioSegment.from_file(io.BytesIO(raw_bytes), format="wav")
        buf = io.BytesIO()
        segment.export(buf, format=target_format)
        return buf.getvalue()
    except Exception as exc:
        logger.warning("Audio conversion to {} failed ({}), returning raw WAV", target_format, exc)
        return raw_bytes


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------


def create_validator_api(validator: "TuneForgeValidator") -> FastAPI:
    """Create the FastAPI app for organic generation, bound to a validator instance."""

    app = FastAPI(
        title="TuneForge Validator Organic API",
        description="Organic music generation — fan-out to all miners, score, return best",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "version": "1.0.0",
            "validator_uid": validator.uid,
            "is_running": validator.is_running,
            "current_round": validator.current_round,
        }

    @app.get("/status")
    async def get_status():
        """Detailed validator status for load-aware routing.

        Used by the platform load balancer to pick the least-loaded
        validator for organic requests.
        """
        return validator.status()

    @app.post("/organic/generate", response_model=OrganicGenerateResponse)
    async def organic_generate(request: OrganicGenerateRequest) -> OrganicGenerateResponse:
        """Fan out to all miners, score, return top N."""

        if not validator.is_running:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Validator is not running yet.",
            )

        request_id = uuid.uuid4().hex
        t_start = time.perf_counter()

        logger.info(
            "[ORGANIC] Request {}: prompt='{}' genre={} duration={}s variations={}",
            request_id,
            request.prompt[:80],
            request.genre,
            request.duration_seconds,
            request.num_variations,
        )

        try:
            ranked_results = await validator.run_organic_generation(
                prompt=request.prompt,
                genre=request.genre or "",
                mood=request.mood or "",
                tempo_bpm=request.tempo_bpm or 120,
                duration_seconds=request.duration_seconds,
                key_signature=request.key_signature,
                instruments=request.instruments,
                request_id=request_id,
            )
        except Exception as exc:
            logger.error("[ORGANIC] Request {} failed: {}", request_id, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Generation failed: {exc}",
            )

        if not ranked_results:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="No miners returned valid audio.",
            )

        # Take top N by score
        top_results = ranked_results[: request.num_variations]

        tracks: list[OrganicTrack] = []
        for result in top_results:
            converted = _convert_audio(result["audio_bytes"], request.format)
            tracks.append(
                OrganicTrack(
                    track_id=uuid.uuid4().hex,
                    audio_b64=base64.b64encode(converted).decode("ascii"),
                    duration_seconds=request.duration_seconds,
                    sample_rate=result["sample_rate"],
                    format=request.format,
                    generation_time_ms=result["generation_time_ms"],
                    miner_uid=result["miner_uid"],
                    miner_hotkey=result["miner_hotkey"],
                    composite_score=result["composite_score"],
                    model_id=result.get("model_id"),
                )
            )

        total_ms = int((time.perf_counter() - t_start) * 1000)
        logger.info(
            "[ORGANIC] Request {} complete: {} tracks from {} valid responses in {}ms",
            request_id,
            len(tracks),
            ranked_results[0]["total_valid"] if ranked_results else 0,
            total_ms,
        )

        return OrganicGenerateResponse(
            request_id=request_id,
            tracks=tracks,
            total_miners_queried=ranked_results[0]["total_queried"] if ranked_results else 0,
            total_valid_responses=ranked_results[0]["total_valid"] if ranked_results else 0,
            total_time_ms=total_ms,
            variations_requested=request.num_variations,
            variations_delivered=len(tracks),
        )

    return app
