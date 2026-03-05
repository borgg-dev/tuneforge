"""
Pydantic models for the TuneForge Subnet API.

Only contains models for the organic generation endpoint and health.
SaaS models (auth, credits, annotations, etc.) live in tuneforge-web/api.
"""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for music generation via organic router."""

    prompt: str = Field(..., min_length=1, max_length=2000)
    genre: str | None = Field(default=None)
    mood: str | None = Field(default=None)
    tempo_bpm: int | None = Field(default=None, ge=20, le=300)
    duration_seconds: float = Field(default=15.0, ge=1.0, le=60.0)
    key_signature: str | None = Field(default=None)
    instruments: list[str] | None = Field(default=None)
    num_variations: int = Field(default=1, ge=1, le=5)
    format: str = Field(default="wav", pattern=r"^(mp3|wav|ogg|flac)$")


class TrackInfo(BaseModel):
    """Information about a single generated track."""

    track_id: str
    audio_url: str
    duration_seconds: float
    sample_rate: int
    format: str
    generation_time_ms: int
    miner_hotkey: str
    scores: dict[str, float] = Field(default_factory=dict)


class GenerateResponse(BaseModel):
    """Response for music generation endpoint."""

    request_id: str
    tracks: list[TrackInfo]
    total_time_ms: int


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: str
    version: str
    block_height: int
    connected_miners: int
    uptime_seconds: float
