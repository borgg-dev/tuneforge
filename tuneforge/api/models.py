"""
Pydantic models for the TuneForge API.

Defines request/response schemas for music generation, browsing,
and health endpoints.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for music generation."""

    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt describing desired music")
    genre: str | None = Field(default=None, description="Target genre")
    mood: str | None = Field(default=None, description="Target mood")
    tempo_bpm: int | None = Field(default=None, ge=20, le=300, description="Desired tempo in BPM")
    duration_seconds: float = Field(default=15.0, ge=1.0, le=60.0, description="Audio duration in seconds")
    key_signature: str | None = Field(default=None, description="Musical key signature")
    instruments: list[str] | None = Field(default=None, description="Preferred instruments")
    num_variations: int = Field(default=1, ge=1, le=5, description="Number of variations to generate")
    format: str = Field(default="mp3", pattern=r"^(mp3|wav|ogg|flac)$", description="Output audio format")


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


class BrowseRequest(BaseModel):
    """Query parameters for browsing tracks."""

    genre: str | None = None
    mood: str | None = None
    min_tempo: int | None = Field(default=None, ge=20, le=300)
    max_tempo: int | None = Field(default=None, ge=20, le=300)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class TrackMetadata(BaseModel):
    """Metadata for a track in browse results."""

    track_id: str
    prompt: str
    genre: str | None = None
    mood: str | None = None
    tempo_bpm: int | None = None
    duration_seconds: float
    audio_url: str
    scores: dict[str, float] = Field(default_factory=dict)
    miner_hotkey: str
    created_at: datetime


class BrowseResponse(BaseModel):
    """Paginated response for track browsing."""

    tracks: list[TrackMetadata]
    total: int
    page: int
    pages: int


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: str
    version: str
    block_height: int
    connected_miners: int
    uptime_seconds: float
