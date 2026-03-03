"""
Pydantic models for the TuneForge API.

Defines request/response schemas for auth, credits, music generation,
browsing, and health endpoints.
"""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


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


# ---------------------------------------------------------------------------
# Auth models
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    """Registration request."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    display_name: str | None = Field(default=None, max_length=100)


class LoginRequest(BaseModel):
    """Login request."""

    email: EmailStr
    password: str


class UserProfile(BaseModel):
    """Public user profile."""

    id: str
    email: str
    display_name: str | None
    avatar_url: str | None
    plan_tier: str
    created_at: datetime


class AuthResponse(BaseModel):
    """Response containing tokens and user profile."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: UserProfile


class TokenResponse(BaseModel):
    """Refreshed token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UpdateProfileRequest(BaseModel):
    """Profile update request."""

    display_name: str | None = Field(default=None, max_length=100)
    avatar_url: str | None = Field(default=None, max_length=2000)


class RefreshRequest(BaseModel):
    """Token refresh request body (alternative to cookie)."""

    refresh_token: str


# ---------------------------------------------------------------------------
# Credit models
# ---------------------------------------------------------------------------


class CreditBalance(BaseModel):
    """Current credit balance."""

    daily_balance: int
    daily_allowance: int
    next_reset: datetime


class CreditTransaction(BaseModel):
    """Single credit transaction."""

    id: str
    amount: int
    tx_type: str
    reference_id: str | None
    description: str | None
    created_at: datetime


class CreditHistoryResponse(BaseModel):
    """Paginated credit transaction history."""

    transactions: list[CreditTransaction]
    total: int
    page: int
    pages: int


# ---------------------------------------------------------------------------
# Generation status models
# ---------------------------------------------------------------------------


class GenerationStatusResponse(BaseModel):
    """Generation status for polling."""

    request_id: str
    status: str
    tracks: list[TrackInfo] = Field(default_factory=list)
    created_at: datetime
    completed_at: datetime | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# API Key models
# ---------------------------------------------------------------------------


class CreateApiKeyRequest(BaseModel):
    """Request to create a new API key."""

    name: str = Field(default="Default", max_length=100)


class ApiKeyInfo(BaseModel):
    """API key info (without the full key)."""

    id: str
    name: str
    key_prefix: str
    last_used_at: datetime | None
    created_at: datetime


class ApiKeyCreated(BaseModel):
    """Response when a new API key is created (shows full key once)."""

    id: str
    name: str
    key: str  # Full key — only shown once
    key_prefix: str
    created_at: datetime
