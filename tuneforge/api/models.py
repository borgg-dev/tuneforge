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
    is_admin: bool = False
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


# ---------------------------------------------------------------------------
# Validator round models
# ---------------------------------------------------------------------------


class MinerResponseEntry(BaseModel):
    """A single miner's response within a validation round submission."""

    miner_uid: int = Field(..., ge=0)
    miner_hotkey: str = Field(..., max_length=64)
    audio_b64: str = Field(..., description="Base64-encoded WAV audio")
    generation_time_ms: int | None = Field(default=None, ge=0)


class SubmitValidationRoundRequest(BaseModel):
    """Submit a complete validation round with miner audio responses."""

    challenge_id: str = Field(..., max_length=64)
    prompt: str = Field(..., min_length=1, max_length=2000)
    genre: str | None = Field(default=None, max_length=64)
    mood: str | None = Field(default=None, max_length=64)
    tempo_bpm: int | None = Field(default=None, ge=20, le=300)
    duration_seconds: float = Field(..., ge=1.0, le=60.0)
    validator_hotkey: str = Field(default="", max_length=64)
    responses: list[MinerResponseEntry] = Field(..., min_length=1, max_length=256)


class ValidationAudioInfo(BaseModel):
    """Info about a saved miner audio entry."""

    miner_uid: int
    miner_hotkey: str | None = None
    audio_blob_id: str
    generation_time_ms: int | None = None
    score: float | None = None


class SubmitValidationRoundResponse(BaseModel):
    """Response after submitting a validation round."""

    round_id: str
    challenge_id: str
    audio_entries: list[ValidationAudioInfo]


class ScoreEntry(BaseModel):
    """A single score update for a miner."""

    miner_uid: int = Field(..., ge=0)
    score: float = Field(..., ge=0.0, le=1.0)


class UpdateScoresRequest(BaseModel):
    """Update scores for a completed round."""

    scores: list[ScoreEntry] = Field(..., min_length=1, max_length=256)


class UpdateScoresResponse(BaseModel):
    """Response after updating scores."""

    updated: int


class ValidationRoundInfo(BaseModel):
    """Validation round metadata."""

    id: str
    challenge_id: str
    prompt: str
    genre: str | None = None
    mood: str | None = None
    tempo_bpm: int | None = None
    duration_seconds: float
    validator_hotkey: str | None = None
    created_at: datetime


class ValidationRoundListResponse(BaseModel):
    """Paginated list of validation rounds."""

    rounds: list[ValidationRoundInfo]
    total: int
    page: int
    pages: int


class ValidationRoundDetailResponse(BaseModel):
    """Single validation round with audio entries."""

    round: ValidationRoundInfo
    audio_entries: list[ValidationAudioInfo]


# ---------------------------------------------------------------------------
# Annotation models
# ---------------------------------------------------------------------------


class GenerateAnnotationTasksRequest(BaseModel):
    """Generate A/B annotation tasks from validation rounds."""

    round_ids: list[str] | None = Field(default=None, description="Round IDs to process, or null for all")
    quorum: int = Field(default=5, ge=3, le=15, description="Votes needed per task")


class GenerateAnnotationTasksResponse(BaseModel):
    """Result of task generation."""

    tasks_created: int
    tasks_skipped: int
    total_tasks: int


class AnnotationProgress(BaseModel):
    """User's annotation progress."""

    total_open: int
    completed_by_user: int
    remaining_for_user: int


class AnnotationTaskDetail(BaseModel):
    """A single annotation task with audio URLs."""

    task_id: str
    prompt: str
    genre: str | None = None
    mood: str | None = None
    tempo_bpm: int | None = None
    duration_seconds: float
    audio_a_url: str
    audio_b_url: str
    progress: AnnotationProgress


class SubmitVoteRequest(BaseModel):
    """Submit a preference vote."""

    choice: str = Field(..., pattern=r"^[ab]$")
    duration_ms: int | None = Field(default=None, ge=0)


class MilestoneUnlocked(BaseModel):
    """Info about a milestone just unlocked after a vote."""

    key: str
    label: str
    credits_awarded: int
    grants_pro: bool = False


class RecurringRewardEarned(BaseModel):
    """Info about a recurring reward just earned after a vote."""

    credits_awarded: int
    batches: int
    streak_days: int
    multiplier: float
    tier_label: str


class VoteResponse(BaseModel):
    """Response after submitting a vote."""

    recorded: bool
    task_status: str
    next_task: AnnotationTaskDetail | None = None
    milestone_unlocked: MilestoneUnlocked | None = None
    recurring_reward: RecurringRewardEarned | None = None


class AgreementStats(BaseModel):
    """Aggregate annotation statistics."""

    total_tasks: int
    open_tasks: int
    completed_tasks: int
    discarded_tasks: int
    total_annotations: int
    annotator_count: int
    total_gold_tasks: int = 0
    flagged_annotator_count: int = 0


class AnnotationTaskInfo(BaseModel):
    """Task info for admin listing."""

    id: str
    round_id: str
    prompt: str
    genre: str | None = None
    audio_a_blob_id: str
    audio_b_blob_id: str
    quorum: int
    vote_count: int
    status: str
    winner: str | None = None
    created_at: datetime


class AnnotationTaskListResponse(BaseModel):
    """Paginated task listing."""

    tasks: list[AnnotationTaskInfo]
    total: int
    page: int
    pages: int


class AnnotationResultsResponse(BaseModel):
    """Aggregation results for completed tasks."""

    results: list[AnnotationTaskInfo]
    total: int
    page: int
    pages: int
    stats: AgreementStats


class AnnotationExportEntry(BaseModel):
    """Single exported annotation for training."""

    pair_id: str
    audio_a: str
    audio_b: str
    preferred: str
    confidence: float = 1.0
    n_trusted_votes: int = 0
    challenge_id: str
    prompt: str
    genre: str | None = None


class PreferenceModelInfo(BaseModel):
    """Metadata about a preference model checkpoint."""

    id: str
    version: int
    sha256: str
    val_accuracy: float | None = None
    n_train_pairs: int | None = None
    is_active: bool
    created_at: datetime


# ---------------------------------------------------------------------------
# Gold Standard Quality Control
# ---------------------------------------------------------------------------


class CreateGoldTaskRequest(BaseModel):
    """Create a gold/honeypot annotation task with a known answer."""

    audio_a_id: str
    audio_b_id: str
    gold_answer: str = Field(..., pattern=r"^[ab]$")
    round_id: str


class GoldTaskResponse(BaseModel):
    """Response for a single gold task."""

    task_id: str
    audio_a_id: str
    audio_b_id: str
    gold_answer: str
    vote_count: int = 0
    gold_accuracy: float | None = None
    created_at: datetime


class GoldTaskListResponse(BaseModel):
    """Paginated gold task listing."""

    tasks: list[GoldTaskResponse]
    total: int
    page: int
    pages: int


class AnnotatorReliabilityInfo(BaseModel):
    """Annotator reliability information."""

    user_id: str
    email: str
    display_name: str | None = None
    gold_total: int
    gold_correct: int
    accuracy: float
    is_flagged: bool
    flagged_at: datetime | None = None
    total_annotations: int = 0


class AnnotatorListResponse(BaseModel):
    """Paginated annotator reliability listing."""

    annotators: list[AnnotatorReliabilityInfo]
    total: int
    page: int
    pages: int


# ---------------------------------------------------------------------------
# Milestone models
# ---------------------------------------------------------------------------


class MilestoneInfo(BaseModel):
    """A single milestone definition with claim status."""

    key: str
    label: str
    threshold: int
    credits: int
    grants_pro: bool = False
    is_claimed: bool = False


class NextMilestoneInfo(BaseModel):
    """Progress toward the next unclaimed milestone."""

    key: str
    label: str
    threshold: int
    credits: int
    grants_pro: bool = False
    current_count: int
    remaining: int


class NextStreakTierInfo(BaseModel):
    """Info about the next streak tier."""

    label: str
    min_days: int
    multiplier: float
    days_remaining: int


class StreakInfo(BaseModel):
    """Full streak/recurring reward info for a user."""

    is_recurring_eligible: bool
    current_streak_days: int = 0
    streak_multiplier: float = 1.0
    streak_tier_label: str = "Base"
    credits_per_batch: int = 30
    batch_size: int = 20
    annotations_toward_next_batch: int = 0
    annotations_remaining: int = 20
    total_recurring_credits_earned: int = 0
    total_recurring_batches_claimed: int = 0
    today_annotation_count: int = 0
    daily_streak_threshold: int = 10
    streak_at_risk: bool = False
    grace_used: bool = False
    next_tier: NextStreakTierInfo | None = None


class MilestoneProgressResponse(BaseModel):
    """Full milestone progress for a user."""

    trusted_annotation_count: int
    completed_milestones: list[MilestoneInfo]
    next_milestone: NextMilestoneInfo | None = None
    streak: StreakInfo | None = None
