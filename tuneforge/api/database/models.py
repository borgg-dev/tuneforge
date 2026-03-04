"""
ORM models for the TuneForge SaaS platform.

Defines all database tables for users, authentication, credits,
generations, and tracks.
"""

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def _uuid_hex() -> str:
    return uuid.uuid4().hex


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Users & Authentication
# ---------------------------------------------------------------------------


class UserRow(Base):
    """Registered user account."""

    __tablename__ = "users"

    id = Column(String(32), primary_key=True, default=_uuid_hex)
    email = Column(String(320), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=True)  # Null for OAuth-only users
    display_name = Column(String(100), nullable=True)
    avatar_url = Column(Text, nullable=True)
    plan_tier = Column(String(16), nullable=False, default="free")  # free/pro/premier/api
    auth_provider = Column(String(16), nullable=False, default="email")  # email/google
    google_id = Column(String(64), nullable=True, unique=True, index=True)
    email_verified = Column(Boolean, nullable=False, default=False)
    is_active = Column(Boolean, nullable=False, default=True)
    is_admin = Column(Boolean, nullable=False, default=False)
    pro_expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now)


class ApiKeyRow(Base):
    """User-managed API keys for programmatic access."""

    __tablename__ = "api_keys"

    id = Column(String(32), primary_key=True, default=_uuid_hex)
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    key_prefix = Column(String(12), nullable=False)  # "tf_abc..." for display
    key_hash = Column(String(128), nullable=False, unique=True)  # SHA-256 hash
    name = Column(String(100), nullable=False, default="Default")
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    revoked_at = Column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Credits & Billing
# ---------------------------------------------------------------------------


class CreditRow(Base):
    """Per-user credit balance tracking."""

    __tablename__ = "credits"

    id = Column(String(32), primary_key=True, default=_uuid_hex)
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    daily_balance = Column(Integer, nullable=False, default=50)
    daily_allowance = Column(Integer, nullable=False, default=50)
    last_daily_reset = Column(DateTime(timezone=True), nullable=False, default=_utc_now)


class CreditTransactionRow(Base):
    """Credit transaction log."""

    __tablename__ = "credit_transactions"

    id = Column(String(32), primary_key=True, default=_uuid_hex)
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    amount = Column(Integer, nullable=False)  # Positive = grant/refund, negative = spend
    tx_type = Column(String(16), nullable=False)  # daily_grant, spend, refund
    reference_id = Column(String(64), nullable=True)  # generation request_id
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)


# ---------------------------------------------------------------------------
# Generations
# ---------------------------------------------------------------------------


class GenerationRow(Base):
    """Tracks the lifecycle of a music generation request."""

    __tablename__ = "generations"

    id = Column(String(32), primary_key=True, default=_uuid_hex)
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    request_id = Column(String(64), unique=True, nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    params_json = Column(Text, nullable=False, default="{}")
    status = Column(String(16), nullable=False, default="queued")  # queued/routing/generating/completed/failed
    credits_reserved = Column(Integer, nullable=False, default=0)
    credits_spent = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    completed_at = Column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Audio Storage
# ---------------------------------------------------------------------------


class AudioBlobRow(Base):
    """Binary audio data stored in PostgreSQL."""

    __tablename__ = "audio_blobs"

    id = Column(String(32), primary_key=True, default=_uuid_hex)
    audio_data = Column(LargeBinary, nullable=False)
    format = Column(String(8), nullable=False, default="wav")
    size_bytes = Column(Integer, nullable=False)
    content_type = Column(String(32), nullable=False, default="audio/wav")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)


# ---------------------------------------------------------------------------
# Validation Rounds
# ---------------------------------------------------------------------------


class ValidationRoundRow(Base):
    """Metadata for a single validator challenge round."""

    __tablename__ = "validation_rounds"
    __table_args__ = (
        UniqueConstraint("challenge_id", "validator_hotkey", name="uq_validation_rounds_challenge_validator"),
    )

    id = Column(String(32), primary_key=True, default=_uuid_hex)
    challenge_id = Column(String(64), nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    genre = Column(String(64), nullable=True)
    mood = Column(String(64), nullable=True)
    tempo_bpm = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=False)
    validator_hotkey = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)


class ValidationAudioRow(Base):
    """Per-miner audio response within a validation round."""

    __tablename__ = "validation_audio"

    id = Column(String(32), primary_key=True, default=_uuid_hex)
    round_id = Column(String(32), ForeignKey("validation_rounds.id", ondelete="CASCADE"), nullable=False, index=True)
    miner_uid = Column(Integer, nullable=False)
    miner_hotkey = Column(String(64), nullable=True)
    audio_blob_id = Column(String(32), ForeignKey("audio_blobs.id", ondelete="CASCADE"), nullable=False)
    generation_time_ms = Column(Integer, nullable=True)
    score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)


# ---------------------------------------------------------------------------
# Tracks (extended from original schema)
# ---------------------------------------------------------------------------


class TrackRow(Base):
    """Persisted track metadata."""

    __tablename__ = "tracks"

    id = Column(String(32), primary_key=True)
    request_id = Column(String(64), index=True, nullable=False)
    prompt = Column(Text, nullable=False)
    genre = Column(String(64), nullable=True)
    mood = Column(String(64), nullable=True)
    tempo_bpm = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=False)
    audio_path = Column(Text, nullable=True)  # Legacy; nullable for postgres storage
    format = Column(String(8), nullable=False, default="mp3")
    sample_rate = Column(Integer, nullable=False, default=32000)
    generation_time_ms = Column(Integer, nullable=False, default=0)
    miner_hotkey = Column(String(64), nullable=False, default="")
    scores_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    # SaaS extensions
    user_id = Column(String(32), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    generation_id = Column(String(32), ForeignKey("generations.id", ondelete="SET NULL"), nullable=True)
    audio_blob_id = Column(String(32), ForeignKey("audio_blobs.id", ondelete="SET NULL"), nullable=True)


# ---------------------------------------------------------------------------
# Annotations (Crowd Preference Labeling)
# ---------------------------------------------------------------------------


class AnnotationTaskRow(Base):
    """A/B comparison pair to be rated by multiple annotators."""

    __tablename__ = "annotation_tasks"
    __table_args__ = (
        UniqueConstraint("audio_a_id", "audio_b_id", name="uq_annotation_tasks_pair"),
    )

    id = Column(String(32), primary_key=True, default=_uuid_hex)
    round_id = Column(String(32), ForeignKey("validation_rounds.id", ondelete="CASCADE"), nullable=False, index=True)
    audio_a_id = Column(String(32), ForeignKey("validation_audio.id", ondelete="CASCADE"), nullable=False)
    audio_b_id = Column(String(32), ForeignKey("validation_audio.id", ondelete="CASCADE"), nullable=False)
    quorum = Column(Integer, nullable=False, default=5)  # votes needed before aggregation
    status = Column(String(16), nullable=False, default="open")  # open/completed/discarded
    winner = Column(String(1), nullable=True)  # "a" or "b"
    vote_count = Column(Integer, nullable=False, default=0)
    is_gold = Column(Boolean, nullable=False, default=False, index=True)
    gold_answer = Column(String(1), nullable=True)  # "a" or "b"; only set when is_gold=True
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class AnnotationRow(Base):
    """Individual annotator's vote on an A/B task."""

    __tablename__ = "annotations"
    __table_args__ = (
        UniqueConstraint("task_id", "user_id", name="uq_annotations_user_task"),
    )

    id = Column(String(32), primary_key=True, default=_uuid_hex)
    task_id = Column(String(32), ForeignKey("annotation_tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    choice = Column(String(1), nullable=False)  # "a" or "b"
    duration_ms = Column(Integer, nullable=True)  # time taken to decide
    is_gold_response = Column(Boolean, nullable=False, default=False)
    is_trusted = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)


class AnnotatorReliabilityRow(Base):
    """Rolling reliability score per annotator based on gold task performance."""

    __tablename__ = "annotator_reliability"

    id = Column(String(32), primary_key=True, default=_uuid_hex)
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    gold_total = Column(Integer, nullable=False, default=0)
    gold_correct = Column(Integer, nullable=False, default=0)
    accuracy = Column(Float, nullable=False, default=1.0)
    is_flagged = Column(Boolean, nullable=False, default=False, index=True)
    flagged_at = Column(DateTime(timezone=True), nullable=True)
    unflagged_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now)


class AnnotationMilestoneRow(Base):
    """Record of a user claiming an annotation milestone reward."""

    __tablename__ = "annotation_milestones"
    __table_args__ = (
        UniqueConstraint("user_id", "milestone_key", name="uq_annotation_milestones_user_key"),
    )

    id = Column(String(32), primary_key=True, default=_uuid_hex)
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    milestone_key = Column(String(20), nullable=False)
    credits_awarded = Column(Integer, nullable=False)
    claimed_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)


class AnnotationStreakRow(Base):
    """Per-user streak and recurring reward state for annotations."""

    __tablename__ = "annotation_streaks"

    id = Column(String(32), primary_key=True, default=_uuid_hex)
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    current_streak_days = Column(Integer, nullable=False, default=0)
    last_active_date = Column(Date, nullable=True)
    grace_used = Column(Boolean, nullable=False, default=False)
    recurring_annotations_count = Column(Integer, nullable=False, default=0)
    total_recurring_credits_earned = Column(Integer, nullable=False, default=0)
    total_recurring_batches_claimed = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now)


class PreferenceModelRow(Base):
    """Trained preference model checkpoint metadata."""

    __tablename__ = "preference_models"

    id = Column(String(32), primary_key=True, default=_uuid_hex)
    version = Column(Integer, nullable=False, index=True)
    sha256 = Column(String(64), nullable=False)
    val_accuracy = Column(Float, nullable=True)
    n_train_pairs = Column(Integer, nullable=True)
    is_active = Column(Boolean, nullable=False, default=False)
    model_data = Column(LargeBinary, nullable=True)  # checkpoint bytes (~200KB)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def row_to_scores(row: TrackRow) -> dict[str, float]:
    """Parse the scores_json column into a dict."""
    try:
        return json.loads(row.scores_json)
    except (json.JSONDecodeError, TypeError):
        return {}
