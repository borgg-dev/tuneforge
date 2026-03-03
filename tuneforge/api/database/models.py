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
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
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
    audio_path = Column(Text, nullable=False)
    format = Column(String(8), nullable=False, default="mp3")
    sample_rate = Column(Integer, nullable=False, default=32000)
    generation_time_ms = Column(Integer, nullable=False, default=0)
    miner_hotkey = Column(String(64), nullable=False, default="")
    scores_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    # SaaS extensions
    user_id = Column(String(32), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    generation_id = Column(String(32), ForeignKey("generations.id", ondelete="SET NULL"), nullable=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def row_to_scores(row: TrackRow) -> dict[str, float]:
    """Parse the scores_json column into a dict."""
    try:
        return json.loads(row.scores_json)
    except (json.JSONDecodeError, TypeError):
        return {}
