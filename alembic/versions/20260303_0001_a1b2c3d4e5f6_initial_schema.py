"""Initial schema — users, api_keys, credits, generations, tracks.

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-03-03 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Users ---
    op.create_table(
        "users",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("email", sa.String(320), unique=True, nullable=False),
        sa.Column("password_hash", sa.String(128), nullable=True),
        sa.Column("display_name", sa.String(100), nullable=True),
        sa.Column("avatar_url", sa.Text, nullable=True),
        sa.Column("plan_tier", sa.String(16), nullable=False, server_default="free"),
        sa.Column("auth_provider", sa.String(16), nullable=False, server_default="email"),
        sa.Column("google_id", sa.String(64), nullable=True, unique=True),
        sa.Column("email_verified", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_google_id", "users", ["google_id"])

    # --- API Keys ---
    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("key_prefix", sa.String(12), nullable=False),
        sa.Column("key_hash", sa.String(128), nullable=False, unique=True),
        sa.Column("name", sa.String(100), nullable=False, server_default="Default"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"])

    # --- Credits ---
    op.create_table(
        "credits",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("daily_balance", sa.Integer, nullable=False, server_default=sa.text("50")),
        sa.Column("daily_allowance", sa.Integer, nullable=False, server_default=sa.text("50")),
        sa.Column("last_daily_reset", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_credits_user_id", "credits", ["user_id"])

    # --- Credit Transactions ---
    op.create_table(
        "credit_transactions",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("amount", sa.Integer, nullable=False),
        sa.Column("tx_type", sa.String(16), nullable=False),
        sa.Column("reference_id", sa.String(64), nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_credit_transactions_user_id", "credit_transactions", ["user_id"])

    # --- Generations ---
    op.create_table(
        "generations",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("request_id", sa.String(64), unique=True, nullable=False),
        sa.Column("prompt", sa.Text, nullable=False),
        sa.Column("params_json", sa.Text, nullable=False, server_default="{}"),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("credits_reserved", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("credits_spent", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_generations_user_id", "generations", ["user_id"])
    op.create_index("ix_generations_request_id", "generations", ["request_id"])

    # --- Tracks ---
    op.create_table(
        "tracks",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("request_id", sa.String(64), nullable=False),
        sa.Column("prompt", sa.Text, nullable=False),
        sa.Column("genre", sa.String(64), nullable=True),
        sa.Column("mood", sa.String(64), nullable=True),
        sa.Column("tempo_bpm", sa.Integer, nullable=True),
        sa.Column("duration_seconds", sa.Float, nullable=False),
        sa.Column("audio_path", sa.Text, nullable=False),
        sa.Column("format", sa.String(8), nullable=False, server_default="mp3"),
        sa.Column("sample_rate", sa.Integer, nullable=False, server_default=sa.text("32000")),
        sa.Column("generation_time_ms", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("miner_hotkey", sa.String(64), nullable=False, server_default=""),
        sa.Column("scores_json", sa.Text, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("generation_id", sa.String(32), sa.ForeignKey("generations.id", ondelete="SET NULL"), nullable=True),
    )
    op.create_index("ix_tracks_request_id", "tracks", ["request_id"])
    op.create_index("ix_tracks_user_id", "tracks", ["user_id"])


def downgrade() -> None:
    op.drop_table("tracks")
    op.drop_table("generations")
    op.drop_table("credit_transactions")
    op.drop_table("credits")
    op.drop_table("api_keys")
    op.drop_table("users")
