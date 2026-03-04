"""Add PostgreSQL audio storage tables.

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-04

New tables: audio_blobs, validation_rounds, validation_audio
Altered: tracks (add audio_blob_id FK, make audio_path nullable)
"""

from alembic import op
import sqlalchemy as sa

revision = "b2c3d4e5f6g7"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. audio_blobs — binary audio storage
    op.create_table(
        "audio_blobs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("audio_data", sa.LargeBinary, nullable=False),
        sa.Column("format", sa.String(8), nullable=False, server_default="wav"),
        sa.Column("size_bytes", sa.Integer, nullable=False),
        sa.Column("content_type", sa.String(32), nullable=False, server_default="audio/wav"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # 2. validation_rounds — challenge metadata
    op.create_table(
        "validation_rounds",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("challenge_id", sa.String(64), unique=True, nullable=False),
        sa.Column("prompt", sa.Text, nullable=False),
        sa.Column("genre", sa.String(64), nullable=True),
        sa.Column("mood", sa.String(64), nullable=True),
        sa.Column("tempo_bpm", sa.Integer, nullable=True),
        sa.Column("duration_seconds", sa.Float, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_validation_rounds_challenge_id", "validation_rounds", ["challenge_id"])

    # 3. validation_audio — per-miner audio in a round
    op.create_table(
        "validation_audio",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("round_id", sa.String(32), sa.ForeignKey("validation_rounds.id", ondelete="CASCADE"), nullable=False),
        sa.Column("miner_uid", sa.Integer, nullable=False),
        sa.Column("miner_hotkey", sa.String(64), nullable=True),
        sa.Column("audio_blob_id", sa.String(32), sa.ForeignKey("audio_blobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("generation_time_ms", sa.Integer, nullable=True),
        sa.Column("score", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_validation_audio_round_id", "validation_audio", ["round_id"])

    # 4. Alter tracks — add audio_blob_id FK, make audio_path nullable
    op.add_column(
        "tracks",
        sa.Column("audio_blob_id", sa.String(32), sa.ForeignKey("audio_blobs.id", ondelete="SET NULL"), nullable=True),
    )
    op.alter_column("tracks", "audio_path", existing_type=sa.Text, nullable=True)


def downgrade() -> None:
    op.alter_column("tracks", "audio_path", existing_type=sa.Text, nullable=False)
    op.drop_column("tracks", "audio_blob_id")
    op.drop_table("validation_audio")
    op.drop_table("validation_rounds")
    op.drop_table("audio_blobs")
