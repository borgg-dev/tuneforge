"""Add validator_hotkey to validation_rounds.

Revision ID: c3d4e5f6g7h8
Revises: b2c3d4e5f6g7
Create Date: 2026-03-04

Tracks which validator submitted each round for multi-validator support.
"""

from alembic import op
import sqlalchemy as sa

revision = "c3d4e5f6g7h8"
down_revision = "b2c3d4e5f6g7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "validation_rounds",
        sa.Column("validator_hotkey", sa.String(64), nullable=True),
    )
    op.create_index(
        "ix_validation_rounds_validator_hotkey",
        "validation_rounds",
        ["validator_hotkey"],
    )


def downgrade() -> None:
    op.drop_index("ix_validation_rounds_validator_hotkey", "validation_rounds")
    op.drop_column("validation_rounds", "validator_hotkey")
