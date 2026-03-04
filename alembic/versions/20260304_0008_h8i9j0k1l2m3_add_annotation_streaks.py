"""Add annotation streak tracking table for recurring rewards.

Revision ID: h8i9j0k1l2m3
Revises: g7h8i9j0k1l2
Create Date: 2026-03-04
"""

from alembic import op
import sqlalchemy as sa

revision = "h8i9j0k1l2m3"
down_revision = "g7h8i9j0k1l2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "annotation_streaks",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("current_streak_days", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_active_date", sa.Date(), nullable=True),
        sa.Column("grace_used", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("recurring_annotations_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_recurring_credits_earned", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_recurring_batches_claimed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_annotation_streaks_user_id", "annotation_streaks", ["user_id"])


def downgrade() -> None:
    op.drop_table("annotation_streaks")
