"""Add annotation milestones table and pro_expires_at to users.

Revision ID: g7h8i9j0k1l2
Revises: f6g7h8i9j0k1
Create Date: 2026-03-04
"""

from alembic import op
import sqlalchemy as sa

revision = "g7h8i9j0k1l2"
down_revision = "f6g7h8i9j0k1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # New table: annotation_milestones
    op.create_table(
        "annotation_milestones",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("milestone_key", sa.String(20), nullable=False),
        sa.Column("credits_awarded", sa.Integer(), nullable=False),
        sa.Column(
            "claimed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "user_id", "milestone_key", name="uq_annotation_milestones_user_key"
        ),
    )
    op.create_index(
        "ix_annotation_milestones_user_id",
        "annotation_milestones",
        ["user_id"],
    )

    # New column on users for Pro tier expiry
    op.add_column(
        "users",
        sa.Column("pro_expires_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "pro_expires_at")
    op.drop_table("annotation_milestones")
