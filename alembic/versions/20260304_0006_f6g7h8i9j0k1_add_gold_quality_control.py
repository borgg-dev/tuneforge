"""Add gold standard quality control for annotations.

Adds:
- is_admin to users
- is_gold, gold_answer to annotation_tasks
- is_gold_response, is_trusted to annotations
- annotator_reliability table

Revision ID: f6g7h8i9j0k1
Revises: e5f6g7h8i9j0
Create Date: 2026-03-04
"""

from alembic import op
import sqlalchemy as sa

revision = "f6g7h8i9j0k1"
down_revision = "e5f6g7h8i9j0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add is_admin to users
    op.add_column("users", sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.text("false")))

    # 2. Add gold fields to annotation_tasks
    op.add_column("annotation_tasks", sa.Column("is_gold", sa.Boolean(), nullable=False, server_default=sa.text("false")))
    op.add_column("annotation_tasks", sa.Column("gold_answer", sa.String(1), nullable=True))
    op.create_index("ix_annotation_tasks_is_gold", "annotation_tasks", ["is_gold"])

    # 3. Add trust fields to annotations
    op.add_column("annotations", sa.Column("is_gold_response", sa.Boolean(), nullable=False, server_default=sa.text("false")))
    op.add_column("annotations", sa.Column("is_trusted", sa.Boolean(), nullable=False, server_default=sa.text("true")))
    op.create_index("ix_annotations_is_trusted", "annotations", ["is_trusted"])

    # 4. Create annotator_reliability table
    op.create_table(
        "annotator_reliability",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("gold_total", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("gold_correct", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("accuracy", sa.Float(), nullable=False, server_default=sa.text("1.0")),
        sa.Column("is_flagged", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("flagged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("unflagged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", name="uq_annotator_reliability_user"),
    )
    op.create_index("ix_annotator_reliability_user_id", "annotator_reliability", ["user_id"])
    op.create_index("ix_annotator_reliability_is_flagged", "annotator_reliability", ["is_flagged"])


def downgrade() -> None:
    op.drop_table("annotator_reliability")
    op.drop_index("ix_annotations_is_trusted", "annotations")
    op.drop_column("annotations", "is_trusted")
    op.drop_column("annotations", "is_gold_response")
    op.drop_index("ix_annotation_tasks_is_gold", "annotation_tasks")
    op.drop_column("annotation_tasks", "gold_answer")
    op.drop_column("annotation_tasks", "is_gold")
    op.drop_column("users", "is_admin")
