"""Add annotation tables for crowd preference labeling.

Revision ID: e5f6g7h8i9j0
Revises: d4e5f6g7h8i9
Create Date: 2026-03-04

Creates annotation_tasks, annotations, and preference_models tables
for the crowd-annotation system.
"""

import sqlalchemy as sa
from alembic import op

revision = "e5f6g7h8i9j0"
down_revision = "d4e5f6g7h8i9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Annotation tasks: A/B comparison pairs
    op.create_table(
        "annotation_tasks",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "round_id",
            sa.String(32),
            sa.ForeignKey("validation_rounds.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "audio_a_id",
            sa.String(32),
            sa.ForeignKey("validation_audio.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "audio_b_id",
            sa.String(32),
            sa.ForeignKey("validation_audio.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("quorum", sa.Integer, nullable=False, server_default=sa.text("5")),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        sa.Column("winner", sa.String(1), nullable=True),
        sa.Column("vote_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("audio_a_id", "audio_b_id", name="uq_annotation_tasks_pair"),
    )
    op.create_index("ix_annotation_tasks_round_id", "annotation_tasks", ["round_id"])
    op.create_index("ix_annotation_tasks_status", "annotation_tasks", ["status"])

    # Annotations: individual votes
    op.create_table(
        "annotations",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "task_id",
            sa.String(32),
            sa.ForeignKey("annotation_tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.String(32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("choice", sa.String(1), nullable=False),
        sa.Column("duration_ms", sa.Integer, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("task_id", "user_id", name="uq_annotations_user_task"),
    )
    op.create_index("ix_annotations_task_id", "annotations", ["task_id"])
    op.create_index("ix_annotations_user_id", "annotations", ["user_id"])

    # Preference models: trained checkpoint metadata + binary data
    op.create_table(
        "preference_models",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("val_accuracy", sa.Float, nullable=True),
        sa.Column("n_train_pairs", sa.Integer, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("model_data", sa.LargeBinary, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_preference_models_version", "preference_models", ["version"])


def downgrade() -> None:
    op.drop_table("annotations")
    op.drop_table("annotation_tasks")
    op.drop_table("preference_models")
