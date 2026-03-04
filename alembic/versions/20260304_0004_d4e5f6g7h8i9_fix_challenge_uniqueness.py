"""Fix challenge_id uniqueness for multi-validator.

Revision ID: d4e5f6g7h8i9
Revises: c3d4e5f6g7h8
Create Date: 2026-03-04

The challenge_id was UNIQUE globally, but multiple validators generate
independent challenges. Change to UNIQUE(challenge_id, validator_hotkey)
so the same validator can't submit the same challenge twice, but different
validators can submit independently.
"""

from alembic import op

revision = "d4e5f6g7h8i9"
down_revision = "c3d4e5f6g7h8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop the global unique constraint on challenge_id
    op.drop_constraint("validation_rounds_challenge_id_key", "validation_rounds", type_="unique")

    # Add composite unique constraint: same validator + same challenge = reject
    op.create_unique_constraint(
        "uq_validation_rounds_challenge_validator",
        "validation_rounds",
        ["challenge_id", "validator_hotkey"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_validation_rounds_challenge_validator", "validation_rounds", type_="unique")
    op.create_unique_constraint("validation_rounds_challenge_id_key", "validation_rounds", ["challenge_id"])
