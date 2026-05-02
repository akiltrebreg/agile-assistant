"""Add user_profiles and conversation_summaries for long-term memory.

Revision ID: 004
Revises: 003
Create Date: 2026-04-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- user_profiles ---
    op.create_table(
        "user_profiles",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("external_id", sa.String(255), nullable=False, unique=True),
        sa.Column("display_name", sa.String(255), nullable=True),
        sa.Column(
            "preferences",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("context_summary", sa.Text(), nullable=True),
        sa.Column(
            "total_conversations",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "total_messages",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.CheckConstraint(
            "total_conversations >= 0",
            name="ck_user_profiles_total_conversations_nonneg",
        ),
        sa.CheckConstraint(
            "total_messages >= 0",
            name="ck_user_profiles_total_messages_nonneg",
        ),
    )

    # Reuse set_updated_at() function created in 003
    op.execute("""
        CREATE TRIGGER trg_user_profiles_updated_at
        BEFORE UPDATE ON user_profiles
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    """)

    # --- conversation_summaries ---
    op.create_table(
        "conversation_summaries",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "conversation_id",
            UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("user_profiles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column(
            "topics",
            sa.ARRAY(sa.String()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.Column("turn_count", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.CheckConstraint(
            "turn_count IS NULL OR turn_count >= 0",
            name="ck_conversation_summaries_turn_count_nonneg",
        ),
    )

    op.create_index(
        "idx_conv_summaries_user",
        "conversation_summaries",
        ["user_id", sa.text("created_at DESC")],
    )

    # --- Finalise conversations.user_id FK (column exists from 003) ---
    op.create_foreign_key(
        "fk_conversations_user_id",
        "conversations",
        "user_profiles",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_conversations_user_id", "conversations", type_="foreignkey")

    op.drop_index("idx_conv_summaries_user", table_name="conversation_summaries")
    op.drop_table("conversation_summaries")

    op.execute("DROP TRIGGER IF EXISTS trg_user_profiles_updated_at ON user_profiles;")
    op.drop_table("user_profiles")
