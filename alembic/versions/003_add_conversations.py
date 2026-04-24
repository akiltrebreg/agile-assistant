"""Add conversations and messages tables for short-term memory.

Revision ID: 003
Revises: 002
Create Date: 2026-04-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- conversations ---
    # NB: user_id — plain nullable UUID here; FK to user_profiles is added in 004
    op.create_table(
        "conversations",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("user_id", UUID(as_uuid=True), nullable=True),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "summary_turn_index",
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
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.CheckConstraint(
            "summary_turn_index >= 0",
            name="ck_conversations_summary_turn_index_nonneg",
        ),
    )

    op.create_index(
        "idx_conversations_user_active",
        "conversations",
        ["user_id", "is_active"],
        postgresql_where=sa.text("is_active = true"),
    )
    op.create_index(
        "idx_conversations_updated",
        "conversations",
        [sa.text("updated_at DESC")],
    )

    # Trigger to auto-update updated_at on row UPDATE.
    # Kept SQL-side so any client (psql, Alembic, app) stays consistent.
    op.execute("""
        CREATE OR REPLACE FUNCTION set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_conversations_updated_at
        BEFORE UPDATE ON conversations
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    """)

    # --- messages ---
    op.create_table(
        "messages",
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
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(10), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_truncated", sa.Text(), nullable=True),
        sa.Column(
            "metadata",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.CheckConstraint(
            "role IN ('user', 'assistant')",
            name="ck_messages_role",
        ),
        sa.CheckConstraint(
            "turn_index >= 0",
            name="ck_messages_turn_index_nonneg",
        ),
        sa.CheckConstraint(
            "char_length(content) <= 50000",
            name="ck_messages_content_length",
        ),
        sa.UniqueConstraint(
            "conversation_id",
            "turn_index",
            name="uq_messages_conversation_turn",
        ),
    )

    op.create_index(
        "idx_messages_conversation_turn",
        "messages",
        ["conversation_id", sa.text("turn_index DESC")],
    )


def downgrade() -> None:
    op.drop_index("idx_messages_conversation_turn", table_name="messages")
    op.drop_table("messages")

    op.execute("DROP TRIGGER IF EXISTS trg_conversations_updated_at ON conversations;")
    op.execute("DROP FUNCTION IF EXISTS set_updated_at();")

    op.drop_index("idx_conversations_updated", table_name="conversations")
    op.drop_index("idx_conversations_user_active", table_name="conversations")
    op.drop_table("conversations")
