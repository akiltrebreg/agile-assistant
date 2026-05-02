"""Add conversation_id FK to tasks for memory-layer wiring.

Revision ID: 005
Revises: 004
Create Date: 2026-04-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Nullable so tasks created before memory rollout keep loading.
    # ON DELETE SET NULL — a closed/purged conversation should not
    # orphan the task record (it's still useful for audit/debug).
    op.add_column(
        "tasks",
        sa.Column("conversation_id", UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_tasks_conversation_id",
        "tasks",
        "conversations",
        ["conversation_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "idx_tasks_conversation_id",
        "tasks",
        ["conversation_id"],
        postgresql_where=sa.text("conversation_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_tasks_conversation_id", table_name="tasks")
    op.drop_constraint("fk_tasks_conversation_id", "tasks", type_="foreignkey")
    op.drop_column("tasks", "conversation_id")
