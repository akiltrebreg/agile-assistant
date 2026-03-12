"""Add tasks table for async workflow tracking.

Revision ID: 001
Revises:
Create Date: 2026-02-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tasks",
        sa.Column(
            "task_id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.String(32),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column("result", JSONB, nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("celery_task_id", sa.String(255), unique=True, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("workflow_state", JSONB, nullable=True),
        sa.CheckConstraint(
            "status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED')",
            name="ck_tasks_status",
        ),
    )

    # Indexes for performance
    op.create_index("idx_tasks_status", "tasks", ["status"])
    op.create_index("idx_tasks_created_at", "tasks", [sa.text("created_at DESC")])
    op.create_index("idx_tasks_celery_task_id", "tasks", ["celery_task_id"])
    op.create_index(
        "idx_tasks_status_created",
        "tasks",
        ["status", sa.text("created_at DESC")],
    )
    # Index for TTL cleanup of old completed tasks
    op.create_index(
        "idx_tasks_completed_at",
        "tasks",
        [sa.text("completed_at DESC")],
        postgresql_where=sa.text("completed_at IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_tasks_completed_at", table_name="tasks")
    op.drop_index("idx_tasks_status_created", table_name="tasks")
    op.drop_index("idx_tasks_celery_task_id", table_name="tasks")
    op.drop_index("idx_tasks_created_at", table_name="tasks")
    op.drop_index("idx_tasks_status", table_name="tasks")
    op.drop_table("tasks")
