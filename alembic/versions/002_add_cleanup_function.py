"""Add cleanup function for old tasks and workflow_state.

Revision ID: 002
Revises: 001
Create Date: 2026-02-13
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Function to clear workflow_state from completed tasks older than N days
    # and delete terminal tasks older than M days.
    # Run periodically via pg_cron or external scheduler.
    op.execute("""
        CREATE OR REPLACE FUNCTION cleanup_old_tasks(
            clear_state_after_days INT DEFAULT 7,
            delete_after_days INT DEFAULT 30
        ) RETURNS TABLE(states_cleared BIGINT, tasks_deleted BIGINT)
        LANGUAGE plpgsql AS $$
        DECLARE
            v_cleared BIGINT;
            v_deleted BIGINT;
        BEGIN
            -- 1) Null out workflow_state for completed tasks older than N days
            UPDATE tasks
            SET workflow_state = NULL
            WHERE status IN ('COMPLETED', 'FAILED')
              AND completed_at < NOW() - make_interval(days => clear_state_after_days)
              AND workflow_state IS NOT NULL;
            GET DIAGNOSTICS v_cleared = ROW_COUNT;

            -- 2) Delete terminal tasks older than M days
            DELETE FROM tasks
            WHERE status IN ('COMPLETED', 'FAILED')
              AND completed_at < NOW() - make_interval(days => delete_after_days);
            GET DIAGNOSTICS v_deleted = ROW_COUNT;

            RETURN QUERY SELECT v_cleared, v_deleted;
        END;
        $$;
    """)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS cleanup_old_tasks;")
