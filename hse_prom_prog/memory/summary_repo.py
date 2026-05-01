"""Repository for conversation_summaries (long-term memory)."""

import logging
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from hse_prom_prog.database.connection import DatabaseConnection
from hse_prom_prog.models.memory import ConversationSummary

logger = logging.getLogger(__name__)


class SummaryRepository:
    """CRUD for ``conversation_summaries`` table (raw SQL)."""

    def __init__(self, db: DatabaseConnection) -> None:
        """Initialize SummaryRepository.

        Args:
            db: Database connection used to acquire sessions.
        """
        self.db = db

    def create(
        self,
        conversation_id: UUID,
        user_id: UUID,
        summary: str,
        topics: list[str],
        turn_count: int | None = None,
    ) -> ConversationSummary:
        """Persist a conversation summary and return it.

        Args:
            conversation_id: Conversation that produced the summary.
            user_id: Internal user owning the conversation.
            summary: Summary text.
            topics: Extracted topic names.
            turn_count: Total number of turns covered by the summary.

        Returns:
            The persisted ``ConversationSummary`` row.

        Raises:
            SQLAlchemyError: When the INSERT fails.
        """
        sql = """
            INSERT INTO conversation_summaries
                (conversation_id, user_id, summary, topics, turn_count)
            VALUES
                (:conversation_id, :user_id, :summary, :topics, :turn_count)
            RETURNING id, conversation_id, user_id, summary, topics,
                      turn_count, created_at
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text(sql),
                    {
                        "conversation_id": str(conversation_id),
                        "user_id": str(user_id),
                        "summary": summary,
                        "topics": topics,
                        "turn_count": turn_count,
                    },
                )
                row = result.fetchone()
                return self._row_to_summary(row)
        except SQLAlchemyError as e:
            logger.error(
                f"[SummaryRepository] Failed to create summary for conv={conversation_id}: {e}"
            )
            raise

    def get_recent(self, user_id: UUID, limit: int = 10) -> list[ConversationSummary]:
        """Return the user's most recent summaries, newest first.

        Args:
            user_id: Internal user identifier.
            limit: Maximum number of summaries to return. Defaults to 10.

        Returns:
            Summaries ordered by ``created_at`` descending.

        Raises:
            SQLAlchemyError: When the SELECT fails.
        """
        sql = """
            SELECT id, conversation_id, user_id, summary, topics,
                   turn_count, created_at
            FROM conversation_summaries
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            LIMIT :limit
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(text(sql), {"user_id": str(user_id), "limit": limit})
                return [self._row_to_summary(row) for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"[SummaryRepository] Failed to get recent summaries for {user_id}: {e}")
            raise

    def _row_to_summary(self, row: Any) -> ConversationSummary:
        """Project a SQLAlchemy row into a ``ConversationSummary`` model."""
        return ConversationSummary(
            id=row.id,
            conversation_id=row.conversation_id,
            user_id=row.user_id,
            summary=row.summary,
            topics=list(row.topics) if row.topics else [],
            turn_count=row.turn_count,
            created_at=row.created_at,
        )
