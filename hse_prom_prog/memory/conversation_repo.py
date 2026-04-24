"""Repository for conversations and messages (short-term memory)."""

import json
import logging
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from hse_prom_prog.database.connection import DatabaseConnection
from hse_prom_prog.models.memory import Conversation, Message

logger = logging.getLogger(__name__)


class ConversationRepository:
    """CRUD for ``conversations`` and ``messages`` tables (raw SQL)."""

    def __init__(self, db: DatabaseConnection) -> None:
        self.db = db

    # ------------------------------------------------------------------ #
    # conversations                                                      #
    # ------------------------------------------------------------------ #

    def create(self, user_id: UUID | None = None) -> Conversation:
        """Create a new conversation and return it."""
        sql = """
            INSERT INTO conversations (user_id)
            VALUES (:user_id)
            RETURNING id, user_id, title, summary, summary_turn_index,
                      created_at, updated_at, is_active
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text(sql),
                    {"user_id": str(user_id) if user_id else None},
                )
                row = result.fetchone()
                conv = self._row_to_conversation(row)
                logger.info(f"[ConversationRepository] Created conversation {conv.id}")
                return conv
        except SQLAlchemyError as e:
            logger.error(f"[ConversationRepository] Failed to create conversation: {e}")
            raise

    def get(self, conversation_id: UUID) -> Conversation | None:
        """Fetch a conversation by id, or ``None`` if not found."""
        sql = """
            SELECT id, user_id, title, summary, summary_turn_index,
                   created_at, updated_at, is_active
            FROM conversations
            WHERE id = :id
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(text(sql), {"id": str(conversation_id)})
                row = result.fetchone()
                return self._row_to_conversation(row) if row else None
        except SQLAlchemyError as e:
            logger.error(
                f"[ConversationRepository] Failed to get conversation {conversation_id}: {e}"
            )
            raise

    def get_or_create(
        self,
        conversation_id: UUID | None,
        user_id: UUID | None,
    ) -> Conversation:
        """Return existing conversation or create a new one.

        If ``conversation_id`` is given but doesn't exist, a new conversation
        is created (the caller's id is discarded) — the UI is the source of
        truth for identity, so an invalid id is treated as "start fresh".
        """
        if conversation_id is not None:
            existing = self.get(conversation_id)
            if existing is not None:
                return existing
        return self.create(user_id=user_id)

    def list_by_user(self, user_id: UUID, limit: int = 20, offset: int = 0) -> list[Conversation]:
        """List conversations for a user, newest first."""
        sql = """
            SELECT id, user_id, title, summary, summary_turn_index,
                   created_at, updated_at, is_active
            FROM conversations
            WHERE user_id = :user_id
            ORDER BY updated_at DESC
            LIMIT :limit OFFSET :offset
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text(sql),
                    {"user_id": str(user_id), "limit": limit, "offset": offset},
                )
                return [self._row_to_conversation(row) for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(
                f"[ConversationRepository] Failed to list conversations for {user_id}: {e}"
            )
            raise

    def update_summary(
        self,
        conversation_id: UUID,
        summary: str,
        turn_index: int,
    ) -> None:
        """Update rolling summary and last-summarised turn index."""
        sql = """
            UPDATE conversations
            SET summary = :summary,
                summary_turn_index = :turn_index
            WHERE id = :id
        """
        try:
            with self.db.get_session() as session:
                session.execute(
                    text(sql),
                    {
                        "id": str(conversation_id),
                        "summary": summary,
                        "turn_index": turn_index,
                    },
                )
        except SQLAlchemyError as e:
            logger.error(
                f"[ConversationRepository] Failed to update summary for {conversation_id}: {e}"
            )
            raise

    def update_title(self, conversation_id: UUID, title: str) -> None:
        """Set the display title of a conversation (idempotent)."""
        sql = "UPDATE conversations SET title = :title WHERE id = :id"
        try:
            with self.db.get_session() as session:
                session.execute(
                    text(sql),
                    {"id": str(conversation_id), "title": title},
                )
        except SQLAlchemyError as e:
            logger.error(
                f"[ConversationRepository] Failed to update title for {conversation_id}: {e}"
            )
            raise

    def close(self, conversation_id: UUID) -> None:
        """Mark a conversation as closed (``is_active=false``)."""
        sql = "UPDATE conversations SET is_active = false WHERE id = :id"
        try:
            with self.db.get_session() as session:
                session.execute(text(sql), {"id": str(conversation_id)})
        except SQLAlchemyError as e:
            logger.error(f"[ConversationRepository] Failed to close {conversation_id}: {e}")
            raise

    def touch(self, conversation_id: UUID) -> None:
        """Bump ``updated_at`` explicitly (trigger also fires on any UPDATE)."""
        sql = "UPDATE conversations SET updated_at = NOW() WHERE id = :id"
        try:
            with self.db.get_session() as session:
                session.execute(text(sql), {"id": str(conversation_id)})
        except SQLAlchemyError as e:
            logger.error(f"[ConversationRepository] Failed to touch {conversation_id}: {e}")
            raise

    # ------------------------------------------------------------------ #
    # messages                                                           #
    # ------------------------------------------------------------------ #

    def save_message(  # noqa: PLR0913
        self,
        conversation_id: UUID,
        turn_index: int,
        role: str,
        content: str,
        content_truncated: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Insert a message and return it."""
        sql = """
            INSERT INTO messages
                (conversation_id, turn_index, role, content, content_truncated, metadata)
            VALUES
                (:conversation_id, :turn_index, :role, :content, :content_truncated,
                 CAST(:metadata AS JSONB))
            RETURNING id, conversation_id, turn_index, role, content,
                      content_truncated, metadata, created_at
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text(sql),
                    {
                        "conversation_id": str(conversation_id),
                        "turn_index": turn_index,
                        "role": role,
                        "content": content,
                        "content_truncated": content_truncated,
                        "metadata": json.dumps(metadata or {}, default=str),
                    },
                )
                row = result.fetchone()
                return self._row_to_message(row)
        except SQLAlchemyError as e:
            logger.error(
                f"[ConversationRepository] Failed to save message for "
                f"{conversation_id} turn={turn_index}: {e}"
            )
            raise

    def get_messages(self, conversation_id: UUID, limit: int | None = None) -> list[Message]:
        """Return messages for a conversation in ascending turn order."""
        sql = """
            SELECT id, conversation_id, turn_index, role, content,
                   content_truncated, metadata, created_at
            FROM messages
            WHERE conversation_id = :conversation_id
            ORDER BY turn_index ASC
        """
        if limit is not None:
            sql += " LIMIT :limit"
        params: dict[str, Any] = {"conversation_id": str(conversation_id)}
        if limit is not None:
            params["limit"] = limit
        try:
            with self.db.get_session() as session:
                result = session.execute(text(sql), params)
                return [self._row_to_message(row) for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(
                f"[ConversationRepository] Failed to get messages for {conversation_id}: {e}"
            )
            raise

    def get_latest_turn_index(self, conversation_id: UUID) -> int:
        """Return the highest ``turn_index`` in a conversation, or -1 if empty.

        Callers should add 1 to decide the next ``turn_index`` to write. The
        UNIQUE(conversation_id, turn_index) constraint on the messages table
        protects against races if two workers compute this concurrently —
        the loser's INSERT will raise ``IntegrityError`` and should retry.
        """
        sql = """
            SELECT COALESCE(MAX(turn_index), -1) AS max_idx
            FROM messages
            WHERE conversation_id = :conversation_id
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(text(sql), {"conversation_id": str(conversation_id)})
                row = result.fetchone()
                return int(row.max_idx)
        except SQLAlchemyError as e:
            logger.error(
                f"[ConversationRepository] Failed to get latest turn_index for "
                f"{conversation_id}: {e}"
            )
            raise

    def count_messages(self, conversation_id: UUID) -> int:
        """Return total number of messages in a conversation."""
        sql = "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = :conversation_id"
        try:
            with self.db.get_session() as session:
                result = session.execute(text(sql), {"conversation_id": str(conversation_id)})
                return int(result.fetchone().n)
        except SQLAlchemyError as e:
            logger.error(
                f"[ConversationRepository] Failed to count messages for {conversation_id}: {e}"
            )
            raise

    # ------------------------------------------------------------------ #
    # row → model                                                        #
    # ------------------------------------------------------------------ #

    def _row_to_conversation(self, row: Any) -> Conversation:
        return Conversation(
            id=row.id,
            user_id=row.user_id,
            title=row.title,
            summary=row.summary,
            summary_turn_index=row.summary_turn_index,
            created_at=row.created_at,
            updated_at=row.updated_at,
            is_active=row.is_active,
        )

    def _row_to_message(self, row: Any) -> Message:
        return Message(
            id=row.id,
            conversation_id=row.conversation_id,
            turn_index=row.turn_index,
            role=row.role,
            content=row.content,
            content_truncated=row.content_truncated,
            metadata=row.metadata or {},
            created_at=row.created_at,
        )
