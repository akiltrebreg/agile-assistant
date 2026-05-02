"""Repository for user_profiles (long-term memory)."""

import json
import logging
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from hse_prom_prog.database.connection import DatabaseConnection
from hse_prom_prog.models.memory import UserProfile

logger = logging.getLogger(__name__)


class ProfileRepository:
    """CRUD for ``user_profiles`` table (raw SQL)."""

    def __init__(self, db: DatabaseConnection) -> None:
        """Initialize ProfileRepository.

        Args:
            db: Database connection used to acquire sessions.
        """
        self.db = db

    def get_or_create(self, external_id: str) -> UserProfile:
        """Return existing profile for ``external_id`` or create a new one.

        Uses ``ON CONFLICT`` so two concurrent workers can't create a
        duplicate — one wins the insert, the other receives the existing row.

        Args:
            external_id: External (cookie / SSO) identifier.

        Returns:
            The resolved ``UserProfile``.

        Raises:
            SQLAlchemyError: When the upsert fails.
        """
        sql = """
            INSERT INTO user_profiles (external_id)
            VALUES (:external_id)
            ON CONFLICT (external_id) DO UPDATE
                SET external_id = EXCLUDED.external_id
            RETURNING id, external_id, display_name, preferences, context_summary,
                      total_conversations, total_messages, created_at, updated_at
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(text(sql), {"external_id": external_id})
                row = result.fetchone()
                profile = self._row_to_profile(row)
                logger.info(
                    f"[ProfileRepository] get_or_create profile "
                    f"{profile.id} for external_id={external_id}"
                )
                return profile
        except SQLAlchemyError as e:
            logger.error(f"[ProfileRepository] Failed to get_or_create: {e}")
            raise

    def get(self, user_id: UUID) -> UserProfile | None:
        """Fetch a profile by id, or ``None`` if not found.

        Args:
            user_id: Internal user identifier.

        Returns:
            ``UserProfile`` if found, otherwise ``None``.

        Raises:
            SQLAlchemyError: When the SELECT fails.
        """
        sql = """
            SELECT id, external_id, display_name, preferences, context_summary,
                   total_conversations, total_messages, created_at, updated_at
            FROM user_profiles
            WHERE id = :id
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(text(sql), {"id": str(user_id)})
                row = result.fetchone()
                return self._row_to_profile(row) if row else None
        except SQLAlchemyError as e:
            logger.error(f"[ProfileRepository] Failed to get {user_id}: {e}")
            raise

    def update_preferences(self, user_id: UUID, preferences: dict[str, Any]) -> None:
        """Replace the ``preferences`` JSONB blob.

        Args:
            user_id: Internal user identifier.
            preferences: Fully-formed preferences dict to persist.

        Raises:
            SQLAlchemyError: When the UPDATE fails.
        """
        sql = """
            UPDATE user_profiles
            SET preferences = CAST(:preferences AS JSONB)
            WHERE id = :id
        """
        try:
            with self.db.get_session() as session:
                session.execute(
                    text(sql),
                    {
                        "id": str(user_id),
                        "preferences": json.dumps(preferences, default=str),
                    },
                )
        except SQLAlchemyError as e:
            logger.error(f"[ProfileRepository] Failed to update preferences for {user_id}: {e}")
            raise

    def update_context_summary(self, user_id: UUID, summary: str) -> None:
        """Replace the rolling ``context_summary`` text.

        Args:
            user_id: Internal user identifier.
            summary: New rolling summary text.

        Raises:
            SQLAlchemyError: When the UPDATE fails.
        """
        sql = "UPDATE user_profiles SET context_summary = :summary WHERE id = :id"
        try:
            with self.db.get_session() as session:
                session.execute(text(sql), {"id": str(user_id), "summary": summary})
        except SQLAlchemyError as e:
            logger.error(f"[ProfileRepository] Failed to update context_summary for {user_id}: {e}")
            raise

    def increment_counters(
        self,
        user_id: UUID,
        conversations_delta: int = 0,
        messages_delta: int = 0,
    ) -> None:
        """Atomically increment usage counters.

        Args:
            user_id: Internal user identifier.
            conversations_delta: Delta applied to ``total_conversations``.
            messages_delta: Delta applied to ``total_messages``.

        Raises:
            SQLAlchemyError: When the UPDATE fails.
        """
        sql = """
            UPDATE user_profiles
            SET total_conversations = total_conversations + :conv_delta,
                total_messages = total_messages + :msg_delta
            WHERE id = :id
        """
        try:
            with self.db.get_session() as session:
                session.execute(
                    text(sql),
                    {
                        "id": str(user_id),
                        "conv_delta": conversations_delta,
                        "msg_delta": messages_delta,
                    },
                )
        except SQLAlchemyError as e:
            logger.error(f"[ProfileRepository] Failed to increment counters for {user_id}: {e}")
            raise

    def _row_to_profile(self, row: Any) -> UserProfile:
        """Project a SQLAlchemy row into a ``UserProfile`` domain model."""
        return UserProfile(
            id=row.id,
            external_id=row.external_id,
            display_name=row.display_name,
            preferences=row.preferences or {},
            context_summary=row.context_summary,
            total_conversations=row.total_conversations,
            total_messages=row.total_messages,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
