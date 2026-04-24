"""Facade over the memory repositories and builders.

``MemoryManager`` is what the workflow / Celery task will talk to — they
shouldn't know about individual repositories or the context assembly
strategy.
"""

import logging
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from hse_prom_prog.database.connection import DatabaseConnection
from hse_prom_prog.memory.context_builder import ContextBuilder
from hse_prom_prog.memory.conversation_repo import ConversationRepository
from hse_prom_prog.memory.profile_extractor import ProfileExtractor
from hse_prom_prog.memory.profile_repo import ProfileRepository
from hse_prom_prog.memory.summary_repo import SummaryRepository
from hse_prom_prog.memory.truncator import truncate_message
from hse_prom_prog.models.memory import Conversation, ConversationContext

logger = logging.getLogger(__name__)

BOT_TRUNCATE_TOKENS = 150  # ≈ 450 chars — fits one paragraph in replay
TITLE_MAX_CHARS = 60
SAVE_TURN_MAX_RETRIES = 3


class MemoryManager:
    """High-level API for conversation history and user profiles."""

    def __init__(  # noqa: PLR0913
        self,
        db: DatabaseConnection,
        *,
        conversation_repo: ConversationRepository | None = None,
        profile_repo: ProfileRepository | None = None,
        summary_repo: SummaryRepository | None = None,
        context_builder: ContextBuilder | None = None,
        profile_extractor: ProfileExtractor | None = None,
    ) -> None:
        self.db = db
        self.conversation_repo = conversation_repo or ConversationRepository(db)
        self.profile_repo = profile_repo or ProfileRepository(db)
        self.summary_repo = summary_repo or SummaryRepository(db)
        self.context_builder = context_builder or ContextBuilder(self.conversation_repo)
        self.profile_extractor = profile_extractor or ProfileExtractor()

    # ------------------------------------------------------------------ #
    # conversations                                                      #
    # ------------------------------------------------------------------ #

    def get_or_create_conversation(
        self,
        conversation_id: UUID | None,
        user_id: UUID | None,
    ) -> Conversation:
        """Return an existing conversation or create a fresh one."""
        return self.conversation_repo.get_or_create(conversation_id, user_id)

    def get_context(self, conversation_id: UUID, token_budget: int) -> ConversationContext:
        """Assemble the short-term memory context for an agent prompt."""
        return self.context_builder.build(conversation_id, token_budget)

    def save_turn(
        self,
        conversation_id: UUID,
        user_message: str,
        bot_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[int, int]:
        """Persist a user/assistant turn pair.

        Returns the ``(user_turn_index, assistant_turn_index)`` actually
        written. Handles the UNIQUE(conversation_id, turn_index) race by
        re-reading the max turn_index and retrying a few times — cheaper
        than SELECT … FOR UPDATE on every save for our traffic profile.
        """
        metadata = metadata or {}

        for attempt in range(SAVE_TURN_MAX_RETRIES):
            latest = self.conversation_repo.get_latest_turn_index(conversation_id)
            user_idx = latest + 1
            bot_idx = latest + 2

            try:
                self.conversation_repo.save_message(
                    conversation_id=conversation_id,
                    turn_index=user_idx,
                    role="user",
                    content=user_message,
                    content_truncated=None,
                    metadata=metadata,
                )
                self.conversation_repo.save_message(
                    conversation_id=conversation_id,
                    turn_index=bot_idx,
                    role="assistant",
                    content=bot_message,
                    content_truncated=truncate_message(bot_message, BOT_TRUNCATE_TOKENS),
                    metadata=metadata,
                )
                break
            except IntegrityError:
                if attempt == SAVE_TURN_MAX_RETRIES - 1:
                    logger.error(
                        f"[MemoryManager] save_turn failed after "
                        f"{SAVE_TURN_MAX_RETRIES} retries for {conversation_id}"
                    )
                    raise
                logger.warning(
                    f"[MemoryManager] turn_index race for {conversation_id}, "
                    f"retry {attempt + 1}/{SAVE_TURN_MAX_RETRIES}"
                )

        # First turn in an unnamed conversation → derive title from the query.
        if latest == -1:
            conv = self.conversation_repo.get(conversation_id)
            if conv is not None and not conv.title:
                title = user_message.strip()[:TITLE_MAX_CHARS]
                if title:
                    self.conversation_repo.update_title(conversation_id, title)
        else:
            # Any UPDATE on conversations bumps updated_at via trigger.
            self.conversation_repo.touch(conversation_id)

        return user_idx, bot_idx

    # ------------------------------------------------------------------ #
    # profiles                                                           #
    # ------------------------------------------------------------------ #

    def get_profile(self, user_id: UUID) -> dict[str, Any] | None:
        """Return the user profile as a plain dict, or ``None``."""
        profile = self.profile_repo.get(user_id)
        return profile.to_dict() if profile else None

    def get_or_create_profile_by_external_id(self, external_id: str) -> dict[str, Any]:
        """Resolve an external id (e.g. cookie UUID) to a profile dict."""
        return self.profile_repo.get_or_create(external_id).to_dict()

    def update_profile(self, user_id: UUID, conversation_id: UUID) -> dict[str, Any]:
        """Recompute and persist the profile's preferences.

        Aggregates metadata across *all* messages in the given conversation.
        The wider scope (all conversations of a user) is a separate job —
        this method is the per-turn lightweight update.
        """
        messages = self.conversation_repo.get_messages(conversation_id)
        metadata_list = [m.metadata for m in messages if m.metadata]
        preferences = self.profile_extractor.extract(metadata_list)
        self.profile_repo.update_preferences(user_id, preferences)
        return preferences
