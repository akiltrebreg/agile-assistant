"""Memory layer domain models.

Plain domain classes (no SQLAlchemy ORM) for the conversation history and
user profile tables introduced by Alembic migrations 003/004. Consistent
with the repository-pattern style used by ``models/task.py``.
"""

from datetime import datetime
from typing import Any, TypedDict
from uuid import UUID


class Conversation:
    """Domain model for a conversation between user and assistant."""

    def __init__(  # noqa: PLR0913
        self,
        id: UUID,  # noqa: A002
        user_id: UUID | None,
        title: str | None,
        summary: str | None,
        summary_turn_index: int,
        created_at: datetime,
        updated_at: datetime,
        is_active: bool,
    ) -> None:
        """Initialize Conversation.

        Args:
            id: Conversation identifier.
            user_id: Owner user id, or ``None`` for anonymous sessions.
            title: Display title, or ``None`` until derived from a turn.
            summary: Rolling summary text covering older turns.
            summary_turn_index: Highest ``turn_index`` covered by ``summary``.
            created_at: Creation timestamp.
            updated_at: Last activity timestamp.
            is_active: Whether the conversation is still open.
        """
        self.id = id
        self.user_id = user_id
        self.title = title
        self.summary = summary
        self.summary_turn_index = summary_turn_index
        self.created_at = created_at
        self.updated_at = updated_at
        self.is_active = is_active

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the conversation."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "title": self.title,
            "summary": self.summary,
            "summary_turn_index": self.summary_turn_index,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_active": self.is_active,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly summary string."""
        return f"Conversation(id={self.id}, title={self.title!r}, active={self.is_active})"


class Message:
    """Domain model for a single message within a conversation."""

    def __init__(  # noqa: PLR0913
        self,
        id: UUID,  # noqa: A002
        conversation_id: UUID,
        turn_index: int,
        role: str,
        content: str,
        content_truncated: str | None,
        metadata: dict[str, Any],
        created_at: datetime,
    ) -> None:
        """Initialize Message.

        Args:
            id: Message identifier.
            conversation_id: Owning conversation id.
            turn_index: Position of the message within the conversation.
            role: Author role (``user`` or ``assistant``).
            content: Full message body.
            content_truncated: Optional pre-truncated body for replay budgets.
            metadata: JSONB metadata bag.
            created_at: Creation timestamp.
        """
        self.id = id
        self.conversation_id = conversation_id
        self.turn_index = turn_index
        self.role = role
        self.content = content
        self.content_truncated = content_truncated
        self.metadata = metadata
        self.created_at = created_at

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the message."""
        return {
            "id": str(self.id),
            "conversation_id": str(self.conversation_id),
            "turn_index": self.turn_index,
            "role": self.role,
            "content": self.content,
            "content_truncated": self.content_truncated,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly summary string."""
        return f"Message(turn={self.turn_index}, role={self.role}, len={len(self.content)})"


class UserProfile:
    """Domain model for a user profile (long-term memory)."""

    def __init__(  # noqa: PLR0913
        self,
        id: UUID,  # noqa: A002
        external_id: str,
        display_name: str | None,
        preferences: dict[str, Any],
        context_summary: str | None,
        total_conversations: int,
        total_messages: int,
        created_at: datetime,
        updated_at: datetime,
    ) -> None:
        """Initialize UserProfile.

        Args:
            id: Internal user identifier.
            external_id: External (cookie / SSO) identifier.
            display_name: Optional display name.
            preferences: Derived preferences (teams, metrics, detail level).
            context_summary: Rolling summary across recent conversations.
            total_conversations: Lifetime conversation counter.
            total_messages: Lifetime message counter.
            created_at: Creation timestamp.
            updated_at: Last update timestamp.
        """
        self.id = id
        self.external_id = external_id
        self.display_name = display_name
        self.preferences = preferences
        self.context_summary = context_summary
        self.total_conversations = total_conversations
        self.total_messages = total_messages
        self.created_at = created_at
        self.updated_at = updated_at

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the profile."""
        return {
            "id": str(self.id),
            "external_id": self.external_id,
            "display_name": self.display_name,
            "preferences": self.preferences,
            "context_summary": self.context_summary,
            "total_conversations": self.total_conversations,
            "total_messages": self.total_messages,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly summary string."""
        return (
            f"UserProfile(id={self.id}, external_id={self.external_id!r}, "
            f"conversations={self.total_conversations})"
        )


class ConversationSummary:
    """Domain model for a finalised conversation summary (long-term memory)."""

    def __init__(  # noqa: PLR0913
        self,
        id: UUID,  # noqa: A002
        conversation_id: UUID,
        user_id: UUID,
        summary: str,
        topics: list[str],
        turn_count: int | None,
        created_at: datetime,
    ) -> None:
        """Initialize ConversationSummary.

        Args:
            id: Summary identifier.
            conversation_id: Source conversation id.
            user_id: Owner user id.
            summary: Summary text.
            topics: Extracted topic names.
            turn_count: Total turns covered by the summary.
            created_at: Creation timestamp.
        """
        self.id = id
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.summary = summary
        self.topics = topics
        self.turn_count = turn_count
        self.created_at = created_at

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the summary."""
        return {
            "id": str(self.id),
            "conversation_id": str(self.conversation_id),
            "user_id": str(self.user_id),
            "summary": self.summary,
            "topics": self.topics,
            "turn_count": self.turn_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly summary string."""
        return f"ConversationSummary(conv={self.conversation_id}, topics={self.topics})"


class ConversationContext(TypedDict):
    """Short-term memory payload injected into agent prompts.

    Structure:
      * ``summary``         — rolling summary of turns that fell out of the window
      * ``recent_turns``    — list of dicts ``{role, content, metadata}``
      * ``history_token_count`` — estimated token cost of the above
      * ``needs_summarization`` — True if new turns outside the window aren't in summary
    """

    summary: str
    recent_turns: list[dict[str, Any]]
    history_token_count: int
    needs_summarization: bool
