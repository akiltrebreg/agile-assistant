"""Assemble a ``ConversationContext`` for agent prompts from raw messages.

Strategy: sliding window (last K full turns) + lazy summary for everything
that falls out of the window. K is adaptive — derived from the token
budget of the calling agent, not a fixed constant.
"""

import logging
from uuid import UUID

from hse_prom_prog.memory.conversation_repo import ConversationRepository
from hse_prom_prog.memory.token_estimator import estimate_tokens
from hse_prom_prog.metrics import MEMORY_CONTEXT_TOKENS, MEMORY_CONTEXT_TURNS
from hse_prom_prog.models.memory import ConversationContext, Message

logger = logging.getLogger(__name__)

SUMMARY_RESERVE_TOKENS = 150


class ContextBuilder:
    """Builds a ``ConversationContext`` from persisted messages."""

    def __init__(self, repo: ConversationRepository) -> None:
        self.repo = repo

    def build(self, conversation_id: UUID, token_budget: int) -> ConversationContext:
        """Return the context that fits within ``token_budget`` tokens.

        Algorithm:
          1. Load all messages + conversation (for its rolling summary).
          2. Walk from newest → oldest, accumulating ``Message`` objects
             while token total stays under budget. Always keep ≥ 1 turn.
          3. If messages older than the kept window are not covered by the
             existing summary, flag ``needs_summarization=True`` — the
             summariser runs as a separate async job.
        """
        conversation = self.repo.get(conversation_id)
        if conversation is None:
            return ConversationContext(
                summary="",
                recent_turns=[],
                history_token_count=0,
                needs_summarization=False,
            )

        messages = self.repo.get_messages(conversation_id)
        if not messages:
            return ConversationContext(
                summary=conversation.summary or "",
                recent_turns=[],
                history_token_count=0,
                needs_summarization=False,
            )

        existing_summary = conversation.summary or ""
        summary_reserve = SUMMARY_RESERVE_TOKENS if existing_summary else 0

        # Effective per-turn budget; always allow at least one turn even at 0.
        available = max(0, token_budget - summary_reserve)

        kept: list[Message] = []
        running_tokens = 0
        for msg in reversed(messages):
            cost = self._message_token_cost(msg)
            if kept and running_tokens + cost > available:
                break
            kept.append(msg)
            running_tokens += cost

        kept.reverse()
        kept_start_idx = kept[0].turn_index

        # Something older than the kept window exists: does the stored summary
        # already cover up to (and including) the turn right before the window?
        needs_summarization = (
            kept_start_idx > 0 and conversation.summary_turn_index < kept_start_idx
        )

        history_tokens = running_tokens + (
            estimate_tokens(existing_summary) if existing_summary else 0
        )

        MEMORY_CONTEXT_TOKENS.observe(history_tokens)
        MEMORY_CONTEXT_TURNS.observe(len(kept))

        return ConversationContext(
            summary=existing_summary,
            recent_turns=[self._message_to_turn(m) for m in kept],
            history_token_count=history_tokens,
            needs_summarization=needs_summarization,
        )

    @staticmethod
    def _message_token_cost(msg: Message) -> int:
        """Token cost of a message when injected into a prompt.

        Prefer ``content_truncated`` (always ≤ 150 tok) over the full body so
        long assistant replies don't blow the budget during replay.
        """
        body = msg.content_truncated or msg.content
        # +2 tokens for the role label and a separator, stable across turns.
        return estimate_tokens(body) + 2

    @staticmethod
    def _message_to_turn(msg: Message) -> dict:
        body = msg.content_truncated or msg.content
        return {
            "role": msg.role,
            "content": body,
            "turn_index": msg.turn_index,
            "metadata": msg.metadata or {},
        }
