"""Celery tasks for memory-layer maintenance.

* ``summarize_session`` — on conversation close: LLM summary + rolling
  profile summary + preferences refresh.
* ``update_profile_async`` — per-turn lightweight profile refresh, run
  off the critical path so the user's response is never delayed.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from hse_prom_prog.database.connection import get_database
from hse_prom_prog.llm.client import get_llm_client
from hse_prom_prog.memory.manager import MemoryManager
from hse_prom_prog.metrics import MEMORY_PROFILE_UPDATES, MEMORY_SUMMARIZATIONS
from hse_prom_prog.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT_SYSTEM = (
    "Ты суммаризатор диалогов. Напиши 2-3 предложения: о чём спрашивал "
    "пользователь, какие команды / метрики / задачи упоминались, какой "
    "был итог. Отвечай только текстом, без markdown и без преамбул."
)
_ROLLING_SUMMARY_PROMPT_SYSTEM = (
    "Ты составляешь краткий профиль пользователя (3-4 предложения) на "
    "основе истории его диалогов: какие команды / метрики он обычно "
    "отслеживает, предполагаемая роль, типичные задачи."
)

_MAX_DIALOG_CHARS = 2000
_SUMMARY_MAX_TOKENS = 150
_ROLLING_SUMMARY_MAX_TOKENS = 200
_MIN_TURNS_FOR_SUMMARY = 2
_MIN_SUMMARIES_FOR_META = 3
_RECENT_SUMMARIES_WINDOW = 10


@celery_app.task(name="hse_prom_prog.tasks.summarize_session")
def summarize_session(conversation_id: str, user_id: str) -> dict[str, Any] | None:
    """Persist a finalised summary and refresh the user profile.

    Runs when a conversation is closed. On failure at any step we log and
    return ``None`` — memory maintenance is best-effort and must never
    break the user flow.
    """
    conv_uuid = UUID(conversation_id)
    user_uuid = UUID(user_id)
    MEMORY_SUMMARIZATIONS.inc()

    db = None
    try:
        db = get_database()
        memory = MemoryManager(db)
        messages = memory.conversation_repo.get_messages(conv_uuid)
        if len(messages) < _MIN_TURNS_FOR_SUMMARY:
            logger.info(
                "[summarize_session] conv=%s has %d messages — skipping",
                conv_uuid,
                len(messages),
            )
            return None

        # 1. Compact dialog transcript (bounded by _MAX_DIALOG_CHARS).
        dialog_text = _format_dialog_compact(messages, _MAX_DIALOG_CHARS)

        # 2. Generate session summary via the main LLM.
        llm = get_llm_client()
        session_summary = llm.invoke(
            f"{_SUMMARY_PROMPT_SYSTEM}\n\nДиалог:\n{dialog_text}\n\nСуммаризация:",
            max_tokens=_SUMMARY_MAX_TOKENS,
        ).strip()

        # 3. Topics — rule-based from message metadata.
        topics = _extract_topics(messages)

        # 4. Persist into conversation_summaries.
        memory.summary_repo.create(
            conversation_id=conv_uuid,
            user_id=user_uuid,
            summary=session_summary,
            topics=topics,
            turn_count=len(messages),
        )

        # 5. Recompute preferences across the whole user history.
        #    Uses only this conversation's metadata for the per-turn path;
        #    the rolling-summary meta-step below widens the window.
        _refresh_preferences(memory, user_uuid, conv_uuid)

        # 6. Rolling context_summary across recent sessions.
        _refresh_rolling_summary(memory, user_uuid, llm)

        logger.info(
            "[summarize_session] conv=%s user=%s summarised (%d turns, %d topics)",
            conv_uuid,
            user_uuid,
            len(messages),
            len(topics),
        )
        return {
            "conversation_id": str(conv_uuid),
            "summary": session_summary,
            "topics": topics,
            "turn_count": len(messages),
        }
    except Exception as e:
        logger.error("[summarize_session] failed for conv=%s: %s", conv_uuid, e, exc_info=True)
        return None
    finally:
        if db is not None:
            db.close()


@celery_app.task(name="hse_prom_prog.tasks.update_profile_async")
def update_profile_async(user_id: str, conversation_id: str) -> dict[str, Any] | None:
    """Recompute preferences for ``user_id`` from the given conversation.

    Fired from ``workflow_task`` after each turn — deliberately cheap
    (rule-based, no LLM call) so it can run on every save_turn without
    hammering the GPU.
    """
    user_uuid = UUID(user_id)
    conv_uuid = UUID(conversation_id)
    MEMORY_PROFILE_UPDATES.inc()

    db = None
    try:
        db = get_database()
        memory = MemoryManager(db)
        preferences = memory.update_profile(user_uuid, conv_uuid)
        logger.info(
            "[update_profile_async] user=%s preferences=%s",
            user_uuid,
            list(preferences.keys()),
        )
        return preferences
    except Exception as e:
        logger.warning("[update_profile_async] failed for user=%s: %s", user_uuid, e)
        return None
    finally:
        if db is not None:
            db.close()


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #


def _format_dialog_compact(messages: list, max_chars: int) -> str:
    """Render messages as ``Role: content`` lines, bounded by ``max_chars``.

    Truncates the *middle* rather than the tail so both opening context and
    resolution are preserved — the opening usually carries user intent and
    the tail carries the answer.
    """
    lines = [f"{m.role.capitalize()}: {m.content}" for m in messages]
    full = "\n".join(lines)
    if len(full) <= max_chars:
        return full
    head = full[: max_chars // 2]
    tail = full[-max_chars // 2 :]
    return f"{head}\n...[...]...\n{tail}"


def _extract_topics(messages: list) -> list[str]:
    """Rule-based topic extraction from message metadata.

    Collects unique values of ``team_name`` / ``metric_name`` /
    ``sprint_name`` across all messages in order of first occurrence.
    """
    topics: list[str] = []
    seen: set[str] = set()
    for msg in messages:
        meta = msg.metadata or {}
        entities = meta.get("entities") or {}
        if not isinstance(entities, dict):
            continue
        for field in ("team_name", "metric_name", "sprint_name"):
            value = entities.get(field)
            if isinstance(value, str) and value and value not in seen:
                seen.add(value)
                topics.append(value)
    return topics


def _refresh_preferences(
    memory: MemoryManager,
    user_id: UUID,
    conversation_id: UUID,
) -> None:
    try:
        memory.update_profile(user_id, conversation_id)
    except Exception as e:
        logger.warning("[summarize_session] preferences refresh failed: %s", e)


def _refresh_rolling_summary(memory: MemoryManager, user_id: UUID, llm: Any) -> None:
    """Update ``user_profiles.context_summary`` from recent summaries.

    When there are enough prior summaries we ask the LLM for a meta-summary
    (3-4 sentences); otherwise we just concatenate the existing ones so
    the profile still reflects recent activity.
    """
    try:
        recent = memory.summary_repo.get_recent(user_id, limit=_RECENT_SUMMARIES_WINDOW)
        if not recent:
            return

        if len(recent) < _MIN_SUMMARIES_FOR_META:
            combined = " ".join(s.summary for s in recent if s.summary)
            memory.profile_repo.update_context_summary(user_id, combined)
            return

        bullet_list = "\n".join(
            f"- {s.created_at:%Y-%m-%d}: {s.summary}" for s in recent if s.summary
        )
        meta = llm.invoke(
            f"{_ROLLING_SUMMARY_PROMPT_SYSTEM}\n\nИстория:\n{bullet_list}\n\nПрофиль:",
            max_tokens=_ROLLING_SUMMARY_MAX_TOKENS,
        ).strip()
        memory.profile_repo.update_context_summary(user_id, meta)
    except Exception as e:
        logger.warning("[summarize_session] rolling summary refresh failed: %s", e)
