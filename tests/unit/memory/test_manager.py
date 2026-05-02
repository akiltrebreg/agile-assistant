"""Unit tests for ``MemoryManager`` — the facade over the memory repositories.

The manager itself is thin: it routes calls to repos / builders / extractors
and adds two pieces of real logic on top:

  1. ``save_turn`` retries on UNIQUE(conversation_id, turn_index) races —
     the first concurrent writer wins, the loser re-reads the latest index.
  2. ``save_turn`` derives the conversation title from the first user
     message, then ``touch``-es on every subsequent turn.

All tests inject mocks for every dependency (``ConversationRepository``,
``ProfileRepository``, ``SummaryRepository``, ``ContextBuilder``,
``ProfileExtractor``) — the default-construction path that wires real
repos to a real DB is excluded from unit coverage by design.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from hse_prom_prog.memory.manager import (
    BOT_TRUNCATE_TOKENS,
    SAVE_TURN_MAX_RETRIES,
    TITLE_MAX_CHARS,
    MemoryManager,
)
from hse_prom_prog.memory.truncator import truncate_message
from hse_prom_prog.models.memory import Conversation, Message

# --------------------------------------------------------------------- #
# Local helpers / fixtures
# --------------------------------------------------------------------- #


def _make_conversation(
    *, title: str | None = None, conversation_id: UUID | None = None
) -> Conversation:
    now = datetime.now(UTC)
    return Conversation(
        id=conversation_id or uuid4(),
        user_id=uuid4(),
        title=title,
        summary=None,
        summary_turn_index=0,
        created_at=now,
        updated_at=now,
        is_active=True,
    )


def _make_message(turn_index: int, role: str = "user", content: str = "x") -> Message:
    return Message(
        id=uuid4(),
        conversation_id=uuid4(),
        turn_index=turn_index,
        role=role,
        content=content,
        content_truncated=None,
        metadata={},
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def conv_repo() -> MagicMock:
    r = MagicMock()
    r.get = MagicMock(return_value=None)
    r.get_or_create = MagicMock()
    r.get_messages = MagicMock(return_value=[])
    r.get_latest_turn_index = MagicMock(return_value=-1)
    r.save_message = MagicMock()
    r.update_title = MagicMock()
    r.touch = MagicMock()
    return r


@pytest.fixture
def profile_repo() -> MagicMock:
    r = MagicMock()
    r.get = MagicMock(return_value=None)
    r.get_or_create = MagicMock()
    r.update_preferences = MagicMock()
    return r


@pytest.fixture
def summary_repo() -> MagicMock:
    return MagicMock()


@pytest.fixture
def context_builder() -> MagicMock:
    b = MagicMock()
    b.build = MagicMock(
        return_value={
            "summary": "",
            "recent_turns": [],
            "history_token_count": 0,
            "needs_summarization": False,
        }
    )
    return b


@pytest.fixture
def profile_extractor() -> MagicMock:
    e = MagicMock()
    e.extract = MagicMock(return_value={})
    return e


@pytest.fixture
def manager(
    conv_repo: MagicMock,
    profile_repo: MagicMock,
    summary_repo: MagicMock,
    context_builder: MagicMock,
    profile_extractor: MagicMock,
) -> MemoryManager:
    # `db` is unused when every repo / helper is injected — pass a sentinel.
    return MemoryManager(
        db=MagicMock(),
        conversation_repo=conv_repo,
        profile_repo=profile_repo,
        summary_repo=summary_repo,
        context_builder=context_builder,
        profile_extractor=profile_extractor,
    )


# ===================================================================== #
# Pass-through delegations
# ===================================================================== #


@pytest.mark.unit
class TestPassthroughs:
    def test_get_or_create_conversation_delegates(
        self, manager: MemoryManager, conv_repo: MagicMock
    ) -> None:
        conv = _make_conversation()
        conv_repo.get_or_create.return_value = conv
        cid, uid = uuid4(), uuid4()

        result = manager.get_or_create_conversation(cid, uid)
        assert result is conv
        conv_repo.get_or_create.assert_called_once_with(cid, uid)

    def test_get_context_delegates_to_builder(
        self, manager: MemoryManager, context_builder: MagicMock
    ) -> None:
        ctx = {
            "summary": "x",
            "recent_turns": [],
            "history_token_count": 0,
            "needs_summarization": False,
        }
        context_builder.build.return_value = ctx
        cid = uuid4()

        result = manager.get_context(cid, token_budget=800)
        assert result is ctx
        context_builder.build.assert_called_once_with(cid, 800)

    def test_get_profile_returns_none_when_missing(
        self, manager: MemoryManager, profile_repo: MagicMock
    ) -> None:
        profile_repo.get.return_value = None
        assert manager.get_profile(uuid4()) is None

    def test_get_profile_returns_dict_when_present(
        self, manager: MemoryManager, profile_repo: MagicMock
    ) -> None:
        profile = MagicMock()
        profile.to_dict.return_value = {"preferences": {"default_team": "cthulhu"}}
        profile_repo.get.return_value = profile
        result = manager.get_profile(uuid4())
        assert result == {"preferences": {"default_team": "cthulhu"}}

    def test_get_or_create_profile_by_external_id_delegates(
        self, manager: MemoryManager, profile_repo: MagicMock
    ) -> None:
        profile = MagicMock()
        profile.to_dict.return_value = {"external_id": "abc"}
        profile_repo.get_or_create.return_value = profile
        result = manager.get_or_create_profile_by_external_id("abc")
        assert result == {"external_id": "abc"}
        profile_repo.get_or_create.assert_called_once_with("abc")


# ===================================================================== #
# save_turn — happy path
# ===================================================================== #


@pytest.mark.unit
class TestSaveTurn:
    def test_first_turn_writes_user_then_assistant(
        self, manager: MemoryManager, conv_repo: MagicMock
    ) -> None:
        # latest=-1 means no turns yet → first user gets index 0, bot index 1.
        conv_repo.get_latest_turn_index.return_value = -1
        conv_repo.get.return_value = _make_conversation(title=None)

        cid = uuid4()
        user_idx, bot_idx = manager.save_turn(cid, "вопрос", "ответ", {"intent": "task"})

        assert (user_idx, bot_idx) == (0, 1)
        # Two save_message calls — order matters (user then assistant).
        assert conv_repo.save_message.call_count == 2
        first_call = conv_repo.save_message.call_args_list[0].kwargs
        second_call = conv_repo.save_message.call_args_list[1].kwargs
        assert first_call["role"] == "user"
        assert first_call["turn_index"] == 0
        assert first_call["content_truncated"] is None
        assert second_call["role"] == "assistant"
        assert second_call["turn_index"] == 1
        # Long bot replies get a truncated copy — short ones get the full body.
        assert second_call["content_truncated"] == truncate_message("ответ", BOT_TRUNCATE_TOKENS)

    def test_first_turn_sets_title_from_user_message(
        self, manager: MemoryManager, conv_repo: MagicMock
    ) -> None:
        cid = uuid4()
        conv = _make_conversation(title=None, conversation_id=cid)
        conv_repo.get_latest_turn_index.return_value = -1
        conv_repo.get.return_value = conv

        manager.save_turn(cid, "Покажи velocity команды cthulhu", "ok")

        conv_repo.update_title.assert_called_once()
        call_args = conv_repo.update_title.call_args
        assert call_args.args[0] == cid
        assert call_args.args[1] == "Покажи velocity команды cthulhu"
        # touch() is the post-first-turn path — must NOT fire on the first save.
        conv_repo.touch.assert_not_called()

    def test_long_first_message_title_truncated(
        self, manager: MemoryManager, conv_repo: MagicMock
    ) -> None:
        conv_repo.get_latest_turn_index.return_value = -1
        conv_repo.get.return_value = _make_conversation(title=None)

        long_msg = "А" * 200
        manager.save_turn(uuid4(), long_msg, "ok")

        title_arg = conv_repo.update_title.call_args.args[1]
        assert len(title_arg) == TITLE_MAX_CHARS
        assert title_arg == "А" * TITLE_MAX_CHARS

    def test_first_turn_with_empty_user_message_skips_title(
        self, manager: MemoryManager, conv_repo: MagicMock
    ) -> None:
        # `.strip()` then truthiness check — pure whitespace must NOT cause
        # an empty title to be persisted.
        conv_repo.get_latest_turn_index.return_value = -1
        conv_repo.get.return_value = _make_conversation(title=None)

        manager.save_turn(uuid4(), "   ", "ok")

        conv_repo.update_title.assert_not_called()

    def test_first_turn_does_not_overwrite_existing_title(
        self, manager: MemoryManager, conv_repo: MagicMock
    ) -> None:
        # Title was set out-of-band (e.g. user renamed the chat) — must
        # not be clobbered by the auto-derived one.
        conv_repo.get_latest_turn_index.return_value = -1
        conv_repo.get.return_value = _make_conversation(title="My pinned chat")
        manager.save_turn(uuid4(), "first msg", "ok")
        conv_repo.update_title.assert_not_called()

    def test_subsequent_turn_increments_indices_and_touches(
        self, manager: MemoryManager, conv_repo: MagicMock
    ) -> None:
        # latest=5 → next user idx=6, bot idx=7. Title path skipped, touch fires.
        conv_repo.get_latest_turn_index.return_value = 5
        cid = uuid4()
        user_idx, bot_idx = manager.save_turn(cid, "q", "a")
        assert (user_idx, bot_idx) == (6, 7)
        conv_repo.update_title.assert_not_called()
        conv_repo.touch.assert_called_once_with(cid)

    def test_long_bot_message_persisted_with_truncated_copy(
        self, manager: MemoryManager, conv_repo: MagicMock
    ) -> None:
        # BOT_TRUNCATE_TOKENS=150 → ~450 char budget; a 5000-char reply
        # gets cut and stored under content_truncated for cheap replay.
        conv_repo.get_latest_turn_index.return_value = 0
        long_reply = "слово " * 1000
        manager.save_turn(uuid4(), "q", long_reply)

        bot_call = conv_repo.save_message.call_args_list[1].kwargs
        # Sanity: actually shorter than the original.
        assert bot_call["content_truncated"] is not None
        assert len(bot_call["content_truncated"]) < len(long_reply)


# ===================================================================== #
# save_turn — UNIQUE race retry
# ===================================================================== #


@pytest.mark.unit
class TestSaveTurnRace:
    def _integrity_error(self) -> IntegrityError:
        return IntegrityError("INSERT", {}, Exception("duplicate turn_index"))

    def test_second_attempt_succeeds_after_one_race(
        self, manager: MemoryManager, conv_repo: MagicMock
    ) -> None:
        # First write of the user message raises IntegrityError; second
        # attempt re-reads latest and succeeds.
        conv_repo.get_latest_turn_index.side_effect = [-1, 1]
        conv_repo.save_message.side_effect = [self._integrity_error(), None, None]
        conv_repo.get.return_value = _make_conversation(title=None)

        user_idx, bot_idx = manager.save_turn(uuid4(), "q", "a")
        # Second attempt landed on indices 2 and 3.
        assert (user_idx, bot_idx) == (2, 3)

    def test_persistent_race_raises_after_max_retries(
        self, manager: MemoryManager, conv_repo: MagicMock
    ) -> None:
        # Every attempt loses the race — manager must give up after
        # SAVE_TURN_MAX_RETRIES (3) and re-raise the last IntegrityError
        # so the upstream handler sees the failure (vs silently dropping).
        conv_repo.get_latest_turn_index.return_value = -1
        conv_repo.save_message.side_effect = self._integrity_error()

        with pytest.raises(IntegrityError):
            manager.save_turn(uuid4(), "q", "a")

        # Each attempt re-reads latest_turn_index and tries the user save.
        assert conv_repo.get_latest_turn_index.call_count == SAVE_TURN_MAX_RETRIES

    def test_retry_constants_pinned(self) -> None:
        # The race-retry budget and bot-truncation budget are tuned together
        # — surface a deliberate change as a test-edit, not a silent regression.
        assert SAVE_TURN_MAX_RETRIES == 3
        assert BOT_TRUNCATE_TOKENS == 150
        assert TITLE_MAX_CHARS == 60


# ===================================================================== #
# update_profile — extractor + repo wiring
# ===================================================================== #


@pytest.mark.unit
class TestUpdateProfile:
    def test_passes_message_metadata_through_extractor(
        self,
        manager: MemoryManager,
        conv_repo: MagicMock,
        profile_extractor: MagicMock,
        profile_repo: MagicMock,
    ) -> None:
        # Three messages, one with empty metadata (must be filtered out
        # before reaching the extractor).
        msgs = [_make_message(0), _make_message(1), _make_message(2)]
        msgs[0].metadata = {"entities": {"team_name": "cthulhu"}}
        msgs[1].metadata = {}
        msgs[2].metadata = {"query_type": "sql"}
        conv_repo.get_messages.return_value = msgs

        profile_extractor.extract.return_value = {"default_team": "cthulhu"}

        uid, cid = uuid4(), uuid4()
        result = manager.update_profile(uid, cid)

        # Extractor saw only the two non-empty metadata dicts.
        assert profile_extractor.extract.call_count == 1
        passed = profile_extractor.extract.call_args.args[0]
        assert {"entities": {"team_name": "cthulhu"}} in passed
        assert {"query_type": "sql"} in passed
        assert {} not in passed

        # And the result was persisted under the user's profile.
        profile_repo.update_preferences.assert_called_once_with(uid, {"default_team": "cthulhu"})
        assert result == {"default_team": "cthulhu"}

    def test_returns_empty_when_extractor_yields_nothing(
        self,
        manager: MemoryManager,
        conv_repo: MagicMock,
        profile_extractor: MagicMock,
        profile_repo: MagicMock,
    ) -> None:
        # Below-min-evidence case from the extractor — manager still calls
        # update_preferences (with {}), the repo call is the contract.
        conv_repo.get_messages.return_value = [_make_message(0)]
        profile_extractor.extract.return_value = {}

        result = manager.update_profile(uuid4(), uuid4())
        assert result == {}
        profile_repo.update_preferences.assert_called_once()


# ===================================================================== #
# Default-wiring smoke
# ===================================================================== #


@pytest.mark.unit
class TestDefaultWiring:
    def test_construct_with_only_db_initialises_default_helpers(self) -> None:
        # The constructor lazily defaults every helper to a real impl
        # bound to the supplied db. We don't exercise those helpers here —
        # just confirm the manager doesn't crash when only `db` is passed.
        # This is the path used by workflow_task / API.
        from hse_prom_prog.memory.context_builder import ContextBuilder
        from hse_prom_prog.memory.conversation_repo import ConversationRepository
        from hse_prom_prog.memory.profile_extractor import ProfileExtractor
        from hse_prom_prog.memory.profile_repo import ProfileRepository
        from hse_prom_prog.memory.summary_repo import SummaryRepository

        mgr = MemoryManager(db=MagicMock())
        assert isinstance(mgr.conversation_repo, ConversationRepository)
        assert isinstance(mgr.profile_repo, ProfileRepository)
        assert isinstance(mgr.summary_repo, SummaryRepository)
        assert isinstance(mgr.context_builder, ContextBuilder)
        assert isinstance(mgr.profile_extractor, ProfileExtractor)
