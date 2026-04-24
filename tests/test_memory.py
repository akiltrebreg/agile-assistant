"""Unit tests for the memory layer.

Covers:
  * TokenEstimator            (4 tests)
  * Truncator                 (3 tests)
  * ContextBuilder            (5 tests)
  * ProfileExtractor          (4 tests)
  * MemoryManager.save_turn   (3 tests)
"""

from datetime import datetime
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from sqlalchemy.exc import IntegrityError

from hse_prom_prog.memory.context_builder import ContextBuilder
from hse_prom_prog.memory.manager import BOT_TRUNCATE_TOKENS, MemoryManager
from hse_prom_prog.memory.profile_extractor import ProfileExtractor
from hse_prom_prog.memory.token_estimator import CHARS_PER_TOKEN, estimate_tokens
from hse_prom_prog.memory.truncator import ELLIPSIS, truncate_message
from hse_prom_prog.models.memory import Conversation, Message

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────


def _conv(
    conv_id: UUID | None = None,
    summary: str | None = None,
    summary_turn_index: int = 0,
    title: str | None = None,
) -> Conversation:
    return Conversation(
        id=conv_id or uuid4(),
        user_id=None,
        title=title,
        summary=summary,
        summary_turn_index=summary_turn_index,
        created_at=datetime(2026, 4, 24, 10, 0, 0),
        updated_at=datetime(2026, 4, 24, 10, 0, 0),
        is_active=True,
    )


def _msg(  # noqa: PLR0913
    turn_index: int,
    role: str,
    content: str,
    *,
    conversation_id: UUID | None = None,
    content_truncated: str | None = None,
    metadata: dict | None = None,
) -> Message:
    return Message(
        id=uuid4(),
        conversation_id=conversation_id or uuid4(),
        turn_index=turn_index,
        role=role,
        content=content,
        content_truncated=content_truncated,
        metadata=metadata or {},
        created_at=datetime(2026, 4, 24, 10, turn_index, 0),
    )


# ────────────────────────────────────────────────────────────────
# TokenEstimator
# ────────────────────────────────────────────────────────────────


class TestTokenEstimator:
    def test_empty_string_is_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_russian_text_uses_char_heuristic(self) -> None:
        text = "Какой velocity у команды cthulhu"  # 32 chars
        assert estimate_tokens(text) == len(text) // CHARS_PER_TOKEN

    def test_english_text_uses_same_heuristic(self) -> None:
        text = "What is the velocity of the team"  # 32 chars
        assert estimate_tokens(text) == len(text) // CHARS_PER_TOKEN

    def test_short_word_counts_at_least_one_token(self) -> None:
        # "ab" → 0 // 3 == 0, but any non-empty text must be ≥ 1 token.
        assert estimate_tokens("ab") == 1


# ────────────────────────────────────────────────────────────────
# Truncator
# ────────────────────────────────────────────────────────────────


class TestTruncator:
    def test_short_text_returned_unchanged(self) -> None:
        text = "короткий ответ"
        assert truncate_message(text, max_tokens=100) == text

    def test_long_text_is_cut_with_ellipsis(self) -> None:
        text = "слово " * 100  # 600 chars
        out = truncate_message(text, max_tokens=20)  # budget ≈ 60 chars
        assert out.endswith(ELLIPSIS)
        assert len(out) <= 20 * CHARS_PER_TOKEN

    def test_cut_snaps_to_word_boundary(self) -> None:
        text = "один два три четыре пять шесть семь восемь девять десять"
        out = truncate_message(text, max_tokens=10)  # budget ≈ 30 chars
        body = out.removesuffix(ELLIPSIS).rstrip()
        # Body must not end in the middle of a word.
        assert not body or body[-1] != " "
        assert " " not in body[-1:] or body.endswith(" ")
        # And every retained token is a prefix word from the original text.
        for tok in body.split():
            assert tok in text.split()


# ────────────────────────────────────────────────────────────────
# ContextBuilder
# ────────────────────────────────────────────────────────────────


class TestContextBuilder:
    def _builder(self, conversation: Conversation, messages: list[Message]) -> ContextBuilder:
        repo = MagicMock()
        repo.get.return_value = conversation
        repo.get_messages.return_value = messages
        return ContextBuilder(repo)

    def test_empty_history_returns_empty_context(self) -> None:
        conv = _conv()
        builder = self._builder(conv, [])

        ctx = builder.build(conv.id, token_budget=1000)

        assert ctx["recent_turns"] == []
        assert ctx["summary"] == ""
        assert ctx["history_token_count"] == 0
        assert ctx["needs_summarization"] is False

    def test_single_turn_fits_in_budget(self) -> None:
        conv = _conv()
        messages = [
            _msg(0, "user", "Привет", conversation_id=conv.id),
            _msg(1, "assistant", "Здравствуйте", conversation_id=conv.id),
        ]
        builder = self._builder(conv, messages)

        ctx = builder.build(conv.id, token_budget=1000)

        assert len(ctx["recent_turns"]) == 2
        assert ctx["recent_turns"][0]["role"] == "user"
        assert ctx["recent_turns"][1]["role"] == "assistant"
        assert ctx["needs_summarization"] is False

    def test_three_turns_all_fit_when_budget_is_generous(self) -> None:
        conv = _conv()
        messages = [
            _msg(i, "user" if i % 2 == 0 else "assistant", f"m{i} " * 5, conversation_id=conv.id)
            for i in range(6)
        ]
        builder = self._builder(conv, messages)

        ctx = builder.build(conv.id, token_budget=5000)

        assert len(ctx["recent_turns"]) == 6
        assert [t["turn_index"] for t in ctx["recent_turns"]] == list(range(6))

    def test_adaptive_k_drops_oldest_when_budget_tight(self) -> None:
        conv = _conv()
        # 10 messages, each ~300 chars → ~100 tokens → +2 = 102 tok cost
        messages = [
            _msg(i, "user" if i % 2 == 0 else "assistant", "x" * 300, conversation_id=conv.id)
            for i in range(10)
        ]
        builder = self._builder(conv, messages)

        ctx = builder.build(conv.id, token_budget=300)  # ≈ 3 messages fit

        assert 1 <= len(ctx["recent_turns"]) < 10
        # Kept messages must be the newest (contiguous tail).
        kept_idx = [t["turn_index"] for t in ctx["recent_turns"]]
        assert kept_idx == sorted(kept_idx)
        assert kept_idx[-1] == 9
        # And anything older than the oldest kept should trigger summarisation.
        assert ctx["needs_summarization"] is True

    def test_zero_budget_still_keeps_at_least_one_turn(self) -> None:
        conv = _conv()
        messages = [
            _msg(0, "user", "hi", conversation_id=conv.id),
            _msg(1, "assistant", "hello", conversation_id=conv.id),
        ]
        builder = self._builder(conv, messages)

        ctx = builder.build(conv.id, token_budget=0)

        # Guarantee from the algorithm: never return 0 turns when history exists.
        assert len(ctx["recent_turns"]) >= 1
        assert ctx["recent_turns"][-1]["turn_index"] == 1


# ────────────────────────────────────────────────────────────────
# ProfileExtractor
# ────────────────────────────────────────────────────────────────


class TestProfileExtractor:
    def test_dominant_team_becomes_default(self) -> None:
        extractor = ProfileExtractor()
        metadata = [
            {"entities": {"team_name": "cthulhu"}, "query_type": "sql"} for _ in range(8)
        ] + [{"entities": {"team_name": "other"}, "query_type": "sql"}]

        prefs = extractor.extract(metadata)

        assert prefs["default_team"] == "cthulhu"

    def test_even_split_leaves_no_default_team(self) -> None:
        extractor = ProfileExtractor()
        metadata = [
            {"entities": {"team_name": "a"}},
            {"entities": {"team_name": "b"}},
            {"entities": {"team_name": "a"}},
            {"entities": {"team_name": "b"}},
        ]

        prefs = extractor.extract(metadata)

        assert "default_team" not in prefs

    def test_empty_input_returns_empty_preferences(self) -> None:
        extractor = ProfileExtractor()

        prefs = extractor.extract([])

        assert prefs == {}

    def test_detail_level_brief_when_sql_dominates(self) -> None:
        extractor = ProfileExtractor()
        metadata = [{"query_type": "sql"} for _ in range(5)] + [
            {"query_type": "rag"},
        ]

        prefs = extractor.extract(metadata)

        assert prefs["preferred_detail_level"] == "brief"

    def test_detail_level_detailed_when_rag_and_hybrid_dominate(self) -> None:
        extractor = ProfileExtractor()
        metadata = [
            {"query_type": "rag"},
            {"query_type": "rag"},
            {"query_type": "hybrid"},
            {"query_type": "sql"},
        ]

        prefs = extractor.extract(metadata)

        assert prefs["preferred_detail_level"] == "detailed"


# ────────────────────────────────────────────────────────────────
# MemoryManager.save_turn
# ────────────────────────────────────────────────────────────────


class TestMemoryManagerSaveTurn:
    def _make_manager(
        self,
        *,
        initial_latest_idx: int = -1,
        conversation: Conversation | None = None,
    ) -> tuple[MemoryManager, MagicMock]:
        """Build a MemoryManager with all repositories mocked."""
        conv_repo = MagicMock()
        conv_repo.get_latest_turn_index.return_value = initial_latest_idx
        conv_repo.get.return_value = conversation
        # save_message returns a minimal Message-shaped object.
        conv_repo.save_message.side_effect = lambda **kw: _msg(
            turn_index=kw["turn_index"],
            role=kw["role"],
            content=kw["content"],
            conversation_id=kw["conversation_id"],
            content_truncated=kw.get("content_truncated"),
            metadata=kw.get("metadata") or {},
        )

        manager = MemoryManager(
            db=MagicMock(),
            conversation_repo=conv_repo,
            profile_repo=MagicMock(),
            summary_repo=MagicMock(),
            context_builder=MagicMock(),
            profile_extractor=MagicMock(),
        )
        return manager, conv_repo

    def test_first_turn_writes_indices_0_and_1_and_sets_title(self) -> None:
        conv_id = uuid4()
        conv = _conv(conv_id=conv_id, title=None)
        manager, repo = self._make_manager(initial_latest_idx=-1, conversation=conv)

        user_idx, bot_idx = manager.save_turn(
            conversation_id=conv_id,
            user_message="Какой velocity у команды cthulhu?",
            bot_message="Velocity: 42 SP.",
            metadata={"query_type": "sql"},
        )

        assert user_idx == 0
        assert bot_idx == 1

        # Two inserts: user (idx 0), assistant (idx 1).
        saved = repo.save_message.call_args_list
        assert len(saved) == 2
        assert saved[0].kwargs["turn_index"] == 0
        assert saved[0].kwargs["role"] == "user"
        assert saved[0].kwargs["content_truncated"] is None
        assert saved[1].kwargs["turn_index"] == 1
        assert saved[1].kwargs["role"] == "assistant"
        # Title derived from the first query.
        repo.update_title.assert_called_once()
        assert repo.update_title.call_args.args[1].startswith("Какой velocity")

    def test_subsequent_turn_continues_index_and_touches_conversation(self) -> None:
        conv_id = uuid4()
        manager, repo = self._make_manager(initial_latest_idx=3)

        user_idx, bot_idx = manager.save_turn(
            conversation_id=conv_id,
            user_message="Ещё один вопрос",
            bot_message="Ответ",
        )

        assert (user_idx, bot_idx) == (4, 5)
        repo.update_title.assert_not_called()
        repo.touch.assert_called_once_with(conv_id)

    def test_bot_message_is_truncated_long_user_message_is_not(self) -> None:
        conv_id = uuid4()
        manager, repo = self._make_manager(initial_latest_idx=0)

        long_user = "вопрос " * 500  # 3500 chars
        long_bot = "развёрнутый ответ " * 200  # 3600 chars

        manager.save_turn(
            conversation_id=conv_id,
            user_message=long_user,
            bot_message=long_bot,
        )

        user_call, bot_call = repo.save_message.call_args_list
        # User content_truncated stays None — full content is preserved in DB.
        assert user_call.kwargs["role"] == "user"
        assert user_call.kwargs["content_truncated"] is None
        assert user_call.kwargs["content"] == long_user
        # Bot content_truncated is populated and within the token budget.
        assert bot_call.kwargs["role"] == "assistant"
        truncated = bot_call.kwargs["content_truncated"]
        assert truncated is not None
        assert truncated.endswith(ELLIPSIS)
        assert len(truncated) <= BOT_TRUNCATE_TOKENS * CHARS_PER_TOKEN
        # Full content still saved — truncation is for replay, not storage.
        assert bot_call.kwargs["content"] == long_bot

    def test_integrity_error_triggers_retry(self) -> None:
        """UNIQUE(conversation_id, turn_index) race → retry with fresh max."""
        conv_id = uuid4()
        conv_repo = MagicMock()
        # First call: latest=0 → try (1,2) but IntegrityError.
        # Second call: latest=2 → try (3,4), succeeds.
        conv_repo.get_latest_turn_index.side_effect = [0, 2]
        conv_repo.get.return_value = _conv(conv_id=conv_id, title="existing")

        call_counter = {"n": 0}

        def save_effect(**kw):
            call_counter["n"] += 1
            # First user-row save fails; after retry, all save_message calls succeed.
            if call_counter["n"] == 1:
                raise IntegrityError("dup", {}, Exception("dup"))
            return _msg(
                turn_index=kw["turn_index"],
                role=kw["role"],
                content=kw["content"],
                conversation_id=kw["conversation_id"],
            )

        conv_repo.save_message.side_effect = save_effect

        manager = MemoryManager(
            db=MagicMock(),
            conversation_repo=conv_repo,
            profile_repo=MagicMock(),
            summary_repo=MagicMock(),
            context_builder=MagicMock(),
            profile_extractor=MagicMock(),
        )

        user_idx, bot_idx = manager.save_turn(
            conversation_id=conv_id,
            user_message="u",
            bot_message="b",
        )

        assert (user_idx, bot_idx) == (3, 4)
        assert conv_repo.get_latest_turn_index.call_count == 2
