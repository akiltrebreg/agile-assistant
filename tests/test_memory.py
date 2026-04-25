"""Unit tests for the memory layer.

Covers:
  * TokenEstimator            (4 tests)
  * Truncator                 (3 tests)
  * ContextBuilder            (5 tests)
  * ProfileExtractor          (4 tests)
  * MemoryManager.save_turn   (3 tests)
  * format_history            (3 tests)
  * EntitySanitizer carry-fwd (3 tests)
  * Supervisor + context      (2 tests)
  * ResponseAgent + history   (2 tests)
"""

from datetime import datetime
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from sqlalchemy.exc import IntegrityError

from hse_prom_prog.agents.entity_sanitizer import (
    _carry_forward_entities,
    _has_anaphora,
)
from hse_prom_prog.agents.response_agent import (
    _BRANCH_HISTORY_BUDGET,
    _MIN_HISTORY_BUDGET_TOKENS,
    ResponseAgent,
)
from hse_prom_prog.agents.supervisor import SupervisorAgent
from hse_prom_prog.memory.context_builder import ContextBuilder
from hse_prom_prog.memory.formatter import format_history
from hse_prom_prog.memory.manager import BOT_TRUNCATE_TOKENS, MemoryManager
from hse_prom_prog.memory.profile_extractor import ProfileExtractor
from hse_prom_prog.memory.token_estimator import CHARS_PER_TOKEN, estimate_tokens
from hse_prom_prog.memory.truncator import ELLIPSIS, truncate_message
from hse_prom_prog.models.memory import Conversation, ConversationContext, Message

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
        # 6 messages so the min_messages gate (5) is cleared and we
        # genuinely exercise the share-threshold logic, not the gate.
        metadata = [
            {"entities": {"team_name": "a"}},
            {"entities": {"team_name": "b"}},
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
        # min_messages=0 — focused unit test on the detail-level rule,
        # the entity-count gate is exercised separately below.
        metadata = [{"query_type": "sql"} for _ in range(5)] + [
            {"query_type": "rag"},
        ]

        prefs = extractor.extract(metadata, min_messages=0)

        assert prefs["preferred_detail_level"] == "brief"

    def test_detail_level_detailed_when_rag_and_hybrid_dominate(self) -> None:
        extractor = ProfileExtractor()
        metadata = [
            {"query_type": "rag"},
            {"query_type": "rag"},
            {"query_type": "hybrid"},
            {"query_type": "sql"},
        ]

        prefs = extractor.extract(metadata, min_messages=0)

        assert prefs["preferred_detail_level"] == "detailed"

    def test_min_messages_gate_blocks_premature_default_team(self) -> None:
        """Three same-team messages must NOT yet produce a default_team.

        Without the gate, the first 2-3 queries about the same team would
        lock in default_team at 100% share — the supervisor would then
        inject it into every subsequent unrelated query. The gate keeps
        preferences empty until enough signal accumulates.
        """
        extractor = ProfileExtractor()
        below = [{"entities": {"team_name": "cthulhu"}} for _ in range(3)]
        above = [{"entities": {"team_name": "cthulhu"}} for _ in range(6)]

        assert extractor.extract(below) == {}
        assert extractor.extract(above)["default_team"] == "cthulhu"


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


# ────────────────────────────────────────────────────────────────
# format_history
# ────────────────────────────────────────────────────────────────


def _ctx(
    recent: list[dict] | None = None,
    summary: str = "",
) -> ConversationContext:
    """Build a minimal ConversationContext for tests."""
    return {
        "summary": summary,
        "recent_turns": recent or [],
        "history_token_count": 0,
        "needs_summarization": False,
    }


class TestFormatHistory:
    def test_empty_recent_returns_empty_string(self) -> None:
        assert format_history(_ctx()) == ""

    def test_none_context_returns_empty_string(self) -> None:
        assert format_history(None) == ""

    def test_summary_and_recent_are_rendered_in_order(self) -> None:
        ctx = _ctx(
            recent=[
                {"role": "user", "content": "Привет"},
                {"role": "assistant", "content": "Здравствуйте"},
            ],
            summary="Раньше обсуждали команду lpop.",
        )

        out = format_history(ctx)

        assert out.startswith("<conversation_history>")
        assert out.endswith("</conversation_history>")
        assert "<summary>Раньше обсуждали команду lpop.</summary>" in out
        # Recent turns appear inside <recent> with English role labels.
        assert out.index("<summary>") < out.index("<recent>")
        assert "User: Привет" in out
        assert "Assistant: Здравствуйте" in out
        # No <summary> when summary is empty.
        out_no_sum = format_history(_ctx(recent=[{"role": "user", "content": "x"}]))
        assert "<summary>" not in out_no_sum


# ────────────────────────────────────────────────────────────────
# EntitySanitizer: carry-forward (layer 6)
# ────────────────────────────────────────────────────────────────


class TestCarryForwardEntities:
    def test_anaphora_with_prev_team_substitutes(self) -> None:
        result = _carry_forward_entities(
            entities={"metric_name": "velocity"},
            prev_entities={"team_name": "cthulhu", "sprint_name": "#1 Q1'26"},
            user_query="А что по velocity у этой команды?",
        )

        assert result["team_name"] == "cthulhu"
        assert result["sprint_name"] == "#1 Q1'26"
        assert result["metric_name"] == "velocity"  # current entities preserved

    def test_no_anaphora_leaves_entities_untouched(self) -> None:
        result = _carry_forward_entities(
            entities={"metric_name": "velocity"},
            prev_entities={"team_name": "cthulhu"},
            user_query="Посчитай velocity",  # no anaphoric marker
        )

        assert "team_name" not in result
        assert result == {"metric_name": "velocity"}

    def test_current_value_beats_prev_even_with_anaphora(self) -> None:
        """Supervisor extracted a team from this turn — do not overwrite."""
        result = _carry_forward_entities(
            entities={"team_name": "newteam"},
            prev_entities={"team_name": "cthulhu"},
            user_query="А у этой команды newteam как дела?",
        )

        assert result["team_name"] == "newteam"

    def test_has_anaphora_detects_common_markers(self) -> None:
        assert _has_anaphora("А что по этой команде?")
        assert _has_anaphora("Покажи ещё")
        assert _has_anaphora("По ней тоже посмотри")
        assert not _has_anaphora("Какой velocity у команды cthulhu?")


# ────────────────────────────────────────────────────────────────
# Supervisor + conversation_context
# ────────────────────────────────────────────────────────────────


class TestSupervisorWithContext:
    def test_context_is_injected_into_llm_prompt(self) -> None:
        """Supervisor should include history block + resolution instruction."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"intent":"metric","query_type":"sql","entities":{"metric_name":"velocity"}}'
        )
        agent = SupervisorAgent(mock_llm)

        ctx = _ctx(
            recent=[
                {
                    "role": "user",
                    "content": "Задачи команды cthulhu",
                    "metadata": {"entities": {"team_name": "cthulhu"}},
                }
            ]
        )
        result = agent.process(
            "А что по velocity у этой команды?",
            conversation_context=ctx,
        )

        # Carry-forward filled team_name from prior turn.
        assert result["entities"].get("team_name") == "cthulhu"
        assert result["entities"].get("metric_name") == "velocity"
        # The history block reached the LLM prompt.
        prompt = mock_llm.invoke.call_args.args[0]
        assert "<conversation_history>" in prompt
        assert "cthulhu" in prompt
        assert "Используй историю" in prompt

    def test_no_context_leaves_prompt_unchanged(self) -> None:
        """Without context, Supervisor prompt does not contain history tags."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"intent":"general","query_type":"simple","entities":{}}'
        agent = SupervisorAgent(mock_llm)

        agent.process("Как дела?")

        prompt = mock_llm.invoke.call_args.args[0]
        assert "<conversation_history>" not in prompt
        assert "Используй историю" not in prompt

    def test_hybrid_without_marker_is_downgraded_to_sql(self) -> None:
        """Spurious LLM upgrade to hybrid → post-processing demotes it.

        Reproduces a real production failure: after a few turns the LLM
        starts returning ``query_type=hybrid`` for plain follow-ups (e.g.
        "scope drop?" with anaphora), which routes through the analysis
        branch of Response Agent and triggers a degenerate-loop on the
        avibe-gptq-8bit model. The advice signal lives in the wording —
        if no hybrid marker is present, the upgrade is spurious.
        """
        mock_llm = MagicMock()
        # LLM "upgrades" a plain follow-up to hybrid even though the
        # query has no hybrid markers ("улучшить", "снизить", "дай совет", …).
        # The team name is in the query so the sanitizer keeps it — that
        # lets us focus the assertion on the rule under test.
        mock_llm.invoke.return_value = (
            '{"intent":"metric","query_type":"hybrid",'
            '"entities":{"team_name":"cthulhu","metric_name":"scope_drop"}}'
        )
        agent = SupervisorAgent(mock_llm)

        result = agent.process("scope drop cthulhu")

        # Post-processing demotes hybrid → sql; intent + entities preserved.
        assert result["query_type"] == "sql"
        assert result["intent"] == "metric"
        assert result["entities"]["team_name"] == "cthulhu"
        assert result["entities"]["metric_name"] == "scope_drop"

    def test_hybrid_with_marker_stays_hybrid(self) -> None:
        """A genuine hybrid request with an advice marker is left alone."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"intent":"metric","query_type":"hybrid",'
            '"entities":{"team_name":"cthulhu","metric_name":"scope_drop"}}'
        )
        agent = SupervisorAgent(mock_llm)

        # "как улучшить" is in _HYBRID_MARKERS — genuine advice request.
        result = agent.process("scope drop cthulhu — как улучшить?")

        assert result["query_type"] == "hybrid"
        assert result["intent"] == "metric"

    def test_metric_plus_team_misrouted_to_rag_is_upgraded(self) -> None:
        """Rule 5: "Scope drop cthulhu" wrongly routed to rag → sql/metric.

        Reproduces eval Case 85: the LLM occasionally drops a terse
        "<metric> <team>" form into the rag bucket, despite the prompt's
        Шаг 4 forbidding rag when concrete identifiers are present.
        """
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"intent":"general","query_type":"rag","entities":{"team_name":"cthulhu"}}'
        )
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Scope drop cthulhu")

        assert result["query_type"] == "sql"
        assert result["intent"] == "metric"
        assert result["entities"]["team_name"] == "cthulhu"
        # metric_name was missing in LLM output — Rule 5 fills it in.
        assert result["entities"]["metric_name"] == "scope_drop"

    def test_generic_metric_word_plus_sprint_is_upgraded(self) -> None:
        """Rule 5: "Метрики за спринт X" → sql/metric, sprint preserved."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"intent":"general","query_type":"rag","entities":{"sprint_name":"#1 Q1\'26"}}'
        )
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Метрики за спринт #1 Q1'26")

        assert result["query_type"] == "sql"
        assert result["intent"] == "metric"
        assert result["entities"]["sprint_name"] == "#1 Q1'26"
        # Generic "метрик" word — no specific metric_name to fill in.
        assert "metric_name" not in result["entities"]

    def test_metric_with_rag_marker_stays_rag(self) -> None:
        """Rule 5 must NOT fire when a rag-marker is present.

        A "что такое <metric> в команде <team>" form is a theoretical
        question even with team context — Rule 5 skips it (and Rule 7
        strips the team).
        """
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"intent":"general","query_type":"rag","entities":{"team_name":"cthulhu"}}'
        )
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Что такое scope drop у cthulhu?")

        assert result["query_type"] == "rag"
        assert result["intent"] == "general"
        # Rule 7 cleans up stale entities on rag/general.
        assert result["entities"] == {}

    def test_plural_noun_demotes_task_to_tasks_filter(self) -> None:
        """Rule 6: plural-noun anaphoric follow-up → tasks_filter, not task.

        Reproduces eval Cases 83/86/90/95: anaphoric plural follow-ups
        like "<a> bugs of theirs?" get tagged as intent=task even though
        no issue_key is present. Team name reaches the result through
        carry-forward — that detail isn't what's under test here, but is
        needed to mirror the real scenario.
        """
        mock_llm = MagicMock()
        # LLM returns intent=task (the bug we're fixing). team_name from
        # the prompt context is omitted by the LLM — anaphora carry-forward
        # fills it back from prev_entities.
        mock_llm.invoke.return_value = (
            '{"intent":"task","query_type":"sql","entities":{"issue_type":"Bug"}}'
        )
        agent = SupervisorAgent(mock_llm)

        ctx = _ctx(
            recent=[
                {
                    "role": "user",
                    "content": "Покажи задачи команды lpop",
                    "metadata": {"entities": {"team_name": "lpop"}},
                }
            ]
        )
        result = agent.process("А баги у них?", conversation_context=ctx)

        assert result["intent"] == "tasks_filter"
        assert result["query_type"] == "sql"
        assert result["entities"]["team_name"] == "lpop"
        assert result["entities"]["issue_type"] == "Bug"

    def test_plural_noun_with_issue_key_stays_task(self) -> None:
        """Rule 6 must NOT fire when an issue_key survives sanitization.

        A regex fast-path issue_key short-circuits the slow path entirely,
        so the rule never runs in that scenario — verified here by
        observing that a query like "задачи AL-12345" lands on intent=task
        with the regex-extracted key.
        """
        mock_llm = MagicMock()
        # The mock will not be invoked because the regex fast path triggers.
        mock_llm.invoke.return_value = "{}"
        agent = SupervisorAgent(mock_llm)

        result = agent.process("задачи AL-12345 покажи")

        assert result["intent"] == "task"
        assert result["entities"]["issue_key"] == "AL-12345"
        mock_llm.invoke.assert_not_called()

    def test_rag_general_drops_stale_entities(self) -> None:
        """Rule 7: rag/general carries no entities.

        Reproduces eval Case 92: after "Velocity cthulhu", the next turn
        "Что такое Definition of Done?" gets team_name=cthulhu carried
        forward by the substring-based anaphora detector ("так" matches
        "такое"). Rule 7 cleans that up unconditionally.
        """
        mock_llm = MagicMock()
        # The LLM may correctly route to rag but still emit a team_name
        # because the conversation history mentioned it; OR carry-forward
        # may inject it. Either way, post-processing strips it.
        mock_llm.invoke.return_value = (
            '{"intent":"general","query_type":"rag","entities":{"team_name":"cthulhu"}}'
        )
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Что такое Definition of Done?")

        assert result["query_type"] == "rag"
        assert result["intent"] == "general"
        assert result["entities"] == {}

    def test_simple_general_drops_stale_entities(self) -> None:
        """Rule 7 also fires for query_type=simple (greetings)."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"intent":"general","query_type":"simple","entities":{"team_name":"cthulhu"}}'
        )
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Привет!")

        assert result["query_type"] == "simple"
        assert result["entities"] == {}


# ────────────────────────────────────────────────────────────────
# Supervisor + user_profile (long-term memory)
# ────────────────────────────────────────────────────────────────


class TestSupervisorWithProfile:
    def test_default_team_fills_in_missing_team_entity(self) -> None:
        """default_team in profile → LLM instructed to use it, entities reflect it.

        With a mocked LLM we can't test the model's actual behaviour — but
        we CAN verify (a) the instruction reached the prompt and (b) the
        entity returned by the LLM round-trips through the pipeline.
        """
        mock_llm = MagicMock()
        # Model "obeys" the default_team instruction.
        mock_llm.invoke.return_value = (
            '{"intent":"metric","query_type":"sql",'
            '"entities":{"metric_name":"velocity","team_name":"cthulhu"}}'
        )
        agent = SupervisorAgent(mock_llm)

        profile = {
            "preferences": {"default_team": "cthulhu"},
            "context_summary": "Пользователь — скрам-мастер команды cthulhu.",
        }
        result = agent.process("Покажи velocity", user_profile=profile)

        assert result["entities"].get("team_name") == "cthulhu"
        # The profile block made it into the LLM prompt.
        prompt = mock_llm.invoke.call_args.args[0]
        assert "<user_profile>" in prompt
        assert "cthulhu" in prompt
        assert "Команда пользователя по умолчанию" in prompt
        assert "скрам-мастер" in prompt  # context_summary was rendered too

    def test_no_profile_leaves_prompt_unchanged(self) -> None:
        """Without a profile, no <user_profile> tag should appear."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"intent":"metric","query_type":"sql","entities":{"metric_name":"velocity"}}'
        )
        agent = SupervisorAgent(mock_llm)

        agent.process("Покажи velocity")

        prompt = mock_llm.invoke.call_args.args[0]
        assert "<user_profile>" not in prompt
        assert "Команда пользователя по умолчанию" not in prompt


# ────────────────────────────────────────────────────────────────
# ResponseAgent + conversation_context
# ────────────────────────────────────────────────────────────────


class TestResponseAgentHistory:
    def test_direct_response_prompt_includes_history_and_instruction(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Готов помочь."
        agent = ResponseAgent(mock_llm)

        ctx = _ctx(
            recent=[
                {"role": "user", "content": "Какой velocity у cthulhu?"},
                {"role": "assistant", "content": "Velocity: 42 SP."},
            ]
        )
        state = {
            "query_type": "simple",
            "intent": "general",
            "original_query": "Спасибо!",
            "route": "direct_response",
            "conversation_context": ctx,
        }

        agent.process(state)

        prompt = mock_llm.invoke.call_args.args[0]
        assert "<conversation_history>" in prompt
        assert "Velocity: 42 SP." in prompt
        assert "Не повторяй" in prompt  # anti-duplication instruction

    def test_brief_preference_adds_brevity_instruction(self) -> None:
        """preferences.preferred_detail_level='brief' → prompt carries the nudge."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Velocity: 42."
        agent = ResponseAgent(mock_llm)

        state = {
            "query_type": "simple",
            "intent": "general",
            "original_query": "Какой velocity?",
            "route": "direct_response",
            "conversation_context": None,
            "user_profile": {
                "preferences": {"preferred_detail_level": "brief"},
            },
        }

        agent.process(state)

        prompt = mock_llm.invoke.call_args.args[0]
        assert "Пользователь предпочитает краткие" in prompt

    def test_detailed_preference_adds_no_instruction(self) -> None:
        """detailed is the default tone — no extra line should be injected."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "ok"
        agent = ResponseAgent(mock_llm)

        state = {
            "query_type": "simple",
            "intent": "general",
            "original_query": "Что умеешь?",
            "route": "direct_response",
            "conversation_context": None,
            "user_profile": {
                "preferences": {"preferred_detail_level": "detailed"},
            },
        }

        agent.process(state)

        prompt = mock_llm.invoke.call_args.args[0]
        assert "Пользователь предпочитает краткие" not in prompt

    def test_hybrid_branch_history_fits_within_budget(self) -> None:
        """With a large SQL payload the hybrid budget never underflows the floor."""
        agent = ResponseAgent(MagicMock())

        # Huge payload should push base - data tokens below the floor, but
        # the helper must still return a sane positive budget.
        big_rows = [{"row": "x" * 2000}] * 10
        data_chars = sum(len(str(r)) for r in big_rows[:20])
        budget = agent._get_history_budget("hybrid", "metric", data_chars)

        assert budget == _MIN_HISTORY_BUDGET_TOKENS
        # And it stays ≤ the nominal hybrid base (never above).
        assert budget <= _BRANCH_HISTORY_BUDGET["hybrid"]

        # When context has history that doesn't fit, prefix should be empty
        # rather than blow the budget.
        long_turns = [
            {"role": "user", "content": "x" * 5000},
            {"role": "assistant", "content": "y" * 5000},
        ]
        prefix = agent._history_prefix(_ctx(recent=long_turns), budget_tokens=50)
        # Budget too small for any turn → formatter drops them all → empty.
        assert prefix == "" or estimate_tokens(prefix) <= 50


# ────────────────────────────────────────────────────────────────
# SQL Agent — entities hint injection (the bug behind anaphoric
# follow-ups landing on a wrong team via ORDER BY ... LIMIT 1)
# ────────────────────────────────────────────────────────────────


class TestSQLAgentEntitiesHint:
    """``_format_entities_hint`` renders Supervisor entities for the SQL prompt.

    Without this hint, Qwen3-8B reliably ignores anaphoric carry-forward —
    even when prev_sql clearly filtered by team, the model regenerates
    a global ORDER BY ... LIMIT 1 query.
    """

    def test_team_name_string(self) -> None:
        from hse_prom_prog.agents.sql_agent import _format_entities_hint

        hint = _format_entities_hint({"team_name": "cthulhu"})
        assert "feature_teams ILIKE '%cthulhu%'" in hint
        assert hint.startswith("\n\nИзвлечённые сущности")

    def test_team_name_list(self) -> None:
        from hse_prom_prog.agents.sql_agent import _format_entities_hint

        hint = _format_entities_hint({"team_name": ["cthulhu", "lpop"]})
        assert "'cthulhu'" in hint and "'lpop'" in hint
        assert "ANY/IN или OR" in hint

    def test_metric_points_at_column(self) -> None:
        from hse_prom_prog.agents.sql_agent import _format_entities_hint

        hint = _format_entities_hint({"metric_name": "scope_drop"})
        assert "`scope_drop`" in hint
        assert "report_agile_dashboard_metrics" in hint

    def test_combined_team_and_metric(self) -> None:
        """The exact scenario that fails today: cthulhu + scope_drop."""
        from hse_prom_prog.agents.sql_agent import _format_entities_hint

        hint = _format_entities_hint({"team_name": "cthulhu", "metric_name": "scope_drop"})
        assert "feature_teams ILIKE '%cthulhu%'" in hint
        assert "`scope_drop`" in hint

    def test_enum_fields_use_exact_match(self) -> None:
        from hse_prom_prog.agents.sql_agent import _format_entities_hint

        hint = _format_entities_hint({"issue_type": "Bug", "status": "In Progress"})
        assert "issue_type = 'Bug'" in hint
        assert "issue_status_act = 'In Progress'" in hint

    def test_empty_entities_returns_empty_string(self) -> None:
        from hse_prom_prog.agents.sql_agent import _format_entities_hint

        # No hint → prompt stays byte-identical to pre-memory behaviour.
        assert _format_entities_hint({}) == ""
        assert _format_entities_hint({"team_name": ""}) == ""
        assert _format_entities_hint({"team_name": None}) == ""

    def test_blank_string_dropped(self) -> None:
        """Whitespace-only values must not produce broken filters."""
        from hse_prom_prog.agents.sql_agent import _format_entities_hint

        assert _format_entities_hint({"team_name": "   "}) == ""

    def test_process_passes_entities_to_system_prompt(self) -> None:
        """``SQLAgent.process()`` must inject the hint into the system prompt."""
        from unittest.mock import patch

        from hse_prom_prog.agents.sql_agent import SQLAgent

        agent = SQLAgent.__new__(SQLAgent)  # bypass __init__ (avoid LangGraph build)
        agent.db = MagicMock()
        agent._graph = MagicMock()
        agent._graph.invoke.return_value = {
            "final_sql": "SELECT 1",
            "final_result": [],
            "final_error": "",
        }

        with (
            patch("hse_prom_prog.agents.sql_agent.get_schema_compact", return_value="<schema>"),
            patch("hse_prom_prog.agents.sql_agent.set_db"),
        ):
            agent.process(
                {
                    "original_query": "А у них scope drop?",
                    "entities": {"team_name": "cthulhu", "metric_name": "scope_drop"},
                }
            )

        invoke_args = agent._graph.invoke.call_args.args[0]
        system_msg = invoke_args["messages"][0].content
        assert "feature_teams ILIKE '%cthulhu%'" in system_msg
        assert "`scope_drop`" in system_msg
