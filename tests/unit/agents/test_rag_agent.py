"""Unit tests for RAG Agent + cross-encoder reranker.

Coverage strategy:

  * ``RAGAgent.process`` — mocked retriever + LLM, reranker disabled via
    monkeypatch so we don't load the real CrossEncoder model.
  * ``RAGAgent._build_context`` — static method, exercised directly.
  * ``Reranker.rerank`` — CrossEncoder mocked to return canned scores so
    we can assert threshold filtering, top-N cut, and Lost-in-the-Middle
    reordering deterministically.

We don't drive the live retriever (Qdrant) here — that's an integration
concern. The retriever protocol is just ``.invoke(query) → list[Document]``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from hse_prom_prog.agents import rag_agent
from hse_prom_prog.agents.rag_agent import RAGAgent
from hse_prom_prog.rag.reranker import Reranker

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _doc(text: str, source: str = "doc.pdf", category: str = "") -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "category": category},
    )


@pytest.fixture
def disabled_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make RAGAgent.__init__ skip the heavy CrossEncoder load."""
    monkeypatch.setattr(rag_agent.settings, "reranker_enabled", False)


@pytest.fixture
def retriever() -> MagicMock:
    r = MagicMock()
    r.invoke = MagicMock(return_value=[])
    return r


@pytest.fixture
def agent(
    disabled_reranker: None,
    mock_llm_client: MagicMock,
    retriever: MagicMock,
) -> RAGAgent:
    return RAGAgent(mock_llm_client, retriever)


# ===================================================================== #
# RAGAgent.process — orchestration
# ===================================================================== #


@pytest.mark.unit
class TestProcess:
    def test_happy_path_invokes_llm_with_context(
        self,
        agent: RAGAgent,
        retriever: MagicMock,
        mock_llm_client: MagicMock,
    ) -> None:
        retriever.invoke.return_value = [
            _doc("Velocity — это метрика командной скорости.", source="agile-guide.pdf"),
            _doc("Sprint Goal — цель спринта.", source="scrum-book.pdf"),
        ]
        mock_llm_client.invoke.return_value = "Ответ на основе контекста."

        out = agent.process({"original_query": "Что такое velocity?"})

        assert out["rag_response"] == "Ответ на основе контекста."
        assert "agile-guide.pdf" in out["rag_sources"]
        assert "scrum-book.pdf" in out["rag_sources"]
        # Prompt body must contain both the user question and the context.
        prompt = mock_llm_client.invoke.call_args.args[0]
        assert "Что такое velocity?" in prompt
        assert "Velocity — это метрика" in prompt

    def test_empty_retriever_returns_empty_without_llm(
        self,
        agent: RAGAgent,
        retriever: MagicMock,
        mock_llm_client: MagicMock,
    ) -> None:
        retriever.invoke.return_value = []
        out = agent.process({"original_query": "anything"})
        assert out == {"rag_response": None, "rag_sources": []}
        # No-docs short-circuit — must not waste an LLM call.
        mock_llm_client.invoke.assert_not_called()

    def test_retriever_exception_degrades_gracefully(
        self,
        agent: RAGAgent,
        retriever: MagicMock,
        mock_llm_client: MagicMock,
    ) -> None:
        # Qdrant unreachable — the agent must NOT crash the workflow;
        # the validator downstream sees an empty rag_response and routes
        # to the off-topic message instead.
        retriever.invoke.side_effect = ConnectionError("qdrant down")
        out = agent.process({"original_query": "anything"})
        assert out == {"rag_response": None, "rag_sources": []}
        mock_llm_client.invoke.assert_not_called()

    def test_llm_exception_preserves_sources_and_sets_error(
        self,
        agent: RAGAgent,
        retriever: MagicMock,
        mock_llm_client: MagicMock,
    ) -> None:
        retriever.invoke.return_value = [_doc("ctx", source="doc.pdf")]
        mock_llm_client.invoke.side_effect = TimeoutError("vllm timed out")

        out = agent.process({"original_query": "q"})
        assert out["rag_response"] is None
        assert out["rag_sources"] == ["doc.pdf"]
        assert "timed out" in out["error"]

    def test_source_labels_use_category_prefix(
        self,
        agent: RAGAgent,
        retriever: MagicMock,
        mock_llm_client: MagicMock,
    ) -> None:
        retriever.invoke.return_value = [
            _doc("a", source="agile-guide.pdf", category="practices"),
            _doc("b", source="scrum.pdf"),
        ]
        mock_llm_client.invoke.return_value = "ok"
        out = agent.process({"original_query": "q"})
        # Category-prefixed when present, raw source otherwise.
        assert "practices/agile-guide.pdf" in out["rag_sources"]
        assert "scrum.pdf" in out["rag_sources"]


# ===================================================================== #
# _build_context — context-budget truncation
# ===================================================================== #


@pytest.mark.unit
class TestBuildContext:
    def test_short_docs_all_included(self) -> None:
        docs = [_doc("short A", source="a.pdf"), _doc("short B", source="b.pdf")]
        ctx, sources = RAGAgent._build_context(docs)
        assert "short A" in ctx
        assert "short B" in ctx
        assert sources == ["a.pdf", "b.pdf"]

    def test_long_doc_gets_truncated_to_remaining_budget(self) -> None:
        # First doc fills budget; the second is sliced to whatever space
        # is left so the total stays at most _MAX_CONTEXT_CHARS.
        max_chars = rag_agent._MAX_CONTEXT_CHARS
        first = _doc("A" * (max_chars - 100), source="a.pdf")
        second = _doc("B" * 500, source="b.pdf")
        ctx, _ = RAGAgent._build_context([first, second])
        assert len(ctx.replace("\n\n", "")) <= max_chars

    def test_overflow_doc_dropped_when_budget_exhausted(self) -> None:
        # First doc consumes the entire budget; the second contributes
        # nothing (cannot include even a slice).
        max_chars = rag_agent._MAX_CONTEXT_CHARS
        first = _doc("A" * max_chars, source="a.pdf")
        second = _doc("B" * 100, source="b.pdf")
        ctx, _ = RAGAgent._build_context([first, second])
        assert "A" in ctx
        # Second doc never made it into the context.
        assert "B" not in ctx

    def test_duplicate_sources_deduplicated(self) -> None:
        # Two chunks from the same file → source listed once.
        docs = [_doc("part 1", source="same.pdf"), _doc("part 2", source="same.pdf")]
        _, sources = RAGAgent._build_context(docs)
        assert sources == ["same.pdf"]


# ===================================================================== #
# Reranker — predict scores, threshold, top-N, reorder
# ===================================================================== #


def _build_reranker(scores: list[float], *, threshold: float = 0.0, top_n: int = 4) -> Reranker:
    """Construct a Reranker with the heavy CrossEncoder model swapped out."""
    fake_model = MagicMock()
    fake_model.predict = MagicMock(return_value=scores)
    with patch("hse_prom_prog.rag.reranker.CrossEncoder", return_value=fake_model):
        return Reranker("fake-model", threshold=threshold, top_n=top_n)


@pytest.mark.unit
class TestReranker:
    def test_empty_input_returns_empty(self) -> None:
        reranker = _build_reranker([])
        assert reranker.rerank("q", []) == []

    def test_all_below_threshold_returns_empty(self) -> None:
        # Scores 0.05 and 0.08 — both below the 0.5 threshold.
        reranker = _build_reranker([0.05, 0.08], threshold=0.5)
        docs = [_doc(f"chunk {i}") for i in range(2)]
        assert reranker.rerank("q", docs) == []

    def test_filters_below_threshold_keeps_above(self) -> None:
        reranker = _build_reranker([0.9, 0.1, 0.7], threshold=0.5, top_n=10)
        docs = [_doc(f"chunk {i}") for i in range(3)]
        kept = reranker.rerank("q", docs)
        kept_texts = [d.page_content for d in kept]
        # Doc 1 (score 0.1) is dropped; docs 0 and 2 survive.
        assert "chunk 0" in kept_texts
        assert "chunk 2" in kept_texts
        assert "chunk 1" not in kept_texts

    def test_top_n_cut_after_sort(self) -> None:
        # 5 docs, top_n=2 — only the two highest-scored survive.
        reranker = _build_reranker([0.1, 0.9, 0.5, 0.8, 0.3], threshold=0.0, top_n=2)
        docs = [_doc(f"chunk {i}") for i in range(5)]
        kept = reranker.rerank("q", docs)
        kept_texts = {d.page_content for d in kept}
        assert kept_texts == {"chunk 1", "chunk 3"}

    def test_no_reorder_for_two_or_fewer_docs(self) -> None:
        # ≤2 docs → identity order after sort. Lost-in-the-middle is a
        # no-op below 3 docs.
        reranker = _build_reranker([0.9, 0.5], threshold=0.0, top_n=10)
        docs = [_doc("first"), _doc("second")]
        kept = reranker.rerank("q", docs)
        assert [d.page_content for d in kept] == ["first", "second"]


# ===================================================================== #
# Lost-in-the-Middle — pure reorder logic
# ===================================================================== #


@pytest.mark.unit
class TestLostInTheMiddle:
    def test_five_docs_reordered_to_edges(self) -> None:
        # Input sorted desc by score: [best, 2nd, 3rd, 4th, 5th].
        # Expected output: [best, 3rd, 5th, 4th, 2nd] — best at the edges,
        # weakest in the middle.
        scored = [
            (0.9, _doc("best")),
            (0.8, _doc("2nd")),
            (0.7, _doc("3rd")),
            (0.6, _doc("4th")),
            (0.5, _doc("5th")),
        ]
        out = Reranker._lost_in_the_middle_reorder(scored)
        assert [d.page_content for d in out] == ["best", "3rd", "5th", "4th", "2nd"]

    def test_three_docs_reorder(self) -> None:
        # [a, b, c] → a goes left, b goes right, c goes left.
        # Result: [a, c, b].
        scored = [
            (0.9, _doc("a")),
            (0.5, _doc("b")),
            (0.1, _doc("c")),
        ]
        out = Reranker._lost_in_the_middle_reorder(scored)
        assert [d.page_content for d in out] == ["a", "c", "b"]

    def test_single_doc_passes_through(self) -> None:
        scored = [(0.9, _doc("only"))]
        out = Reranker._lost_in_the_middle_reorder(scored)
        assert len(out) == 1
        assert out[0].page_content == "only"

    def test_two_docs_pass_through(self) -> None:
        # Boundary: _MIN_DOCS_FOR_REORDER=2 → exactly 2 means no reorder.
        scored = [(0.9, _doc("first")), (0.5, _doc("second"))]
        out = Reranker._lost_in_the_middle_reorder(scored)
        assert [d.page_content for d in out] == ["first", "second"]


# ===================================================================== #
# Reranker integration with RAGAgent
# ===================================================================== #


@pytest.mark.unit
class TestRAGAgentReranker:
    def test_reranker_invoked_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_llm_client: MagicMock,
        retriever: MagicMock,
    ) -> None:
        # Real path: reranker_enabled=True → get_reranker returns our mock,
        # the agent must call .rerank with the retrieved docs.
        monkeypatch.setattr(rag_agent.settings, "reranker_enabled", True)
        fake_reranker = MagicMock()
        kept_docs = [_doc("kept", source="a.pdf")]
        fake_reranker.rerank = MagicMock(return_value=kept_docs)
        monkeypatch.setattr(rag_agent, "get_reranker", lambda: fake_reranker)

        agent = RAGAgent(mock_llm_client, retriever)
        retriever.invoke.return_value = [
            _doc("a", source="a.pdf"),
            _doc("b", source="b.pdf"),
        ]
        mock_llm_client.invoke.return_value = "answer"

        out = agent.process({"original_query": "q"})
        fake_reranker.rerank.assert_called_once()
        # Sources reflect the reranker's surviving docs, not the raw retrieval.
        assert out["rag_sources"] == ["a.pdf"]

    def test_reranker_drops_all_docs_short_circuits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_llm_client: MagicMock,
        retriever: MagicMock,
    ) -> None:
        # Reranker rejects every candidate. _retrieve_and_rerank returns []
        # downstream of the reranker, which trips the same "no docs" guard
        # in process() — the agent must return the empty payload and NOT
        # waste an LLM call grounding on nothing.
        monkeypatch.setattr(rag_agent.settings, "reranker_enabled", True)
        fake_reranker = MagicMock()
        fake_reranker.rerank = MagicMock(return_value=[])
        monkeypatch.setattr(rag_agent, "get_reranker", lambda: fake_reranker)

        agent = RAGAgent(mock_llm_client, retriever)
        retriever.invoke.return_value = [_doc("low-relevance")]

        out = agent.process({"original_query": "q"})
        assert out == {"rag_response": None, "rag_sources": []}
        mock_llm_client.invoke.assert_not_called()
