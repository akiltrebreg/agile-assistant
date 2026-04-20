"""Level 1: Input Guardrail — off-topic filter via embedding cosine similarity.

Reuses the RAG embedding model (`multilingual-e5-base`, loaded via
`langchain_huggingface.HuggingFaceEmbeddings`). Reference topics are
pre-embedded once at startup; each user query gets one `embed_query` call
(~2ms on CPU) and a dot product against the reference matrix.

Three stages:
  1. Hard deny — regex for prompt injection / role hijacking
  2. Fast-path whitelist — issue key, greetings, meta-questions
  3. Embedding cosine similarity — query vs reference topics, threshold gate
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# E5-family models expect "query: ..." / "passage: ..." prefixes
_REFERENCE_TOPICS: list[str] = [
    "passage: задача в Jira, issue key, тикет, баг, story, эпик",
    "passage: статус задачи, In Progress, Done, Open, Closed, Resolved",
    "passage: исполнитель задачи, assignee, reporter, ответственный",
    "passage: story points, оценка задачи, estimate, трудозатраты",
    "passage: спринт, sprint, итерация, планирование спринта",
    "passage: команда разработки, feature team, Scrum-команда",
    "passage: velocity, done total, scope drop, cancel rate, lead time",
    "passage: метрики команды, agile-метрики, KPI, дашборд",
    "passage: groomed backlog, бэклог, backlog dynamics",
    "passage: Scrum, Kanban, Agile, ретроспектива, daily standup",
    "passage: Definition of Done, Definition of Ready, acceptance criteria",
    "passage: sprint goal, sprint review, sprint planning",
    "passage: привет, здравствуйте, что ты умеешь, помощь, help",
]

_ALWAYS_PASS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"[A-Z]{2,10}-\d+", re.IGNORECASE),
    re.compile(r"^\s*(привет|здравствуй|добрый|hi|hello)", re.IGNORECASE),
    re.compile(r"(что (ты )?(умеешь|можешь)|\bhelp\b|помо[гщ])", re.IGNORECASE),
]

_HARD_DENY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(ignore|forget|disregard).{0,30}(instruction|prompt|system|role)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(ты теперь|act as|you are now|pretend).{0,30}"
        r"(не ассистент|not an assistant|другая роль|different role)",
        re.IGNORECASE,
    ),
]

_DEFAULT_THRESHOLD = 0.45


@dataclass
class GuardResult:
    """Result of an input guardrail check."""

    passed: bool
    reason: str = ""
    max_similarity: float = 0.0
    matched_topic: str = ""


@dataclass
class TopicGuard:
    """Embedding-based topic filter.

    Shares the embedding model with the RAG pipeline — no extra memory cost.
    Reference topic embeddings are pre-computed once in `__post_init__`.

    Args:
        embeddings: HuggingFaceEmbeddings (same instance used by RAG retriever).
        threshold: Minimum cosine similarity to pass. Default 0.45.
        reference_topics: Anchor phrases (each prefixed with "passage: ").
    """

    embeddings: HuggingFaceEmbeddings
    threshold: float = _DEFAULT_THRESHOLD
    reference_topics: list[str] = field(default_factory=lambda: list(_REFERENCE_TOPICS))
    _ref_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        vectors = self.embeddings.embed_documents(self.reference_topics)
        self._ref_matrix = np.asarray(vectors, dtype=np.float32)
        logger.info(
            "[TopicGuard] initialized: %d reference topics, threshold=%.2f",
            len(self.reference_topics),
            self.threshold,
        )

    def check(self, query: str) -> GuardResult:
        """Classify a user query as on-topic / off-topic.

        Order:
          1. Hard-deny regex → block with reason='blocked:prompt_injection'
          2. Whitelist regex → pass with reason='whitelist'
          3. Embedding cosine vs reference anchors → threshold gate
        """
        for pattern in _HARD_DENY_PATTERNS:
            if pattern.search(query):
                logger.warning("[TopicGuard] HARD DENY: %r", query[:100])
                return GuardResult(
                    passed=False,
                    reason="blocked:prompt_injection",
                    max_similarity=0.0,
                )

        for pattern in _ALWAYS_PASS_PATTERNS:
            if pattern.search(query):
                return GuardResult(passed=True, reason="whitelist", max_similarity=1.0)

        query_vec = np.asarray(
            self.embeddings.embed_query(f"query: {query}"),
            dtype=np.float32,
        )
        similarities = self._ref_matrix @ query_vec
        max_idx = int(np.argmax(similarities))
        max_sim = float(similarities[max_idx])

        if max_sim < self.threshold:
            logger.info(
                "[TopicGuard] BLOCKED (sim=%.3f < %.3f): %r",
                max_sim,
                self.threshold,
                query[:100],
            )
            return GuardResult(
                passed=False,
                reason="off_topic",
                max_similarity=max_sim,
                matched_topic=self.reference_topics[max_idx],
            )

        logger.debug(
            "[TopicGuard] PASSED (sim=%.3f): %r → %r",
            max_sim,
            query[:80],
            self.reference_topics[max_idx][:60],
        )
        return GuardResult(
            passed=True,
            reason="on_topic",
            max_similarity=max_sim,
            matched_topic=self.reference_topics[max_idx],
        )


OFF_TOPIC_RESPONSE = (
    "К сожалению, этот вопрос выходит за рамки моих компетенций. "
    "Я — ассистент для анализа Jira-задач и Agile-метрик.\n\n"
    "Я могу помочь с:\n"
    "• Поиском информации по задачам (например, «Расскажи о задаче AL-38787»)\n"
    "• Метриками команд и спринтов (например, «Velocity команды lpop»)\n"
    "• Вопросами об Agile-практиках (например, «Как снизить Scope Drop?»)\n\n"
    "Попробуйте задать вопрос по одной из этих тем!"
)
