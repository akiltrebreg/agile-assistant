"""Level 1: Input Guardrail — zero-shot NLI topic classification.

Uses ``MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`` (multilingual XNLI, ~280 MB,
CPU). Unlike embedding cosine similarity (baseline similarity ≈0.75 between
any two Russian sentences, tiny on/off-topic gap ~0.02), NLI answers the
direct question «Does this text belong to topic X?» and returns class
probabilities with wide gaps (typically ≥0.7 between on-topic and off-topic).

Three stages:
  1. Hard-deny regex — prompt injection / role hijacking (≥ the classifier)
  2. Fast-path whitelist — issue keys, greetings, meta-questions
  3. Zero-shot NLI classification — two candidate labels (on/off-topic),
     ``on_topic_score`` thresholded via two-zone logic:
       * `score < hard_block_threshold`   → hard BLOCK
       * `hard_block ≤ score < threshold` → PASS, low_confidence=True
       * `score ≥ threshold`              → confident PASS
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from transformers import pipeline

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# Two candidate labels: first is "on-topic", second is "off-topic".
# The NLI model assigns probabilities to each via the hypothesis template
# (see `_HYPOTHESIS_TEMPLATE` below) — we take the score of the on-topic label.
_ON_TOPIC_LABEL = "управление проектами, Jira, Agile, Scrum, спринты, метрики команд"
_OFF_TOPIC_LABEL = "не связано с работой: личное, развлечения, еда, погода, политика"
_HYPOTHESIS_TEMPLATE = "Этот текст о: {}"

# Two-zone thresholds on the NLI on-topic probability (not cosine similarity).
_DEFAULT_THRESHOLD = 0.5
_DEFAULT_HARD_BLOCK_THRESHOLD = 0.2

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


@dataclass
class GuardResult:
    """Result of an input guardrail check.

    Three outcomes:
      * `score < hard_block_threshold`        → passed=False (clearly off-topic)
      * `hard_block ≤ score < threshold`      → passed=True, low_confidence=True
      * `score ≥ threshold`                   → passed=True (confident)

    `max_similarity` holds the NLI on-topic probability (kept name for
    backward compat with logging / monitoring).
    """

    passed: bool
    reason: str = ""
    max_similarity: float = 0.0
    matched_topic: str = ""
    low_confidence: bool = False


def _build_default_classifier() -> Callable[..., dict[str, Any]]:
    """Load the mDeBERTa NLI pipeline on CPU. Called lazily in `TopicGuard`."""
    logger.info("[TopicGuard] Loading zero-shot classifier: %s", _DEFAULT_MODEL)
    return pipeline(
        "zero-shot-classification",
        model=_DEFAULT_MODEL,
        device=-1,  # CPU; GPU is shared by vLLM models already
    )


@dataclass
class TopicGuard:
    """Zero-shot NLI topic filter with two-zone thresholds.

    The classifier produces a probability per candidate label; we use the
    on-topic probability as the decision score. Defaults are tuned for the
    two labels defined in this module; if you replace labels, recalibrate.

    Args:
        classifier: Optional pre-built ``transformers`` zero-shot pipeline.
            If ``None``, loads ``MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`` on
            CPU at initialisation.
        threshold: Confident-pass threshold on on-topic probability.
            Default 0.5.
        hard_block_threshold: Below this probability — clearly off-topic.
            Default 0.2.
        on_topic_label / off_topic_label: Candidate labels for NLI.
        hypothesis_template: NLI hypothesis template.
    """

    classifier: Any = None
    threshold: float = _DEFAULT_THRESHOLD
    hard_block_threshold: float = _DEFAULT_HARD_BLOCK_THRESHOLD
    on_topic_label: str = _ON_TOPIC_LABEL
    off_topic_label: str = _OFF_TOPIC_LABEL
    hypothesis_template: str = _HYPOTHESIS_TEMPLATE
    _candidate_labels: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.hard_block_threshold >= self.threshold:
            msg = (
                f"hard_block_threshold ({self.hard_block_threshold}) must be strictly "
                f"less than threshold ({self.threshold})"
            )
            raise ValueError(msg)
        if self.classifier is None:
            self.classifier = _build_default_classifier()
        self._candidate_labels = [self.on_topic_label, self.off_topic_label]
        logger.info(
            "[TopicGuard] initialized: NLI classifier, hard_block=%.3f, confident=%.3f",
            self.hard_block_threshold,
            self.threshold,
        )

    def _classify(self, query: str) -> float:
        """Return the NLI on-topic probability for *query*."""
        result = self.classifier(
            query,
            candidate_labels=self._candidate_labels,
            hypothesis_template=self.hypothesis_template,
        )
        # pipeline returns labels/scores sorted by score desc — re-index by label
        scores_by_label = dict(zip(result["labels"], result["scores"], strict=True))
        return float(scores_by_label[self.on_topic_label])

    def check(self, query: str) -> GuardResult:
        """Classify a user query using two-zone thresholds on NLI probability.

        Order:
          1. Hard-deny regex → block with reason='blocked:prompt_injection'
          2. Whitelist regex → pass with reason='whitelist'
          3. Zero-shot NLI classification → two-zone gate
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

        on_score = self._classify(query)

        if on_score < self.hard_block_threshold:
            logger.info(
                "[TopicGuard] BLOCKED (P_on=%.3f < hard_block=%.3f): %r",
                on_score,
                self.hard_block_threshold,
                query[:100],
            )
            return GuardResult(
                passed=False,
                reason="off_topic",
                max_similarity=on_score,
                matched_topic=self.on_topic_label,
            )

        if on_score < self.threshold:
            logger.info(
                "[TopicGuard] BORDERLINE (P_on=%.3f in [%.3f, %.3f)): %r",
                on_score,
                self.hard_block_threshold,
                self.threshold,
                query[:100],
            )
            return GuardResult(
                passed=True,
                reason="borderline",
                max_similarity=on_score,
                matched_topic=self.on_topic_label,
                low_confidence=True,
            )

        logger.debug(
            "[TopicGuard] PASSED (P_on=%.3f): %r",
            on_score,
            query[:80],
        )
        return GuardResult(
            passed=True,
            reason="on_topic",
            max_similarity=on_score,
            matched_topic=self.on_topic_label,
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
