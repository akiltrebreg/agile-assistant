"""Calibrate TopicGuard thresholds on labelled on-/off-topic examples.

Scores queries via zero-shot NLI classifier (same model used by TopicGuard)
and sweeps the confident-pass threshold. Also suggests a safe hard-block
threshold (highest value that does not block any on-topic example).

Usage:
    docker compose run --rm --no-deps app python -m eval.calibrate_topic_guard
"""

from __future__ import annotations

import logging

from tabulate import tabulate

from hse_prom_prog.agents.guardrails.topic_guard import (
    _HARD_DENY_PATTERNS,
    TopicGuard,
)

logger = logging.getLogger(__name__)


ON_TOPIC: list[str] = [
    "Расскажи о задаче AL-38787",
    "Velocity команды cthulhu",
    "Что такое Scope Drop?",
    "Задачи со статусом In Progress",
    "Как считается метрика Done Total?",
    "Метрики команды lpop по спринту 26Q1.1",
    "Баги в кластере Logistics",
    "Покажи scope drop и дай рекомендации",
    "Definition of Done",
    "Sprint goal команды linehaul",
    "Закрытые задачи команды shopping cart",
    "Что такое груминг бэклога?",
    "Кто исполнитель задачи AL-12345?",
    "Как снизить cancel rate в команде?",
    "Покажи все эпики проекта",
    "Как улучшить работу команды?",
    "Что делать, если спринт провален?",
    "Как мотивировать разработчиков?",
    "Советы по планированию спринта",
]

OFF_TOPIC: list[str] = [
    "Расскажи анекдот про программиста",
    "Какая погода сегодня в Москве?",
    "Напиши мне стихотворение о любви",
    "Как приготовить борщ?",
    "Кто победил на выборах президента?",
    "Помоги мне с домашним заданием по физике",
    "Расскажи о своей личной жизни",
    "Сгенерируй код на Python для сортировки",
    "Какой курс доллара сегодня?",
    "Посоветуй фильм на вечер",
    "Как научиться играть на гитаре?",
    "Расскажи про историю Древнего Рима",
    "Переведи эту фразу на английский",
    "Какие акции лучше купить?",
    "Опиши симптомы гриппа",
]


def _is_hard_deny(query: str) -> bool:
    return any(p.search(query) for p in _HARD_DENY_PATTERNS)


def _score(guard: TopicGuard, examples: list[str]) -> list[float]:
    """Return raw NLI on-topic probability (bypass whitelist / hard-deny)."""
    scores = []
    for q in examples:
        if _is_hard_deny(q):
            scores.append(0.0)
            continue
        scores.append(guard._classify(q))
    return scores


def _metrics(on_scores: list[float], off_scores: list[float], threshold: float) -> dict[str, float]:
    tp = sum(1 for s in on_scores if s >= threshold)
    fn = len(on_scores) - tp
    fp = sum(1 for s in off_scores if s >= threshold)
    tn = len(off_scores) - fp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {
        "threshold": threshold,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    logger.info("Loading zero-shot NLI classifier...")
    guard = TopicGuard()

    logger.info("Scoring on-topic examples (%d)...", len(ON_TOPIC))
    on_scores = _score(guard, ON_TOPIC)
    logger.info("Scoring off-topic examples (%d)...", len(OFF_TOPIC))
    off_scores = _score(guard, OFF_TOPIC)

    print("\n── Per-example NLI on-topic probability ──")
    rows = [["ON", q[:60], f"{s:.3f}"] for q, s in zip(ON_TOPIC, on_scores, strict=True)]
    rows += [["OFF", q[:60], f"{s:.3f}"] for q, s in zip(OFF_TOPIC, off_scores, strict=True)]
    print(tabulate(rows, headers=["Label", "Query", "P_on"], tablefmt="simple"))

    print("\n── Threshold sweep (confident-pass) ──")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    table = [_metrics(on_scores, off_scores, t) for t in thresholds]
    print(
        tabulate(
            [
                [
                    f"{r['threshold']:.3f}",
                    r["tp"],
                    r["fn"],
                    r["fp"],
                    r["tn"],
                    f"{r['precision']:.3f}",
                    f"{r['recall']:.3f}",
                    f"{r['f1']:.3f}",
                    f"{r['accuracy']:.3f}",
                ]
                for r in table
            ],
            headers=["thr", "TP", "FN", "FP", "TN", "P", "R", "F1", "acc"],
            tablefmt="simple",
        )
    )

    best = max(table, key=lambda r: r["f1"])
    print(
        f"\nBest by F1: threshold={best['threshold']:.3f}, "
        f"F1={best['f1']:.3f}, P={best['precision']:.3f}, R={best['recall']:.3f}"
    )

    # Two-zone recommendation: T_hard ≤ min(ON) keeps recall at 1.0 on block.
    max_safe_hard = min(on_scores)
    print("\n── Two-zone mode suggestion ──")
    print(f"Confident-pass threshold (Best F1):  {best['threshold']:.3f}")
    print(f"Hard-block threshold (safe max):     ≤ {max_safe_hard:.3f}")
    hard_off_blocked = sum(1 for s in off_scores if s < max_safe_hard)
    hard_off_passed = len(off_scores) - hard_off_blocked
    print(
        f"At T_hard = {max_safe_hard:.3f}: off-topic hard-blocked = "
        f"{hard_off_blocked}/{len(off_scores)}, "
        f"routed to Supervisor (borderline) = {hard_off_passed}/{len(off_scores)}"
    )


if __name__ == "__main__":
    main()
