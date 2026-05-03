"""Load test for Agile Assistant API.

Simulation model:
- 1 Locust user = 1 chat session with stable user_id and conversation_id
- each turn: POST /api/tasks -> polling GET /api/tasks/{id} until COMPLETED/FAILED
- pause 10-20 s between turns (mimics a human reading the reply)

The query mix matches a realistic chat pattern: ~50% small-talk,
~25% theory (RAG), ~20% facts (SQL), ~5% combined (hybrid). Weights are
set via `@task(N)` -- Locust picks each task proportionally to N.

Run (web UI):
    locust -f loadtest/locustfile.py --host http://195.209.218.21

Run (headless via wrapper):
    ./loadtest/run.sh capacity-3u --users 3 --spawn-rate 1 --run-time 10m
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from http import HTTPStatus

from locust import HttpUser, between, events, task

logger = logging.getLogger(__name__)

# Query pools per workflow branch. Larger pools reduce the chance of
# getting stuck on one bad query that always trips the SQL retry loop.

SIMPLE_QUERIES = [
    "Привет!",
    "Спасибо за помощь",
    "Как дела?",
    "Что ты умеешь?",
    "Помоги, пожалуйста",
    "Доброе утро",
    "Понятно, спасибо",
]

RAG_QUERIES = [
    "Что такое story point?",
    "Какие есть Scrum-церемонии?",
    "Чем Kanban отличается от Scrum?",
    "Что такое velocity команды?",
    "Опиши роль скрам-мастера",
    "Что такое definition of done?",
    "Зачем нужен retrospective?",
    "Что делает product owner?",
]

SQL_QUERIES = [
    "Сколько задач в проекте AL?",
    "Покажи незакрытые задачи",
    "Какие команды есть в проекте?",
    "Сколько спринтов завершено?",
    "Покажи задачи с приоритетом High",
]

HYBRID_QUERIES = [
    "Как оценить story points для задачи?",
    "Объясни, что такое velocity, и покажи её для команды Cthulhu",
]

POLL_INTERVAL_S = 1.0  # delay between /tasks/{id} polls
TASK_TIMEOUT_S = 120.0  # how long to wait for COMPLETED before declaring a timeout


class ChatUser(HttpUser):
    """A single simulated chat session."""

    # Delay between turns from the same user -- typical time to read the
    # reply and formulate the next question. 10-20 s is more realistic
    # than the previous 5-15 s, which was too aggressive.
    wait_time = between(10, 20)

    def on_start(self) -> None:
        self.user_id = f"loadtest-{uuid.uuid4()}"
        self.conversation_id: str | None = None

    # @task(N) weights set the share of each branch. The sum is 100 so
    # the numbers read directly as percentages. Real production traffic
    # may differ -- recheck against logs if needed.

    @task(50)
    def simple_turn(self) -> None:
        self._chat(random.choice(SIMPLE_QUERIES), label="simple")

    @task(25)
    def rag_turn(self) -> None:
        self._chat(random.choice(RAG_QUERIES), label="rag")

    @task(20)
    def sql_turn(self) -> None:
        self._chat(random.choice(SQL_QUERIES), label="sql")

    @task(5)
    def hybrid_turn(self) -> None:
        self._chat(random.choice(HYBRID_QUERIES), label="hybrid")

    def _chat(self, query: str, *, label: str) -> None:
        payload = {
            "query": query,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
        }

        with self.client.post(
            "/api/tasks",
            json=payload,
            name="POST /api/tasks",
            catch_response=True,
        ) as resp:
            if resp.status_code != HTTPStatus.ACCEPTED:
                resp.failure(f"create -> {resp.status_code}: {resp.text[:200]}")
                return
            body = resp.json()
            task_id = body["task_id"]
            self.conversation_id = body["conversation_id"]

        e2e_start = time.monotonic()
        deadline = e2e_start + TASK_TIMEOUT_S
        terminal: str | None = None

        while time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL_S)
            with self.client.get(
                f"/api/tasks/{task_id}",
                name="GET /api/tasks/[id]",
                catch_response=True,
            ) as resp:
                if resp.status_code != HTTPStatus.OK:
                    resp.failure(f"poll -> {resp.status_code}")
                    return
                status = resp.json().get("status")
                if status in ("COMPLETED", "FAILED"):
                    terminal = status
                    break

        elapsed_ms = int((time.monotonic() - e2e_start) * 1000)

        # Per-branch metrics via `name=` -- Locust aggregates them
        # separately, so the CSV gets rows like `E2E_simple`, `E2E_rag`,
        # `E2E_sql`, `E2E_hybrid`. That gives per-branch p95 without
        # having to dig into Grafana.
        per_branch_name = f"chat_turn_{label}"

        if terminal == "COMPLETED":
            events.request.fire(
                request_type="E2E",
                name="chat_turn",
                response_time=elapsed_ms,
                response_length=0,
                exception=None,
                context={},
            )
            events.request.fire(
                request_type="E2E",
                name=per_branch_name,
                response_time=elapsed_ms,
                response_length=0,
                exception=None,
                context={},
            )
        elif terminal == "FAILED":
            for nm in ("chat_turn", per_branch_name):
                events.request.fire(
                    request_type="E2E",
                    name=nm,
                    response_time=elapsed_ms,
                    response_length=0,
                    exception=Exception("task FAILED"),
                    context={},
                )
        else:
            for nm in ("chat_turn", per_branch_name):
                events.request.fire(
                    request_type="E2E",
                    name=nm,
                    response_time=int(TASK_TIMEOUT_S * 1000),
                    response_length=0,
                    exception=Exception(f"timeout after {TASK_TIMEOUT_S}s"),
                    context={},
                )
