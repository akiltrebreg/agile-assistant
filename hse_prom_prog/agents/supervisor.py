"""Supervisor agent for classifying user queries and extracting entities.

This agent classifies the user's intent (task lookup, filtered search,
metric query, or general question) and extracts structured entities
(issue_key, team_name, sprint_name, etc.) using regex + LLM.

It also determines *query_type* for the workflow router:
  sql     -- data lives in PostgreSQL
  rag     -- question about practices, theory, internal docs
  hybrid  -- needs DB data + knowledge-base context
  simple  -- greeting / chitchat, no external data needed
"""

import json
import logging
import re
from typing import Any

from hse_prom_prog.agents.schema_description import (
    SCHEMA_DESCRIPTION,
    SUPERVISOR_FEW_SHOT_EXAMPLES,
)
from hse_prom_prog.llm.client import LLMClient

logger = logging.getLogger(__name__)

_ISSUE_KEY_RE = re.compile(r"\b([A-Z]{2,}-\d+)\b")

# Mapping from (intent) -> default query_type
_INTENT_TO_QUERY_TYPE: dict[str, str] = {
    "task": "sql",
    "tasks_filter": "sql",
    "metric": "sql",
    "general": "simple",
}

# Valid query types
_VALID_QUERY_TYPES = frozenset({"sql", "rag", "hybrid", "simple"})


class SupervisorAgent:
    """Agent that classifies queries and extracts structured entities.

    The Supervisor uses regex for fast issue-key detection and LLM
    for intent classification + entity extraction from natural language.

    Attributes:
        llm_client: LLM client for intent classification.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def _extract_issue_key_regex(self, text: str) -> str | None:
        """Extract Jira issue key using regex (fast path)."""
        match = _ISSUE_KEY_RE.search(text)
        return match.group(1) if match else None

    def _classify_with_llm(self, user_query: str) -> dict[str, Any]:
        """Classify query intent, entities, and query_type via LLM.

        Returns:
            Dict with 'intent', 'entities', and 'query_type' keys.
        """
        prompt = (
            "Ты — классификатор запросов к базе данных Jira и базе знаний.\n\n"
            f"{SCHEMA_DESCRIPTION}\n\n"
            "Определи, что хочет пользователь, и верни ТОЛЬКО JSON "
            "(без пояснений, без markdown):\n"
            "{\n"
            '  "intent": "task" | "tasks_filter" | "metric" | "general",\n'
            '  "query_type": "sql" | "rag" | "hybrid" | "simple",\n'
            '  "entities": {\n'
            '    "issue_key": "ABC-123" или null,\n'
            '    "team_name": "cthulhu" или null,\n'
            '    "sprint_name": "имя спринта" или null,\n'
            '    "metric_name": "done_total" или null,\n'
            '    "issue_type": "Bug" или null,\n'
            '    "status": "In Progress" или null,\n'
            '    "assignee": "имя" или null,\n'
            '    "cluster": "Logistics" или null\n'
            "  }\n"
            "}\n\n"
            "ВАЖНО: если запрос содержит приветствие И вопрос — ИГНОРИРУЙ "
            "приветствие, классифицируй по СУТИ вопроса.\n\n"
            "Правила для intent:\n"
            '- intent="task" — запрос о конкретной задаче по ключу\n'
            '- intent="tasks_filter" — поиск задач по фильтрам '
            "(команда, спринт, тип, статус, кластер)\n"
            '- intent="metric" — запрос метрик команды/спринта '
            "(done_total, scope_drop, velocity и т.д.)\n"
            '- intent="general" — ТОЛЬКО приветствие БЕЗ вопроса, '
            "или вопрос о теории/практиках без привязки к данным БД\n\n"
            "Правила для query_type:\n"
            '- query_type="sql" — нужны данные из БД '
            "(конкретная задача, список задач, метрики)\n"
            '- query_type="rag" — вопрос о теории, практиках, регламентах, '
            "рекомендациях (не о конкретных данных из БД)\n"
            '- query_type="hybrid" — нужны данные из БД И рекомендации/контекст '
            "из базы знаний (например: рассчитай метрику и дай рекомендации)\n"
            '- query_type="simple" — приветствие, общий вопрос без данных\n\n'
            f"{SUPERVISOR_FEW_SHOT_EXAMPLES}\n\n"
            "Дополнительные примеры query_type:\n\n"
            'Запрос: "Привет! Подскажи Done Total в спринте 26Q1.1 Конь не валялся"\n'
            '{"intent": "metric", "query_type": "sql", '
            '"entities": {"sprint_name": "26Q1.1 Конь не валялся", '
            '"metric_name": "done_total"}}\n\n'
            'Запрос: "Как снизить Scope Drop?"\n'
            '{"intent": "general", "query_type": "rag", "entities": {}}\n\n'
            'Запрос: "Что такое Definition of Done?"\n'
            '{"intent": "general", "query_type": "rag", "entities": {}}\n\n'
            'Запрос: "Какие бейзлайновые значения метрик?"\n'
            '{"intent": "general", "query_type": "rag", "entities": {}}\n\n'
            'Запрос: "Покажи scope drop команды cthulhu и дай рекомендации"\n'
            '{"intent": "metric", "query_type": "hybrid", '
            '"entities": {"team_name": "cthulhu", "metric_name": "scope_drop"}}\n\n'
            f'Запрос пользователя: "{user_query}"\n'
            "JSON:"
        )

        llm_response = self.llm_client.invoke(prompt)
        return self._parse_llm_json(llm_response)

    def _parse_llm_json(self, raw: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown fences."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                logger.warning("[Supervisor] Failed to parse LLM JSON: %s", raw[:200])
                return {"intent": "general", "entities": {}, "query_type": "simple"}

        intent = parsed.get("intent", "general")
        if intent not in ("task", "tasks_filter", "metric", "general"):
            intent = "general"

        entities = parsed.get("entities", {})
        entities = {k: v for k, v in entities.items() if v is not None}

        query_type = parsed.get("query_type")
        if query_type not in _VALID_QUERY_TYPES:
            query_type = _INTENT_TO_QUERY_TYPE.get(intent, "simple")

        return {"intent": intent, "entities": entities, "query_type": query_type}

    def process(self, user_query: str) -> dict[str, Any]:
        """Classify the user query and extract structured entities.

        Fast path: if regex finds an issue key and the query is simple
        (just the key or "задача KEY"), skip the LLM call entirely.

        Returns state update with: original_query, intent, entities,
        query_type, route.
        """
        logger.info("[Supervisor] Processing query: %s", user_query)

        # Fast path: regex finds an issue key
        issue_key = self._extract_issue_key_regex(user_query)

        if issue_key:
            logger.info("[Supervisor] Fast path — issue key found: %s", issue_key)
            return {
                "original_query": user_query,
                "intent": "task",
                "entities": {"issue_key": issue_key},
                "query_type": "sql",
                "route": "db_query",
            }

        # Slow path: LLM classification
        logger.info("[Supervisor] No issue key in regex, calling LLM")
        try:
            classification = self._classify_with_llm(user_query)
        except Exception as e:
            logger.error("[Supervisor] LLM classification failed: %s", e)
            return {
                "original_query": user_query,
                "intent": "general",
                "entities": {},
                "query_type": "simple",
                "route": "direct_response",
            }

        intent = classification["intent"]
        entities = classification["entities"]
        query_type = classification["query_type"]

        route = "direct_response" if query_type == "simple" else "db_query"

        logger.info(
            "[Supervisor] intent=%s, query_type=%s, entities=%s, route=%s",
            intent,
            query_type,
            entities,
            route,
        )

        return {
            "original_query": user_query,
            "intent": intent,
            "entities": entities,
            "query_type": query_type,
            "route": route,
        }
