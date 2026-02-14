"""Supervisor agent for classifying user queries and extracting entities.

This agent classifies the user's intent (task lookup, filtered search,
metric query, or general question) and extracts structured entities
(issue_key, team_name, sprint_name, etc.) using regex + LLM.
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
        """Classify query intent and extract entities via LLM.

        Returns:
            Dict with 'intent' and 'entities' keys.
        """
        prompt = (
            "Ты — классификатор запросов к базе данных Jira.\n\n"
            f"{SCHEMA_DESCRIPTION}\n\n"
            "Определи, что хочет пользователь, и верни ТОЛЬКО JSON "
            "(без пояснений, без markdown):\n"
            "{\n"
            '  "intent": "task" | "tasks_filter" | "metric" | "general",\n'
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
            "Правила:\n"
            '- intent="task" — запрос о конкретной задаче по ключу (например AL-38787)\n'
            '- intent="tasks_filter" — поиск задач по фильтрам '
            "(команда, спринт, тип, статус, кластер)\n"
            '- intent="metric" — запрос метрик команды/спринта '
            "(done_total, scope_drop, velocity и т.д.)\n"
            '- intent="general" — общий вопрос, не требующий данных из БД\n\n'
            f"{SUPERVISOR_FEW_SHOT_EXAMPLES}\n\n"
            f'Запрос пользователя: "{user_query}"\n'
            "JSON:"
        )

        llm_response = self.llm_client.invoke(prompt)
        return self._parse_llm_json(llm_response)

    def _parse_llm_json(self, raw: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown fences."""
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (with optional language tag)
            cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract first JSON object from response
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                logger.warning(f"[Supervisor] Failed to parse LLM JSON: {raw[:200]}")
                return {"intent": "general", "entities": {}}

        intent = parsed.get("intent", "general")
        if intent not in ("task", "tasks_filter", "metric", "general"):
            intent = "general"

        entities = parsed.get("entities", {})
        # Clean null-valued entities
        entities = {k: v for k, v in entities.items() if v is not None}

        return {"intent": intent, "entities": entities}

    def process(self, user_query: str) -> dict[str, Any]:
        """Classify the user query and extract structured entities.

        Fast path: if regex finds an issue key and the query is simple
        (just the key or "задача KEY"), skip the LLM call entirely.

        Returns state update with: original_query, intent, entities, route.
        """
        logger.info(f"[Supervisor] Processing query: {user_query}")

        # Fast path: regex finds an issue key
        issue_key = self._extract_issue_key_regex(user_query)

        if issue_key:
            logger.info(f"[Supervisor] Fast path — issue key found: {issue_key}")
            return {
                "original_query": user_query,
                "intent": "task",
                "entities": {"issue_key": issue_key},
                "route": "db_query",
            }

        # Slow path: LLM classification
        logger.info("[Supervisor] No issue key in regex, calling LLM for classification")
        try:
            classification = self._classify_with_llm(user_query)
        except Exception as e:
            logger.error(f"[Supervisor] LLM classification failed: {e}")
            # Fallback: treat as general question
            return {
                "original_query": user_query,
                "intent": "general",
                "entities": {},
                "route": "direct_response",
            }

        intent = classification["intent"]
        entities = classification["entities"]

        route = "direct_response" if intent == "general" else "db_query"

        logger.info(f"[Supervisor] Intent: {intent}, entities: {entities}, route: {route}")

        return {
            "original_query": user_query,
            "intent": intent,
            "entities": entities,
            "route": route,
        }
