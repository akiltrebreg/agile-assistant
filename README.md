# Agile AI Assistant

[![Tests](https://github.com/akiltrebreg/hse-prom-prog/actions/workflows/test.yml/badge.svg)](https://github.com/akiltrebreg/hse-prom-prog/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/akiltrebreg/hse-prom-prog/branch/main/graph/badge.svg)](https://codecov.io/gh/akiltrebreg/hse-prom-prog)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)

Multi-agent система для анализа Jira-задач с использованием LangGraph, vLLM,
PostgreSQL, Qdrant, Celery, Redis, nginx, k8s.

## Содержание

- [Архитектура](#архитектура)
  - [Компоненты](#компоненты)
  - [LLM Backend](#llm-backend)
  - [База данных](#база-данных)
  - [Векторное хранилище (Qdrant)](#векторное-хранилище-qdrant)
- [Memory Layer](#memory-layer)
  - [Контекст диалога](#контекст-диалога)
  - [Профиль пользователя](#профиль-пользователя)
- [Структура проекта](#структура-проекта)
- [Быстрый старт с Docker Compose](#быстрый-старт-с-docker-compose)
  - [Настройка переменных окружения](#настройка-переменных-окружения)
- [База знаний и RAG](#база-знаний-и-rag)
  - [Структура knowledge_base/](#структура-knowledge_base)
  - [Загрузка документов в Qdrant](#загрузка-документов-в-qdrant)
- [Локальная разработка](#локальная-разработка)
  - [Требования](#требования)
  - [Шаг 1: Установка зависимостей](#шаг-1-установка-зависимостей)
  - [Шаг 2: Настройка PostgreSQL](#шаг-2-настройка-postgresql)
  - [Шаг 3: Запуск vLLM](#шаг-3-запуск-vllm)
  - [Шаг 4: Запуск приложения](#шаг-4-запуск-приложения)
- [Использование](#использование)
  - [Примеры запросов](#примеры-запросов)
  - [Пример вывода](#пример-вывода)
- [Async API (FastAPI + Celery + Redis)](#async-api-fastapi--celery--redis)
  - [Запуск async-стека](#запуск-async-стека)
  - [Создание задачи](#создание-задачи)
  - [Поллинг статуса](#поллинг-статуса)
  - [Параллельная обработка](#параллельная-обработка)
  - [Мониторинг](#мониторинг)
  - [Горизонтальное масштабирование](#горизонтальное-масштабирование)
- [Streamlit UI](#streamlit-ui)
  - [Возможности](#возможности)
- [Nginx + Production](#nginx--production)
  - [Архитектура контейнеров](#архитектура-контейнеров)
  - [Маршруты nginx](#маршруты-nginx)
  - [Запуск production-стека](#запуск-production-стека)
  - [Проверка](#проверка)
- [Kubernetes (minikube)](#kubernetes-minikube)
  - [Требования](#требования-1)
  - [Структура манифестов](#структура-манифестов)
  - [Архитектурные решения](#архитектурные-решения)
  - [Шаг 1: Запуск minikube с GPU](#шаг-1-запуск-minikube-с-gpu)
  - [Шаг 2: Сборка образов в minikube](#шаг-2-сборка-образов-в-minikube)
  - [Шаг 3: Развёртывание](#шаг-3-развёртывание)
  - [Шаг 4: Проверка](#шаг-4-проверка)
  - [Шаг 5: Использование](#шаг-5-использование)
  - [Маршруты Ingress](#маршруты-ingress)
  - [Полезные команды](#полезные-команды)
- [Guardrails](#guardrails)
  - [L1 — TopicGuard (input)](#l1--topicguard-input)
  - [L2 — SQLGuard (tool-level)](#l2--sqlguard-tool-level)
  - [L3 — ResponseGuard (output)](#l3--responseguard-output)
  - [Тестирование guards](#тестирование-guards)
- [Observability — Prometheus, Grafana, Langfuse](#observability--prometheus-grafana-langfuse)
  - [Метрики Prometheus](#метрики-prometheus)
  - [Дашборды Grafana](#дашборды-grafana)
  - [Алерты Prometheus](#алерты-prometheus)
  - [Трейсинг Langfuse](#трейсинг-langfuse)
  - [Запуск observability-стека](#запуск-observability-стека)
- [Оценка агентов](#оценка-агентов)
  - [RAG-пайплайн (RAGAS)](#rag-пайплайн-ragas)
  - [SQL Agent](#sql-agent)
  - [Supervisor Agent](#supervisor-agent)
  - [Response Agent](#response-agent)
  - [Сравнение RAG-экспериментов](#сравнение-rag-экспериментов)
- [Разработка](#разработка)
  - [Установка dev-зависимостей](#установка-dev-зависимостей)
  - [Code Quality](#code-quality)
  - [Тестирование](#тестирование)
- [Конфигурация](#конфигурация)
- [Текущая лучшая конфигурация RAG](#текущая-лучшая-конфигурация-rag)
- [Лицензия](#лицензия)

## Архитектура

Приложение построено на основе LangGraph и использует пять LLM-агентов
(Supervisor / SQL / RAG / Validator / Response) плюс трёхуровневую систему
[guardrails](#guardrails): regex-фильтр на входе, валидация SQL перед
выполнением, пост-обработка ответа. Supervisor классифицирует запрос
пользователя, определяет `intent`, `entities` и `query_type`, затем
маршрутизирует по одному из пяти путей (включая `off_topic` — запрос не по теме,
ответ выдаётся заготовленным текстом без вызова остальных агентов). Между
сообщениями память слоя [Memory Layer](#memory-layer) сохраняет историю диалога
и профиль пользователя — Supervisor и Response Agent читают короткоживущий
контекст + долгоживущий профиль из блока `MemoryManager`. Поверх этого работает
[observability-стек](#observability--prometheus-grafana-langfuse): Prometheus +
Grafana собирают метрики latency / throughput / error rate / очередей, а
Langfuse получает трейсы каждого запроса с полными промптами и токенами:

```
  [Memory Layer: conversation history + user profile]
    │  (ctx + profile)
    ▼
  Input Guardrail (L1: regex, prompt injection)
    │
    ▼
  Supervisor ──► (conditional routing by query_type)
    │    ▲
    │    └── conversation_context + user_profile (anaphora + default_team)
    │
    ├─ off_topic ─► OFF_TOPIC_RESPONSE ──────────────────────────► END
    │
    ├─ sql       ─► SQL Agent ─────────────► Validator ─► Response Agent ─► Output Guardrail (L3) ─► END
    │               (L2: SQLGuard в run_query)                  ▲
    │                                                          ctx + profile
    ├─ rag       ─► RAG Agent ─────────────► Validator ─► Response Agent ─► Output Guardrail (L3) ─► END
    │
    ├─ hybrid    ─► SQL Agent ──┐
    │              RAG Agent ──┴► Validator ─► Response Agent ─► Output Guardrail (L3) ─► END
    │
    └─ simple    ─► Response Agent (прямой ответ) ─────────────► Output Guardrail (L3) ─► END
                                                               │
                                                               ▼
                              [Memory Layer writes user + assistant turns,
                               schedules async profile refresh + summarisation]

  Параллельно для каждого узла:
    • Prometheus метрики (latency, count, in_progress) — scrape из api:8080/metrics, celery-worker:9100/metrics
    • Langfuse trace + nested spans (полные промпты + usage) — buffered, flush в конце задачи
```

### Компоненты

**1. Supervisor Agent** (классификатор запросов)

- Классифицирует запрос на один из 4 интентов:
  - `task` — запрос о конкретной задаче по ключу (например, "AL-38787")
  - `tasks_filter` — поиск задач по фильтрам (команда, спринт, тип, статус)
  - `metric` — запрос метрик команды/спринта (done_total, scope_drop и т.д.)
  - `general` — общий вопрос, не требующий данных из БД
- Определяет `query_type` для маршрутизации:
  - `sql` — нужны данные из PostgreSQL (конкретная задача, список, метрики)
  - `rag` — вопрос о теории, практиках, регламентах (без данных из БД)
  - `hybrid` — нужны данные из БД **и** рекомендации из базы знаний
  - `simple` — приветствие, общий вопрос без внешних данных
- Извлекает структурированные сущности (entities): issue_key, team_name,
  sprint_name, metric_name, issue_type, status, assignee, cluster
- **Fast path**: regex находит issue key → `intent=task`, `query_type=sql`, LLM
  не вызывается
- **Slow path**: LLM классифицирует запрос и возвращает JSON с intent +
  entities + query_type через structured output (vLLM guided decoding, JSON
  schema). `temperature=0.0` для детерминизма
- **Entity Sanitizer** (`entity_sanitizer.py`): пост-обработка выхода LLM в 7
  слоёв — (1) нормализация синонимов (enum-значения через `_SYNONYM_MAPS`), (2)
  валидация по БД (реальные issue_type / status), (3) фильтр галлюцинаций
  (шаблонные имена, плейсхолдеры), (4) enum query-presence check, (5) Russian
  morphology prefix match (4 символа) для sprint_name / cluster, (6)
  анафорический carry-forward с pymorphy3-лемматизацией маркеров, (7)
  fallback-экстрактор: тот же `_SYNONYM_MAPS` применяется как обратный словарь —
  если LLM не извлёк enum-поле, оно подставляется по совпадению лемм запроса с
  леммами синонимов
- **DB enum cache**: реальные значения `issue_type`, `status` подтягиваются из
  БД с TTL 1 час — сравнение с тем, что вернула LLM
- **Error intent fallback**: при parse failure возвращает
  `intent=error, query_type=error` вместо silent fallback — workflow корректно
  маршрутизирует в Response Agent с заготовленным сообщением

**2. SQL Agent** (LangGraph text-to-SQL с tool calling)

- Реализован как LangGraph StateGraph с узлами:
  `model → tools → check_retry → model → ... → extract`
- **LLM**: Qwen3-8B-AWQ (4-bit) в отдельном vLLM-контейнере (`vllm-sql`,
  порт 8001) с `--enable-auto-tool-choice --tool-call-parser=hermes`
- **Schema injection**: DDL схема БД (с `COMMENT ON` описаниями) загружается из
  `information_schema` через `get_schema_compact()` и встраивается в system
  prompt при каждом запуске (кэш 10 минут). Новые таблицы и колонки
  подхватываются автоматически — промпт менять не нужно
- **Единственный tool**: `run_query(sql)` — выполняет SELECT, возвращает модели
  sample (3 строки × 10 колонок) для экономии токенов
- **Guarantee tool call**: до первого успешного `run_query` LLM вызывается с
  `tool_choice="any"` — модель не может ответить текстом вместо SQL. После
  успешного запроса — свободный режим, чтобы завершить диалог
- **Retry on SQL error**: max 3 попытки. Если модель повторяет тот же SQL,
  добавляется подсказка исправить запрос
- **Full results**: tool отдаёт модели sample, но нода `extract` перевыполняет
  последний успешный SQL для возврата полных данных
- **Контекст-менеджмент**: блоки `<think>...</think>` от Qwen3 вырезаются из
  истории перед каждым вызовом — чтобы не упираться в 6144 токена (max-model-len
  vllm-sql)
- **Безопасность**: `run_query` блокирует всё кроме `SELECT`

**3. RAG Agent** (ответ на основе базы знаний)

- Использует Qdrant как векторное хранилище для документов из `knowledge_base/`
- Три режима поиска (управляется через `SEARCH_TYPE`):
  - `dense` (по умолчанию) — cosine similarity через
    `intfloat/multilingual-e5-base`
  - `sparse` — BM25 через fastembed или BGE-M3 learned sparse (управляется
    `EMBEDDING_SPARSE_MODEL`)
  - `hybrid` — dense + sparse с нативным Qdrant RRF fusion (prefetch API)
- Двухэтапный retrieval (при включённом реранкере): извлечение top-20 чанков →
  cross-encoder reranking (`BAAI/bge-reranker-v2-m3`) → top-4 наиболее
  релевантных
- Ограничение контекста: до 4000 символов из top-k релевантных чанков
- Reranker можно отключить через `RERANKER_ENABLED=false` для A/B-экспериментов
- Генерирует ответ через LLM строго на основе най��енного контекста
- Возвращает ответ с указанием и��точников (category/filename)
- Graceful degradation: если Qdrant недоступен, workflow продолжает работать для
  SQL-запросов. Если sparse vectors отсутствуют в коллекции, hybrid/sparse
  режимы автоматически откатываются на dense search

**4. Validator Agent** (валидация результатов)

- Проверяет выходы SQL Agent и RAG Agent перед передачей в Response Agent
- Для каждого `query_type` определяет, какие данные доступны:
  - `sql` — есть ли результат из БД
  - `rag` — есть ли ответ из базы знаний
  - `hybrid` — какая комбинация данных доступна (оба, только SQL, только RAG)
- Формирует `validation_result` с флагами `use_sql`, `use_rag` и `note`
  (описание ошибки при отсутствии данных)

**5. Response Agent** (генерация ответа)

- Единая константа `_SYSTEM_ROLE` используется во всех промптах («ассистент для
  анализа Jira-задач и Agile-метрик, по делу, без выдумывания данных»)
- Семь веток обработки в зависимости от `query_type` / `intent` /
  `validation_result`:
  - **error** (`query_type=error`): фиксированный ответ «классификатор
    недоступен», LLM не вызывается
  - **simple / direct**: приветствие, «что умеешь», мета-вопрос → LLM генерирует
    короткий дружелюбный ответ
  - **sql + intent=task**: форматирует одну задачу с русскими лейблами → LLM
    генерирует описание (ключ, тип, статус, команда, исполнитель, SP, спринт),
    пропуская поля с `None`
  - **sql + intent=tasks_filter**: компактный список до 20 задач (обрезка
    остатка), при `entities.assignee` в формат строки добавляется `@assignee` →
    LLM описывает результат с упоминанием команды/исполнителя
  - **sql + intent=metric**: метрики в JSON → LLM называет значения + динамику
    по спринтам, без рекомендаций
  - **rag**: passthrough `rag_response` + блок `**Источники:**` (без повторного
    LLM-вызова)
  - **hybrid**: SQL-данные + RAG-контекст → LLM строит ответ в два блока:
    `ДАННЫЕ` (числа/команда/спринт из БД) и `АНАЛИЗ` (рекомендации + источники
    **только** из переданного контекста, без галлюцинаций URL)
- Обрабатывает ошибки, пустые результаты («Задача AL-99999 не найдена») и
  таймауты LLM

**6. Memory Layer** (short-term диалог + long-term профиль)

- **MemoryManager** (`memory/manager.py`) — фасад над тремя репозиториями
  (`ConversationRepository`, `ProfileRepository`, `SummaryRepository`) +
  `ContextBuilder` + `ProfileExtractor`. Единственная точка входа для workflow /
  Celery — шина коммуникации с memory-модулем
- **ContextBuilder** — sliding window по токен-бюджету (`HISTORY_TOKEN_BUDGET`,
  1200 токенов по умолчанию): берёт последние полные ходы диалога, пока
  укладываются в бюджет; всё, что выпало, покрывается rolling summary (строится
  асинхронно Celery-таской `summarize_session`)
- **ProfileExtractor** — правило-based анализ `metadata` всех сообщений
  пользователя, считает частоты: `default_team` (команда с долей ≥ 60%),
  `frequent_metrics` (top-3), `recent_sprints`, `dominant_query_types`.
  Детерминированно и без LLM
- **Анафорический carry-forward** в entity_sanitizer layer 6: при анафорических
  маркерах ("эта команда", "тот спринт", "у них", "в том же") entities из
  предыдущего хода восстанавливаются в текущем, если LLM их не извлёк. Маркеры
  детектируются по леммам через `pymorphy3` — поэтому покрытие
  падежей/родов/чисел автоматическое (например, "в том же" / "тех же"
  лемматизируются к "тот", без ручного перечисления форм)
- **Inactivity rotation** в workflow_task: диалог, неактивный дольше
  `SESSION_TIMEOUT_MINUTES` (30 мин), автоматически закрывается и
  переоткрывается — каждая рабочая сессия остаётся скоупом для short-term
  контекста
- **Persistence**: таблицы `conversations`, `messages`, `user_profiles`,
  `conversation_summaries` (миграции Alembic 003–005). Content двойной —
  `content` (полный) + `content_truncated` (150 токенов для быстрой реплейки в
  промпт)

**7. Observability** (Prometheus + Grafana + Langfuse)

- **Prometheus метрики** (`hse_prom_prog/metrics.py`) — единый реестр из ~50
  кастомных метрик в неймспейсе `agile_assistant_*`. Покрывают весь pipeline:
  end-to-end latency и queue wait, Celery task lifecycle, per-agent durations,
  guardrails L1/L2/L3 results, entity sanitizer corrections by layer, RAG
  retrieval/reranker, memory context tokens, session rotations, LLM-as-a-Judge
  scores и periodic data-sync timestamps (Celery Beat)
- **Эндпоинты `/metrics`**: FastAPI экспонирует метрики через
  `prometheus-fastapi-instrumentator`, Celery worker и судья — через side-car
  `start_http_server` (порт 9100 для основного воркера, 9101 для judge).
  Prometheus скрейпит 9 целей: api, celery-worker, celery-judge, vllm-main,
  vllm-sql, postgresql (через pg-exporter), redis (через redis-exporter),
  qdrant, prometheus self
- **Grafana дашборды** (`monitoring/grafana/dashboards/`) — пять дашбордов в
  папке "Agile Assistant": Overview (e2e latency, throughput, error rate, data
  freshness), Infrastructure (PostgreSQL / Redis / Qdrant / vLLM), Agents Deep
  Dive (per-agent latency, SQL retries, RAG fallbacks, response branches),
  Guardrails & Safety (L1/L2/L3 blocks, sanitizer corrections, memory
  rotations), Quality (LLM-as-a-Judge per-criterion scores и weighted total)
- **Алерты Prometheus** (`monitoring/prometheus/alerts.yml`) — 18 правил в
  четырёх группах: `agile_assistant_pipeline` (latency / error rate / queue
  backlog), `infrastructure` (vLLM очереди и KV-cache, PostgreSQL replication
  lag, Qdrant healthcheck, sanitizer fallback rate), `agile_assistant_quality`
  (LLM-as-a-Judge weighted/per-criterion drops, judge API availability),
  `agile_assistant_data_freshness` (Jira CSV / KB stale, sync failures, Celery
  Beat liveness)
- **Langfuse трейсинг** (`hse_prom_prog/tracing.py`) — singleton-клиент с
  graceful degradation. На каждый Celery-таск создаётся root trace (`user_id`,
  `session_id`, `input` / `output`), внутри — вложенные spans для каждого агента
  (`@observe(name="...")`) и generations для LLM-вызовов
  (`@observe(as_type="generation")` с `model`, `input/output`, `usage`).
  Контекст распространяется через Python `contextvars` — параллелизма в hybrid
  нет (SQL+RAG последовательно), сиротских spans не возникает
- **Kill-switch**: `LANGFUSE_ENABLED=false` отключает SDK полностью — `@observe`
  становится no-op, приложение работает идентично без трейсов. Аналогично,
  отсутствие `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` не ломает запуск,
  только не отправляет данные. Подробности — в разделе
  [Observability](#observability--prometheus-grafana-langfuse)

### LLM Backend

Два раздельных vLLM-контейнера на одной GPU (совместный шаринг памяти):

- **Основной LLM** (`vllm-server`, порт 8000): avibe-gptq-8bit (GPTQ 8-bit),
  используется Supervisor, RAG Agent, Validator, Response Agent
- **SQL LLM** (`vllm-sql-server`, порт 8001): Qwen3-8B-AWQ (4-bit,
  compressed-tensors), используется только SQL Agent. Запущен с флагами
  `--enable-auto-tool-choice --tool-call-parser=hermes` для поддержки tool
  calling, `--max-model-len=6144`, `--gpu-memory-utilization=0.38`
- **GPU-бюджет** на RTX 4090 (24 GB): vllm (avibe) — `max-model-len=4096`,
  `gpu-memory-utilization=0.55`; vllm-sql (qwen3) — `max-model-len=6144`,
  `gpu-memory-utilization=0.38`. Итого ~22.3 GB из 24 GB. vllm-sql стартует
  первым (`depends_on: service_healthy`), чтобы избежать OOM-race
- **Модели** скачиваются из Yandex Cloud S3 при первом старте контейнера

### База данных

- **СУБД**: PostgreSQL 16 (Alpine)
- **Таблицы** (обе доступны SQL Agent-у, модель сама выбирает нужную):
  - `report_agile_dashboard` — задачи Jira (58 полей): issue_key, feature_teams,
    sprint_name, issue_status_act, storypoints_act, issue_type, assignee_name и
    др. Используется для запросов по задачам
  - `report_agile_dashboard_metrics` — агрегированные метрики команд по спринтам
    (35 полей): done_total, scope_drop, complete_sp (velocity), sprint_goal,
    cancel_rate и др. Используется для запросов по метрикам
- **Описания для LLM**: `COMMENT ON TABLE/COLUMN` в `init.sql` объясняют
  назначение таблиц и колонок — LLM SQL Agent читает их через `pg_description` и
  использует для выбора правильной таблицы и колонки
- **Индексы**: issue_key, jirasprint_id, sprint_state, assignee_name,
  feature_teams
- **Данные**: Загружаются из CSV файлов с использованием команды COPY

### Векторное хранилище (Qdrant)

- **Версия**: Qdrant v1.13.2
- **Коллекция**: `business_docs` (настраивается через `QDRANT_COLLECTION_NAME`)
- **Dense vectors**: `intfloat/multilingual-e5-base` (768-мерные, cosine
  similarity, мультиязычные — поддержка русского языка)
- **Sparse vectors**: BM25 через `fastembed` (по умолчанию) или BGE-M3 learned
  sparse (`EMBEDDING_SPARSE_MODEL=BAAI/bge-m3`)
- **Документы**: PDF и Markdown из `knowledge_base/` (Agile-практики, описания
  метрик, внутренние регламенты)
- **Ingestion pipeline**: загрузка из S3 (или локальной `knowledge_base/`) →
  pdfplumber (текст + таблицы) → chunking (500 символов, overlap 200) → prepend
  metadata (заголовок документа + секция) → dense embedding + sparse embedding →
  загрузка обоих типов в Qdrant

## Memory Layer

Двухуровневая память: короткоживущий контекст диалога (sliding window + rolling
summary) + долгоживущий профиль пользователя (default_team, frequent_metrics).
Вся память живёт в PostgreSQL, дополнительных контейнеров не требует.

Конфигурируется тремя env-переменными в `app-config` ConfigMap:

| Переменная                | Значение по умолчанию | Что задаёт                                                 |
| ------------------------- | --------------------- | ---------------------------------------------------------- |
| `HISTORY_TOKEN_BUDGET`    | 1200                  | Бюджет токенов на блок `<conversation_history>` в промптах |
| `SESSION_TIMEOUT_MINUTES` | 30                    | Через сколько минут idle диалог закрывается и ротируется   |
| `MAX_CONVERSATION_TURNS`  | 50                    | Верхняя граница хранимых ходов на один диалог              |

### Контекст диалога

Short-term память: последние N ходов + rolling summary более старых.

- **Sliding window по токенам**. `ContextBuilder` загружает все сообщения
  диалога, идёт от свежих к старым и копит ходы, пока суммарная стоимость не
  превышает `HISTORY_TOKEN_BUDGET`. Всегда оставляет хотя бы 1 ход.
- **Двойной контент**. Сообщения хранятся в двух полях: `content` (полный ответ,
  для пользователя) и `content_truncated` (~150 токенов, для replay в промпт).
  Длинный ответ ассистента не съест бюджет на следующем ходе.
- **Rolling summary**. Что не влезло в окно — покрывается `summary` диалога.
  Пересчитывается асинхронно через Celery (`summarize_session`), никогда не
  блокирует хот-путь.
- **Inactivity rotation**. Если после предыдущего хода прошло больше
  `SESSION_TIMEOUT_MINUTES`, workflow автоматически закрывает старый диалог,
  запускает суммаризацию (для long-term rolling profile) и создаёт новый.
- **Явная ротация**. Любой UI-выход из текущего чата — кнопка «Новый диалог» или
  клик по другому диалогу в истории — на бэкенде вызывает
  `POST /conversations/{id}/close`: текущая сессия закрывается, шедулится
  суммаризация и обновление long-term профиля. Отдельной кнопки «Завершить
  диалог» нет — её ментальная модель совпадает с «Новый диалог».
- **Анафорический carry-forward**. При анафорических маркерах («эта команда»,
  «тот спринт», «у них», «в том же», «покажи ещё») entities (`team_name`,
  `sprint_name`, `cluster`, `assignee`) из предыдущего хода восстанавливаются,
  если LLM не извлёк их из текущего запроса. Реализовано как layer 6 в
  `entity_sanitizer`, чтобы санитайзеры 1–5 сначала отбросили галлюцинации.
  Маркеры детектируются через лемматизацию (`pymorphy3`): запрос токенизируется,
  каждое слово приводится к нормальной форме и пересекается с компактным
  лемма-сетом (`этот`, `тот`, `такой`, `они`, `он`, `она`, `свой`, …) — все
  падежи/рода/числа покрываются без ручного перечисления.
- **Fallback enum extractor** (layer 7). Если LLM не заполнил `issue_type` /
  `status` / `metric_name`, тот же `_SYNONYM_MAPS` (что нормализует выход LLM в
  layer 1) переиспользуется как извлекатель из сырого запроса: токены синонима
  лемматизируются и проверяются на подмножество от лемм запроса. Один словарь —
  два направления, нет дублирования источника правды.

**Multi-turn eval** (`eval/run_multiturn_eval.py`, 14 кейсов, 5 подкатегорий):
carry-forward accuracy, false carry-forward rate, routing accuracy. Deploy gate:
≥ 85% carry accuracy, 0% false carry. Dataset расширяет
`eval/supervisor_golden_dataset.json` кейсами с массивом `turns`.

### Профиль пользователя

Long-term память, накапливается по сообщениям пользователя, дропается в промпты
Supervisor и Response Agent.

- **Идентификация**. В Streamlit стабильный UUID пинится в
  `st.query_params["uid"]` (модуль `streamlit_app/auth.py`) — один и тот же таб
  браузера попадает в одну и ту же строку `user_profiles`. Под SSO меняется
  только этот модуль.
- **Что сохраняется** (детерминированно, без LLM — `ProfileExtractor`):
  - `default_team` — команда, которая упоминалась в ≥ 60% сообщений
    пользователя; Supervisor подставляет её как team_name, когда запрос без
    явной команды (защищено rule «query wins over profile»).
  - `frequent_metrics` — top-3 метрики, о которых пользователь обычно
    спрашивает.
  - `recent_sprints`, `dominant_query_types` — для будущих подсказок.
- **Context summary** — rolling roll-up 10 последних session summary'ев; меньше
  3 — склеиваем, иначе LLM генерирует мета-саммари (max 200 токенов). Пишется в
  `user_profiles.context_summary` и отображается в промпте как
  `<user_profile>…</user_profile>`.
- **Асинхронное обновление**. После каждого завершённого хода Celery таска
  `update_profile_async` пересчитывает preferences + context_summary, чтобы не
  удлинять путь ответа пользователю.

## Структура проекта

```
hse-prom-prog/
├── hse_prom_prog/
│   ├── __init__.py
│   ├── config.py                      # Pydantic settings
│   ├── main.py                        # CLI entry point
│   ├── metrics.py                     # Prometheus реестр (~50 метрик в неймспейсе agile_assistant_*)
│   ├── tracing.py                     # Langfuse singleton + @observe реэкспорт + graceful degradation
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py              # Supervisor (intent + entities + query_type)
│   │   ├── entity_sanitizer.py        # 7-слойная пост-обработка entities (synonyms, DB validation, hallucination filter, lemma-based anaphora carry-forward, fallback enum extractor)
│   │   ├── sql_agent.py               # SQL agent (LangGraph + Qwen3 tool calling)
│   │   ├── sql_tools.py               # run_query tool для LangGraph SQL Agent
│   │   ├── schema_loader.py           # Загрузка DDL схемы БД для промпта
│   │   ├── schema_description.py     # Описание схемы БД для Supervisor
│   │   ├── rag_agent.py               # RAG agent (Qdrant + LLM)
│   │   ├── validator_agent.py         # Validator (проверка результатов)
│   │   ├── response_agent.py          # Response agent (LLM, 7 веток)
│   │   └── guardrails/
│   │       ├── __init__.py
│   │       ├── topic_guard.py         # L1: regex — prompt injection + whitelist
│   │       ├── sql_guard.py           # L2: 4 слоя защиты run_query (limits/regex/AST/joins)
│   │       └── response_guard.py      # L3: 6 rule-based проверок финального ответа
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                     # FastAPI application
│   │   ├── dependencies.py            # DI (DB, repo)
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── tasks.py               # POST/GET /tasks endpoints
│   │   │   └── conversations.py       # GET /conversations, GET /conversations/{id}/messages, POST /conversations/{id}/close
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── task.py                # Pydantic request/response (tasks)
│   │       └── conversation.py        # Pydantic request/response (memory layer)
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py              # PostgreSQL connection manager
│   │   ├── load_csv.py                # Load CSV data from S3 into PostgreSQL
│   │   └── task_repository.py         # Task CRUD (raw SQL)
│   ├── graph/
│   │   ├── __init__.py
│   │   └── workflow.py                # LangGraph StateGraph (5 LLM-агентов + 2 guardrail-узла + off_topic)
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py                  # OpenAI client для vLLM
│   ├── memory/                         # Memory Layer (short-term + long-term)
│   │   ├── __init__.py
│   │   ├── manager.py                  # Фасад: ConversationRepository + ProfileRepository + ContextBuilder
│   │   ├── context_builder.py          # Sliding window по HISTORY_TOKEN_BUDGET + rolling summary
│   │   ├── conversation_repo.py        # CRUD conversations + messages (raw SQL)
│   │   ├── profile_repo.py             # CRUD user_profiles (raw SQL)
│   │   ├── summary_repo.py             # CRUD conversation_summaries (raw SQL)
│   │   ├── profile_extractor.py        # Rule-based: default_team, frequent_metrics (no LLM)
│   │   ├── formatter.py                # <conversation_history> блок для промптов
│   │   ├── truncator.py                # Обрезка длинных сообщений до 150 токенов
│   │   └── token_estimator.py          # Деревянная оценка токенов без tiktoken
│   ├── models/
│   │   ├── __init__.py
│   │   ├── task.py                    # TaskStatus enum, Task model
│   │   └── memory.py                  # Conversation / Message / UserProfile / ConversationSummary / ConversationContext
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── bm25_index.py              # BM25 keyword index (rank-bm25)
│   │   ├── embeddings.py              # Shared embedding utils (truncation, Matryoshka)
│   │   ├── ingest.py                  # Ingestion pipeline (S3 → pdfplumber → Qdrant)
│   │   ├── reranker.py                # Cross-encoder reranker (bge-reranker-v2-m3)
│   │   ├── retriever.py               # Multi-mode retriever (dense/sparse/hybrid)
│   │   ├── sparse.py                  # Sparse embeddings (fastembed BM25 / BGE-M3)
│   │   └── tokenizer.py               # Deterministic text→SparseVector tokenizer
│   └── tasks/
│       ├── __init__.py
│       ├── celery_app.py              # Celery app factory + Beat schedule (sync_jira_data, sync_knowledge_base)
│       ├── workflow_task.py           # Celery task (wraps workflow + inactivity rotation)
│       ├── memory_tasks.py            # Celery: summarize_session, update_profile_async, _refresh_rolling_summary
│       ├── judge_task.py              # Phase 4: LLM-as-a-Judge async scoring (queue=judge, vsellm)
│       └── sync_tasks.py              # Phase 5: периодический S3→PostgreSQL и S3→Qdrant sync (Beat-планируется)
├── knowledge_base/                    # Загружается из S3 (S3_KB_BUCKET)
│   ├── agile/                         # Agile-практики, дашборды
│   ├── metrics/                       # Описания метрик
│   └── internal/                      # Внутренние регламенты
├── alembic/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       ├── 001_add_tasks_table.py
│       ├── 002_add_cleanup_function.py
│       ├── 003_add_conversations.py       # conversations + messages (short-term memory)
│       ├── 004_add_user_profiles.py       # user_profiles + conversation_summaries (long-term memory)
│       └── 005_add_conversation_id_to_tasks.py
├── database/
│   ├── init.sql                       # PostgreSQL schema (данные загружаются из S3)
│   └── init-langfuse.sql              # Создаёт langfuse_db рядом с основной БД
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml             # 9 scrape targets (api, celery-worker, celery-judge, vllm-main, vllm-sql, pg, redis, qdrant, self)
│   │   └── alerts.yml                 # 18 правил в 4 группах (pipeline / infrastructure / quality / data-freshness)
│   └── grafana/
│       ├── provisioning/              # Datasource (Prometheus) + dashboard provider
│       └── dashboards/                # 5 JSON-дашбордов: overview, infrastructure, agents, guardrails, quality
├── scripts/
│   └── download_model.sh              # Download avibe-gptq-8bit from S3
├── streamlit_app/
│   ├── app.py                         # Streamlit entrypoint (chat UI)
│   ├── api_client.py                  # HTTP client for FastAPI (tasks + conversations)
│   ├── auth.py                        # Stable uid via ?uid=… (будущий SSO)
│   ├── config.py                      # Env-based settings
│   └── components/
│       ├── __init__.py
│       ├── sidebar.py                 # Sidebar (status, «Новый диалог», история диалогов)
│       └── result.py                  # Result/error rendering
├── nginx/
│   ├── nginx.conf                     # Nginx reverse proxy config
│   └── Dockerfile.static              # Static files container
├── static/
│   ├── favicon.svg                    # Application favicon
│   ├── logo.svg                       # Application logo
│   └── style.css                      # Custom CSS
├── .streamlit/
│   └── config.toml                    # Streamlit theme
├── eval/
│   ├── __init__.py
│   ├── golden_dataset.json                # 41 вопрос для оценки RAG
│   ├── sql_golden_dataset.json            # 46 кейсов для SQL Agent
│   ├── supervisor_golden_dataset.json     # 81 single-turn + 14 multi-turn кейсов (95 всего)
│   ├── response_golden_dataset.json       # 40 кейсов для Response Agent
│   ├── metrics.py                         # RAGAS-метрики (GPT-5.2 as judge)
│   ├── run_eval.py                        # CLI: оценка RAG-пайплайна
│   ├── run_sql_eval.py                    # CLI: оценка SQL Agent
│   ├── run_supervisor_eval.py             # CLI: оценка Supervisor (routing + entities, single-turn)
│   ├── run_multiturn_eval.py              # CLI: multi-turn eval (carry-forward + false carry)
│   ├── run_response_eval.py               # CLI: оценка Response Agent (format + checks)
│   ├── compare.py                         # CLI: сравнение RAG-экспериментов
│   ├── analyze_tokens.py                  # Анализ prompt/completion токенов из логов
│   ├── spot_check.py                      # Точечная проверка retrieval без RAGAS
│   └── results/                           # Результаты (gitignored)
├── tests/
│   ├── __init__.py
│   ├── test_workflow.py               # 65 тестов (все агенты + workflow)
│   ├── test_memory.py                 # Тесты MemoryManager / ContextBuilder / ProfileExtractor
│   └── test_api_memory.py             # Тесты /conversations роутера + inactivity rotation
├── alembic.ini
├── docker-compose.yml
├── Dockerfile
├── .dockerignore
├── .env.example
├── .pre-commit-config.yaml
├── pyproject.toml
├── poetry.lock
└── README.md
```

## Быстрый старт с Docker Compose

Пошаговая инструкция для запуска полного стека (PostgreSQL + Qdrant + vLLM +
Redis + FastAPI + Celery + Streamlit + Nginx).

### Шаг 0: Клонирование и настройка

```bash
# Клонируйте репозиторий
git clone <repository-url>
cd hse-prom-prog

# Скачайте все ветки
git fetch --all

# Переключитесь на актуальную ветку
git checkout checkpoint_18

# Скопируйте файл окружения
cp .env.example .env
```

### Шаг 1: Сборка образа приложения

```bash
docker compose build app
```

Один образ используется для нескольких сервисов: `app`, `api`, `celery-worker`,
`streamlit`, `migrate`.

### Шаг 2: Запуск инфраструктуры

```bash
docker compose up -d postgres qdrant redis vllm
```

При первом запуске сервис `download-model` автоматически скачает модель
`avibe-gptq-8bit` (~5GB) из Yandex Cloud S3 в Docker volume. При повторных
запусках скачивание пропускается (модель уже на диске).

Дождитесь, пока все сервисы станут healthy (vLLM загружает модель — это может
занять несколько минут):

```bash
docker compose ps
```

Ожидаемый результат — все четыре сервиса в статусе `healthy`.

### Шаг 3: Применение миграций

```bash
docker compose run --rm migrate
```

Создаёт таблицу `tasks` в PostgreSQL для хранения статусов и результатов
асинхронных запросов.

### Шаг 4: Загрузка данных

```bash
# 4a. Загрузить CSV-данные из S3 в PostgreSQL
docker compose run --rm load-data

# 4b. Загрузить базу знаний из S3 в Qdrant
docker compose run --rm app python -m hse_prom_prog.rag.ingest
```

`load-data` скачивает CSV из S3 (`S3_DATA_BUCKET`) и загружает в PostgreSQL.
`ingest` скачивает PDF/Markdown из S3 (`S3_KB_BUCKET`) и индексирует в Qdrant.
Без этих шагов SQL-запросы и RAG-запросы не будут работать.

### Шаг 5: Запуск сервисов приложения

```bash
docker compose up -d api celery-worker streamlit nginx
```

### Шаг 6: Проверка и использование

```bash
# Проверьте, что все сервисы запущены
docker compose ps

# Откройте в браузере
open http://localhost
```

Streamlit UI доступен по адресу **http://localhost/** — это чат-интерфейс для
общения с Agile AI Assistant.

### Примеры запросов в Streamlit UI

В чате Streamlit можно отправлять запросы трёх типов:

**1. Запрос данных по конкретной задаче** (таблица `report_agile_dashboard`):

> Расскажи о задаче AL-38787

Supervisor извлекает ключ `AL-38787` и маршрутизирует в SQL Agent. LangGraph SQL
Agent через Qwen3-8B-AWQ генерирует и выполняет
`SELECT * FROM report_agile_dashboard WHERE issue_key ILIKE '%AL-38787%'`,
Response Agent формирует ответ с описанием задачи, статусом, исполнителем,
командой, story points и другими полями.

**2. Запрос метрик по спринту** (таблица `report_agile_dashboard_metrics`):

> Напиши мне Done Total по спринту 26Q1.1 Конь не валялся

Supervisor маршрутизирует в SQL Agent. Qwen3 видит схему обеих таблиц в system
prompt и выбирает `report_agile_dashboard_metrics`, генерирует SELECT с фильтром
по `sprint_name`, Response Agent возвращает значение `done_total`.

**3. Вопрос по документации компании** (RAG, база знаний в Qdrant):

> Расскажи мне, как считается метрика Team Lead Time

Supervisor определяет query_type=`rag`, RAG Agent ищет релевантные фрагменты в
Qdrant (из `knowledge_base/metrics/team_lead_time.pdf`), Response Agent
генерирует ответ на основе найденного контекста с указанием источников.

### Остановка

```bash
docker compose down
```

### Настройка переменных окружения

Перед запуском создайте файл `.env` на основе `.env.example`:

```bash
cp .env.example .env
```

Отредактируйте `.env` файл при необходимости:

```bash
# vLLM Configuration
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=/models/avibe-gptq-8bit
VLLM_API_KEY=EMPTY
VLLM_TEMPERATURE=0.05
VLLM_MAX_TOKENS=600

# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=hse_user
POSTGRES_PASSWORD=hse_password
POSTGRES_DB=hse_jira_db

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=business_docs
EMBEDDING_MODEL=intfloat/multilingual-e5-base

# S3 Model Storage (Yandex Cloud Object Storage)
S3_ENDPOINT=https://storage.yandexcloud.net
S3_BUCKET=quant-models-agile
S3_MODEL_PATH=models/avibe-gptq-8bit
AWS_ACCESS_KEY_ID=your-yc-key-id
AWS_SECRET_ACCESS_KEY=your-yc-secret-key
AWS_DEFAULT_REGION=ru-central1

# S3 Knowledge Base
S3_KB_BUCKET=knowledge-base
S3_KB_PATH=knowledge_base

# S3 CSV Data
S3_DATA_BUCKET=database-agile
S3_DATA_PATH=data

# VSELLM (LLM-as-judge для RAGAS evaluation)
VSELLM_API_KEY=your-vsellm-api-key
VSELLM_BASE_URL=https://api.vsellm.ru/v1

# Logging
LOG_LEVEL=INFO
```

**Примечания:**

- Для Docker Compose используйте `VLLM_BASE_URL=http://vllm:8000/v1`,
  `POSTGRES_HOST=postgres`, `QDRANT_URL=http://qdrant:6333`
- Для Kubernetes сервис vLLM называется `vllm-server` (не `vllm`), чтобы
  избежать конфликта с переменной `VLLM_PORT`, которую Kubernetes создаёт
  автоматически. Используйте `VLLM_BASE_URL=http://vllm-server:8000/v1`
- Для локальной разработки используйте `localhost` для всех сервисов
- Значения по умолчанию подходят для большинства случаев использования

## База знаний и RAG

RAG Agent использует базу знаний для ответов на вопросы о теории, практиках и
регламентах. Документы хранятся в S3 (бакет `S3_KB_BUCKET`, по умолчанию
`knowledge-base`) и загружаются в Qdrant через ingestion pipeline. При
отсутствии S3-конфигурации используется локальная папка `knowledge_base/`.

### Структура knowledge_base/

```
knowledge_base/
├── agile/                  # Agile-практики и дашборды
│   └── agile_dashboard.pdf
├── metrics/                # Описания метрик
│   ├── done_total.pdf
│   ├── scope_drop.pdf
│   ├── velocity_and_capacity.pdf
│   ├── sprint_goals.pdf
│   ├── cancel_rate.pdf
│   ├── groomed_backlog.pdf
│   ├── team_lead_time.pdf
│   ├── team_lead_time_85.pdf
│   ├── backlog_dynamics.pdf
│   ├── sprint_tasks_grooming.pdf
│   ├── sankey_task_issues.pdf
│   ├── retro_ai.pdf
│   └── jira_status_mapping.pdf
└── internal/               # Внутренние регламенты
```

Поддерживаемые форматы: `.pdf`, `.md`.

### Загрузка документов в Qdrant

После обновления документов в S3 необходимо запустить ingestion pipeline для
загрузки в Qdrant:

```bash
# Через Docker Compose (скачивает из S3, Qdrant должен быть запущен)
docker compose run --rm app python -m hse_prom_prog.rag.ingest

# С указанием локальной папки (вместо S3)
docker compose run --rm -e S3_KB_BUCKET= \
  app python -m hse_prom_prog.rag.ingest /path/to/docs
```

Pipeline выполняет:

1. **Загрузка** — скачивает из S3 (или локальной папки), читает .pdf
   (pdfplumber: текст + денормализация таблиц) и .md (TextLoader)
2. **Chunking** — текстовые документы разбиваются на фрагменты (500 символов,
   overlap 200); таблицы уже самодостаточны и не чанкируются
3. **Prepend metadata** — в начало каждого чанка добавляется заголовок документа
   и секция для улучшения retrieval
4. **Dense embedding** — `intfloat/multilingual-e5-base` (768-d, CPU), с
   опциональной Matryoshka-truncation (`EMBEDDING_DIMENSION`)
5. **Sparse embedding** — BM25 через `fastembed` или BGE-M3 learned sparse
6. **Загрузка в Qdrant** — пересоздаёт коллекцию с dense + sparse vectors и
   загружает все чанки

При повторном запуске коллекция пересоздаётся (идемпотентность).

## Локальная разработка

### Требования

- Python 3.12+
- Poetry
- PostgreSQL 16 (или Docker для PostgreSQL)
- Qdrant (или Docker для Qdrant)
- vLLM (или Docker для vLLM)
- NVIDIA GPU (рекомендуется для vLLM)

### Шаг 1: Установка зависимостей

```bash
# Клонируйте репозиторий
git clone <repository-url>
cd hse-prom-prog

# Установите зависимости через Poetry
poetry install
```

### Шаг 2: Настройка PostgreSQL

**Вариант A: Используя Docker**

```bash
docker run --name hse-postgres \
    -e POSTGRES_USER=hse_user \
    -e POSTGRES_PASSWORD=hse_password \
    -e POSTGRES_DB=hse_jira_db \
    -p 5432:5432 \
    -v $(pwd)/database/init.sql:/docker-entrypoint-initdb.d/init.sql \
    -d postgres:16-alpine
```

**Вариант B: Локальный PostgreSQL**

```bash
# Создайте базу данных
createdb -U postgres hse_jira_db

# Инициализируйте схему и данные
psql -U postgres -d hse_jira_db -f database/init.sql
```

### Шаг 3: Скачивание модели и запуск vLLM

Модель `avibe-gptq-8bit` хранится в Yandex Cloud S3. Скачайте её локально:

```bash
# Установите AWS CLI (если ещё не установлен)
pip install awscli

# Скачайте модель из S3
./scripts/download_model.sh /models/avibe-gptq-8bit
```

Запустите vLLM с квантизованной моделью:

```bash
docker run -d --gpus all --name vllm-server -p 8000:8000 \
    -v /models:/models \
    vllm/vllm-openai:v0.8.5 \
    --model /models/avibe-gptq-8bit --quantization gptq --dtype float16 --max-model-len 8192
```

### Шаг 4: Запуск приложения

```bash
poetry run python -m hse_prom_prog.main "Привет! Выведи данные по задаче AL-38787"
```

## Использование

### Примеры запросов

```bash
# Через Docker Compose
docker compose run --rm app python -m hse_prom_prog.main "Выведи данные по задаче AL-38787"

# Локально
poetry run python -m hse_prom_prog.main "Выведи данные по задаче AL-38787"

# ── query_type=sql, intent=task (конкретная задача по ключу) ──
poetry run python -m hse_prom_prog.main "Информация о задаче AL-38799"
poetry run python -m hse_prom_prog.main "Покажи мне что там с AL-39041"
poetry run python -m hse_prom_prog.main "AL-39043"

# ── query_type=sql, intent=tasks_filter (поиск задач по фильтрам) ──
poetry run python -m hse_prom_prog.main "Все задачи команды cthulhu"
poetry run python -m hse_prom_prog.main "Задачи команды lpop в спринте #1 Q1'26"
poetry run python -m hse_prom_prog.main "Баги в кластере Logistics"
poetry run python -m hse_prom_prog.main "Задачи со статусом In Progress"

# ── query_type=sql, intent=metric (метрики команд/спринтов) ──
poetry run python -m hse_prom_prog.main "Done total из спринта Мандариновый рывок"
poetry run python -m hse_prom_prog.main "Какой scope drop у команды cthulhu"
poetry run python -m hse_prom_prog.main "Метрики команды lpop"
poetry run python -m hse_prom_prog.main "Velocity команды linehaul"

# ── query_type=rag (вопросы о практиках и метриках из базы знаний) ──
poetry run python -m hse_prom_prog.main "Что такое Definition of Done?"
poetry run python -m hse_prom_prog.main "Как снизить Scope Drop?"
poetry run python -m hse_prom_prog.main "Какие бейзлайновые значения метрик?"
poetry run python -m hse_prom_prog.main "Как рассчитывается velocity?"

# ── query_type=hybrid (данные из БД + рекомендации из базы знаний) ──
poetry run python -m hse_prom_prog.main "Покажи scope drop команды cthulhu и дай рекомендации"
poetry run python -m hse_prom_prog.main "Метрики lpop и что можно улучшить"

# ── query_type=simple (общие вопросы, без обращения к БД и RAG) ──
poetry run python -m hse_prom_prog.main "Что такое спринт в Agile?"
poetry run python -m hse_prom_prog.main "Как рассчитываются story points?"
poetry run python -m hse_prom_prog.main "Привет"
```

### Пример вывода

```
🤖 Agile AI Assistant
📝 Query: Выведи данные по задаче AL-38787

============================================================

[Supervisor] Извлекаю ключ задачи...
[Supervisor] Найден ключ: AL-38787

[SQL Agent] Processing: Выведи данные по задаче AL-38787
[SQL Agent] run_query: SELECT * FROM report_agile_dashboard WHERE issue_key ILIKE '%AL-38787%'
[SQL Agent] Returned 2 row(s)

[Response Agent] Форматирую ответ...

============================================================

=== ФИНАЛЬНЫЙ ОТВЕТ ===

Конечно, я расскажу вам о задаче AL-38787. Вот основные детали:

- **Ключ задачи**: AL-38787
- **Проект**: DeepMind Logistics
- **Тип задачи**: Улучшение
- **Описание**: Приемочное тестирование для B2C FBS ПВЗ-Постамата
- **Текущий статус**: В процессе выполнения
- **Статус в конце спринта**: Открытый
- **Дата создания**: 22 сентября 2025 года
- **Исполнитель**: Юпитер Петров
- **Команда**: lpop
- **Команды в начале и конце спринта**: lpop
- **Репортер**: Владимир Реценков
- **Завершение спринта**: 26 января 2026 года в 12:03

В этом спринте было определено 5.0 Story Points, что стало меньше по сравнению с началом спринта (3.0 Story Points). Исполнитель потратил 3427 часов на эту задачу, что означает, что работа заняла больше времени, чем планировалось.

Если у вас есть еще вопросы или нужна дополнительная информация, не стесняйтесь спрашивать!

============================================================
```

## Async API (FastAPI + Celery + Redis)

Помимо CLI-интерфейса, приложение поддерживает асинхронную обработку задач через
REST API. Клиент отправляет запрос и получает `task_id` (HTTP 202), а результат
забирает позже через поллинг.

```
Client → POST /tasks → FastAPI → Redis Queue → Celery Worker → LangGraph Workflow
                ↓                                      ↓
         PostgreSQL (tasks)                   PostgreSQL (tasks)
                ↑                                      ↓
Client ← GET /tasks/{id} ←────────────────────────────┘
```

Celery worker автоматически подключается к Qdrant для RAG-запросов. Если Qdrant
недоступен, worker продолжает обрабатывать SQL- и simple-запросы.

### Запуск async-стека

```bash
# 1. Поднять инфраструктуру
docker compose up -d postgres qdrant redis vllm

# 2. Дождаться готовности и применить миграции Alembic
docker compose run --rm migrate

# 3. Загрузить данные из S3
docker compose run --rm load-data                              # CSV → PostgreSQL
docker compose run --rm app python -m hse_prom_prog.rag.ingest # KB → Qdrant

# 4. Запустить API и Celery-воркер
docker compose up -d api celery-worker

# 5. Проверить, что все сервисы healthy
docker compose ps
```

### Создание задачи

```bash
curl -s -X POST http://localhost/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"query": "Расскажи о задаче AL-38787"}'
```

Ожидаемый ответ (HTTP 202):

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PENDING",
  "message": "Task created and queued for processing"
}
```

### Поллинг статуса

```bash
# Подставьте task_id из ответа выше
curl -s http://localhost/api/tasks/<task_id> | python3 -m json.tool
```

Задача проходит через статусы: `PENDING` → `PROCESSING` → `COMPLETED` /
`FAILED`.

Пример финального ответа (HTTP 200):

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "Расскажи о задаче AL-38787",
  "status": "COMPLETED",
  "result": {
    "final_response": "Задача AL-38787: проект DeepMind Logistics, статус In Progress...",
    "issue_key": "AL-38787",
    "query_type": "sql"
  },
  "error": null,
  "created_at": "2026-02-13T12:00:00Z",
  "started_at": "2026-02-13T12:00:01Z",
  "completed_at": "2026-02-13T12:00:15Z"
}
```

### Параллельная обработка

Чтобы убедиться, что тяжёлая задача не блокирует остальные, отправьте несколько
запросов одновременно:

```bash
for i in 38787 38799 39041; do
  curl -s -X POST http://localhost/api/tasks \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"Расскажи о задаче AL-\$i\"}" &
done
wait
```

Все три запроса вернут `202 Accepted` **моментально**. Celery-воркер
обрабатывает до 4 задач параллельно (`--pool=threads --concurrency=4`), поэтому
каждая задача выполняется в отдельном потоке и не блокирует соседние.

Проверка результатов в PostgreSQL:

```bash
docker compose exec postgres psql -U hse_user -d hse_jira_db -c \
  "SELECT task_id, status, created_at, started_at, completed_at FROM tasks ORDER BY created_at;"
```

### Мониторинг

```bash
# Логи воркера в реальном времени
docker compose logs -f celery-worker

# Активные задачи
docker compose exec celery-worker \
  celery -A hse_prom_prog.tasks.celery_app inspect active

# Длина очереди в Redis
docker compose exec redis redis-cli LLEN celery
```

### Горизонтальное масштабирование

```bash
# Запустить 3 воркера (= 12 параллельных задач)
docker compose up --scale celery-worker=3 -d

# Проверить, что все воркеры подключены
docker compose exec celery-worker \
  celery -A hse_prom_prog.tasks.celery_app inspect ping
```

## Streamlit UI

Веб-интерфейс для общения с Agile AI Assistant в формате чата. Streamlit
работает как тонкий клиент и общается **только** с FastAPI по HTTP. Доступен
через nginx на `http://localhost/`.

### Возможности

- Чат-интерфейс с историей сообщений
- Поддержка всех типов запросов:
  - Поиск конкретной задачи по ключу ("Расскажи о задаче AL-38787")
  - Поиск задач по фильтрам ("Задачи команды cthulhu")
  - Запрос метрик команд ("Done total из спринта Мандариновый рывок")
  - Вопросы о практиках из базы знаний ("Как снизить Scope Drop?")
  - Гибридные запросы ("Метрики lpop и что можно улучшить")
  - Общие вопросы ("Что такое спринт?", "Привет")
- Прогресс-индикатор обработки задачи (PENDING -> PROCESSING -> COMPLETED)
- Обработка ошибок и таймаутов с понятными сообщениями
- Детали выполнения (timestamps) в раскрывающемся блоке

### Сайдбар и память

- **Стабильный `user_id`** пинится в URL как `?uid=...` (модуль
  `streamlit_app/auth.py`) — рефреш страницы сохраняет личность юзера, и сайдбар
  продолжает показывать его историю диалогов.
- **История диалогов** — список последних 20 сессий с заголовком (первый запрос
  пользователя, обрезанный до 50 символов) и относительным временем («5 мин»,
  «вчера», «3 дн.»). Клик переключает чат на выбранный диалог и восстанавливает
  транскрипт из API (`GET /conversations/{id}/messages`).
- **Новый диалог** — единственная chat-control кнопка. Любой выход из текущего
  чата (нажатие «Новый диалог» или клик по другому диалогу в истории) сначала
  тихо вызывает `POST /conversations/{id}/close` для текущей сессии, если в ней
  были сообщения. На бэкенде это ставит в очередь `summarize_session` (long-term
  summary + rolling profile refresh), пользователь ничего лишнего не нажимает.
- **URL как source of truth при refresh**. При загрузке страницы
  `conversation_id` достаётся из `?cid=...` и транскрипт перезагружается через
  API — не теряется при F5.
- **Auto-rotation из worker'а**. Если Celery при обработке нашёл, что старая
  сессия idle > `SESSION_TIMEOUT_MINUTES`, он ротирует её и возвращает новый
  `conversation_id` в результате; Streamlit считывает его и переподпинывает URL
  — UI остаётся в согласованном состоянии.

## Nginx + Production

Nginx выступает единой точкой входа (порт 80). FastAPI, Streamlit, Prometheus и
Grafana не имеют внешних портов и доступны только через reverse proxy.
Внутренние upstream-ы (`fastapi`, `streamlit`, `prometheus`, `grafana`,
`langfuse`) объявлены в [nginx/nginx.conf](nginx/nginx.conf). Для тяжёлых JS
бандлов Grafana настроены увеличенные `proxy_buffers` и gzip — без них первая
загрузка дашбордов уходила в 30-60 секунд из-за дисковой буферизации.

### Архитектура контейнеров

```
                          ┌─────────────────────────────────────┐
                          │        Docker Compose network       │
  Browser ──► :80 ──►     │                                     │
                          │  ┌─────────┐                        │
                          │  │  nginx  │ (Image 1)              │
                          │  └────┬────┘                        │
                          │       │                             │
                ┌─────────┼───────┼──────────┐                  │
                │         │       │          │                  │
          /static/        │   /api/*       / (default)          │
                │         │       │          │                  │
          ┌─────▼───┐     │ ┌─────▼───┐ ┌───▼──────┐           │
          │ static  │     │ │   api   │ │streamlit │           │
          │(Image 3)│     │ │(Image 2)│ │          │           │
          │ volume  │     │ │gunicorn │ │          │           │
          └─────────┘     │ └─────────┘ └──────────┘           │
                          └─────────────────────────────────────┘
```

| Контейнер          | Образ                                   | Роль                                                                  |
| ------------------ | --------------------------------------- | --------------------------------------------------------------------- |
| **nginx**          | `nginx:1.27-alpine`                     | Reverse proxy, единственный открытый порт (80)                        |
| **api**            | Dockerfile + gunicorn                   | FastAPI в production (gunicorn + UvicornWorker), экспонирует /metrics |
| **celery-worker**  | Dockerfile + celery threads             | Workflow-исполнитель, side-car `start_http_server(9100)` для метрик   |
| **static**         | `busybox` + volume                      | Хранит статические файлы, шарит volume с nginx                        |
| **qdrant**         | `qdrant:v1.13.2`                        | Векторное хранилище для RAG Agent                                     |
| **prometheus**     | `prom/prometheus:v2.53.0`               | Time-series для метрик; subpath `/prometheus/` за nginx               |
| **grafana**        | `grafana/grafana:11.1.0`                | Дашборды поверх Prometheus; subpath `/grafana/` за nginx              |
| **pg-exporter**    | `prometheuscommunity/postgres-exporter` | Side-car для PostgreSQL метрик                                        |
| **redis-exporter** | `oliver006/redis_exporter`              | Side-car для Redis метрик                                             |
| **langfuse**       | `langfuse/langfuse:2`                   | LLM-tracing UI + API; пишет в `langfuse_db` (тот же PostgreSQL)       |

### Маршруты nginx

| Путь              | Куда проксирует   | Особенности                                                                                   |
| ----------------- | ----------------- | --------------------------------------------------------------------------------------------- |
| `/api/*`          | `api:8080`        | Prefix `/api` удаляется (`/api/tasks` -> `/tasks`); `/api/metrics` отдаёт 404 (internal-only) |
| `/static/*`       | Диск / Streamlit  | Сначала volume, затем fallback на Streamlit                                                   |
| `/docs`, `/redoc` | `api:8080`        | Swagger UI                                                                                    |
| `/_stcore/stream` | `streamlit:8501`  | WebSocket (Upgrade + Connection headers)                                                      |
| `/prometheus/`    | `prometheus:9090` | Prometheus UI (gzip, large proxy_buffers для JS-бандлов)                                      |
| `/grafana/`       | `grafana:3000`    | Grafana UI (gzip, WebSocket для Live, large proxy_buffers)                                    |
| `/langfuse/`      | `langfuse:3000`   | Langfuse UI (опционально; основной доступ — `localhost:3001`)                                 |
| `/` (default)     | `streamlit:8501`  | Streamlit UI с WebSocket-поддержкой                                                           |

### Запуск production-стека

```bash
# 1. Запустить всё
docker compose up -d

# 2. Загрузить данные из S3 (при первом запуске)
docker compose run --rm load-data
docker compose run --rm app python -m hse_prom_prog.rag.ingest

# 3. Открыть в браузере
open http://localhost

# FastAPI доступен через /api/:
curl http://localhost/api/tasks -X POST -H "Content-Type: application/json" \
  -d '{"query": "Расскажи о задаче AL-38787"}'

# Swagger UI:
open http://localhost/docs

# Статика:
curl http://localhost/static/style.css
```

### Проверка

```bash
# Статус всех сервисов
docker compose ps

# Логи nginx
docker compose logs nginx

# Проверить, что статика отдаётся напрямую
curl -I http://localhost/static/favicon.svg
# Должен вернуть Content-Type: image/svg+xml и Cache-Control: public

# Проверить, что FastAPI работает через gunicorn
docker compose exec api ps aux | grep gunicorn

# Проверить Qdrant
curl http://localhost:6333/healthz
```

## Kubernetes (minikube)

Проект можно развернуть в Kubernetes с помощью minikube. Все манифесты находятся
в `k8s/`.

### Требования

- [minikube](https://minikube.sigs.k8s.io/) v1.38+
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- Docker (driver для minikube)
- NVIDIA GPU + драйверы (для vLLM)
- [CloudNativePG](https://cloudnative-pg.io/) v1.25+ (устанавливается на шаге 1)

### Структура манифестов

```
k8s/
├── namespace.yaml                # Namespace agile-assistant
├── ingress.yaml                  # Ingress (API rewrite, static MIME fix, Streamlit/WebSocket)
├── configmaps/
│   ├── app-config.yaml           # Env vars (DNS names, credentials)
│   └── postgres-init.yaml        # init.sql (schema + COPY)
├── jobs/
│   ├── migrate.yaml              # Job: Alembic migrations (backoffLimit: 3)
│   ├── qdrant-ingest.yaml        # Job: load knowledge base into Qdrant
│   └── postgres-load-data.yaml   # Job: load CSV data into HA PostgreSQL
├── secrets/
│   ├── app-secrets.yaml          # Opaque Secret (POSTGRES_PASSWORD, VLLM_API_KEY)
│   └── postgres-credentials.yaml # basic-auth Secret для CloudNativePG (username/password)
│   # registry-credentials и basic-auth создаются через kubectl (шаг 2)
├── statefulsets/
│   └── postgres-cluster.yaml     # CloudNativePG Cluster (1 primary + 2 standby)
├── storage/
│   ├── qdrant-pvc.yaml           # PVC 2Gi
│   └── vllm-cache-pvc.yaml      # PVC 10Gi (HuggingFace model cache)
├── deployments/
│   ├── qdrant.yaml               # Qdrant v1.13.2
│   ├── redis.yaml                # Redis 7 (ephemeral)
│   ├── redis-exporter.yaml       # oliver006/redis_exporter (порт 9121, для ServiceMonitor)
│   ├── vllm.yaml                 # vLLM main (avibe-gptq-8bit, GPU, S3 init)
│   ├── vllm-sql.yaml             # vLLM SQL (Qwen3-8B-AWQ, GPU, S3 init)
│   ├── api.yaml                  # FastAPI (gunicorn, 2 replicas)
│   ├── celery-worker.yaml        # Celery main worker (threads, concurrency=4, side-car :9100)
│   ├── celery-judge.yaml         # Celery judge worker (queue=judge, side-car :9101)
│   ├── celery-beat.yaml          # Celery Beat scheduler (replicas:1 + Recreate)
│   └── streamlit.yaml            # Streamlit UI
├── services/
│   ├── qdrant-svc.yaml               # ClusterIP :6333, :6334
│   ├── redis-svc.yaml                # ClusterIP :6379
│   ├── redis-exporter-svc.yaml       # ClusterIP :9121 (port name: http-metrics)
│   ├── vllm-svc.yaml                 # ClusterIP :8000 (name: vllm-server)
│   ├── vllm-sql-svc.yaml             # ClusterIP :8000 (Service: vllm-sql)
│   ├── api-svc.yaml                  # ClusterIP :8080 (port name: http)
│   ├── celery-worker-svc.yaml        # Headless (clusterIP: None) — для ServiceMonitor discovery
│   ├── celery-judge-svc.yaml         # Headless — для ServiceMonitor discovery
│   └── streamlit-svc.yaml            # ClusterIP :8501
└── monitoring/                       # Prometheus Operator CRD (заменяют static_configs из docker-compose)
    ├── kube-prometheus-stack-values.yaml  # Helm values для оператора
    ├── api-servicemonitor.yaml            # → job=fastapi
    ├── celery-worker-servicemonitor.yaml  # → job=celery-worker
    ├── celery-judge-servicemonitor.yaml   # → job=celery-judge
    ├── vllm-main-servicemonitor.yaml      # → job=vllm-main
    ├── vllm-sql-servicemonitor.yaml       # → job=vllm-sql
    ├── qdrant-servicemonitor.yaml         # → job=qdrant
    ├── postgres-podmonitor.yaml           # PodMonitor по cnpg.io/cluster, job=postgresql
    ├── redis-exporter-servicemonitor.yaml # → job=redis
    ├── prometheus-rules.yaml              # PrometheusRule CRD — те же 18 алертов, что в alerts.yml
    └── README.md                          # Установка оператора + миграция с static_configs
```

### Архитектурные решения

- **PostgreSQL HA** — управляется оператором CloudNativePG. Кластер из 3
  инстансов (1 primary + 2 standby) со streaming replication и автоматическим
  failover. Оператор создаёт Service-ы `postgres-cluster-rw` (запись) и
  `postgres-cluster-ro` (чтение). Приложение подключается через
  `postgres-cluster-rw`.
- **Secrets** — пароли (`POSTGRES_PASSWORD`, `VLLM_API_KEY`) хранятся в
  `k8s/secrets/app-secrets.yaml` (Secret типа Opaque), а не в ConfigMap.
  `registry-credentials` (docker-registry) — аутентификация в GHCR. `basic-auth`
  — защита Ingress паролем (Basic Auth).
- **Docker Registry** — образы публикуются в GitHub Container Registry
  (`ghcr.io/akiltrebreg/agile-assistant`). Deployment-ы и Job-ы используют
  `imagePullPolicy: IfNotPresent` с `imagePullSecrets` вместо локальной сборки
  через `minikube docker-env`.
- **Basic Auth** — все Ingress-ресурсы защищены HTTP Basic Authentication. При
  открытии в браузере запрашивается логин и пароль.
- **Jobs** — миграции Alembic и загрузка данных (CSV в PostgreSQL, чанки в
  Qdrant) запускаются как Job-ы с `backoffLimit` для автоматического повтора при
  ошибках и `ttlSecondsAfterFinished: 3600` для автоочистки.
- **Ingress** — snippet-аннотации для корректных MIME-типов статических файлов
  (SVG, CSS). Требуют включения `allow-snippet-annotations` в
  ingress-controller.
- **vLLM** — Service называется `vllm-server` (не `vllm`), чтобы избежать
  конфликта с переменной `VLLM_PORT`, которую Kubernetes автоматически создаёт
  из имени Service.

### Шаг 1: Запуск minikube с GPU

```bash
minikube start --driver=docker --gpus=all --cpus=8 --memory=13000 --disk-size=40g
```

Включите необходимые аддоны:

```bash
minikube addons enable ingress
minikube addons enable metrics-server
minikube addons enable nvidia-device-plugin
```

Настройте snippet-аннотации для ingress-controller (нужны для корректных
MIME-типов статических файлов):

```bash
kubectl -n ingress-nginx wait --for=condition=ready pod -l app.kubernetes.io/component=controller --timeout=120s
kubectl -n ingress-nginx patch configmap ingress-nginx-controller \
  --type merge \
  -p '{"data":{"allow-snippet-annotations":"true","annotations-risk-level":"Critical"}}'
kubectl -n ingress-nginx rollout restart deployment ingress-nginx-controller
kubectl -n ingress-nginx rollout status deployment ingress-nginx-controller
```

Установите CloudNativePG оператор для PostgreSQL HA:

```bash
kubectl apply --server-side -f \
  https://raw.githubusercontent.com/cloudnative-pg/cloudnative-pg/release-1.25/releases/cnpg-1.25.1.yaml
kubectl -n cnpg-system wait --for=condition=ready pod -l app.kubernetes.io/name=cloudnative-pg --timeout=120s
```

### Шаг 2: Сборка и публикация образа

Соберите и запушьте образ в GitHub Container Registry:

```bash
docker build -t ghcr.io/akiltrebreg/agile-assistant:latest .
docker push ghcr.io/akiltrebreg/agile-assistant:latest
```

> **Для minikube** — образ ~2GB (CPU-only PyTorch, без CUDA). Чтобы ускорить
> деплой, можно загрузить образ напрямую из локального Docker в minikube вместо
> pull из GHCR:
>
> ```bash
> minikube image load ghcr.io/akiltrebreg/agile-assistant:latest
> ```

> **Аутентификация в GHCR** (обязательна для push и pull):
>
> ```bash
> echo "<GHCR_PAT>" | docker login ghcr.io -u <github-username> --password-stdin
> ```
>
> Создайте Personal Access Token (classic) с правами `read:packages` /
> `write:packages`: https://github.com/settings/tokens → Generate new token
> (classic). Контрибьюторы репозитория могут pull-ить образ, залогинившись со
> своим PAT.

Создайте Secret для аутентификации в registry:

```bash
kubectl create namespace agile-assistant 2>/dev/null || true
kubectl -n agile-assistant create secret docker-registry registry-credentials \
  --docker-server=ghcr.io \
  --docker-username=akiltrebreg \
  --docker-password=<GHCR_PAT> \
  --docker-email=<ваш-email>
```

Создайте Secret для Basic Auth (защита UI/API паролем):

```bash
sudo apt install apache2-utils -y    # если ещё не установлен
htpasswd -c auth admin               # введите пароль дважды
kubectl -n agile-assistant create secret generic basic-auth --from-file=auth
rm auth                              # локальный файл больше не нужен
```

### Шаг 3: Развёртывание

Применяйте манифесты строго в указанном порядке — каждый следующий шаг зависит
от предыдущего.

**3.1. Базовые ресурсы** — конфигурация, секреты, PVC:

```bash
# Namespace уже создан на шаге 2, но на всякий случай:
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmaps/             # app-config (env vars) + postgres-init (SQL схема)
kubectl apply -f k8s/secrets/                # app-secrets + postgres-credentials (для CloudNativePG)
# registry-credentials и basic-auth уже созданы на шаге 2
kubectl apply -f k8s/storage/                # PVC для Qdrant (2Gi) и vLLM model cache (10Gi)
```

**3.2. Service-ы** — создают DNS-имена для межсервисного взаимодействия:

```bash
kubectl apply -f k8s/services/               # qdrant, redis, redis-exporter, vllm-server, vllm-sql, api, streamlit + headless celery-worker / celery-judge для ServiceMonitor
```

> Service-ы применяются до Deployment-ов, чтобы DNS-имена были доступны при
> старте контейнеров. PostgreSQL Service создаётся автоматически оператором
> CloudNativePG.

**3.3. Инфраструктура** — БД, векторное хранилище, кеш, LLM:

```bash
kubectl apply -f k8s/statefulsets/postgres-cluster.yaml   # CloudNativePG: 1 primary + 2 standby
kubectl apply -f k8s/deployments/qdrant.yaml              # Qdrant v1.13.2
kubectl apply -f k8s/deployments/redis.yaml               # Redis 7 (брокер Celery)
kubectl apply -f k8s/deployments/redis-exporter.yaml      # Side-car exporter для redis-метрик
kubectl apply -f k8s/deployments/vllm.yaml                # vLLM main: avibe-gptq-8bit (GPU, S3 download)
kubectl apply -f k8s/deployments/vllm-sql.yaml            # vLLM SQL: Qwen3-8B-AWQ (GPU, S3 download)
```

**3.4. Ожидание готовности инфраструктуры:**

```bash
# PostgreSQL HA: 3 пода поднимаются, настраивается replication (2-3 мин)
kubectl -n agile-assistant wait --for=condition=ready cluster/postgres-cluster --timeout=300s

kubectl -n agile-assistant wait --for=condition=ready pod -l app=qdrant --timeout=120s
kubectl -n agile-assistant wait --for=condition=ready pod -l app=redis --timeout=60s
```

> vLLM не ждём — он загружает модель 10-15 минут при первом запуске (скачивание
> ~6.5GB + компиляция CUDA-графов). При повторных запусках модель берётся из
> PVC.

**3.5. Инициализация данных** — три Job-а выполняются последовательно:

```bash
# Загрузка CSV-данных в PostgreSQL (report_agile_dashboard, report_agile_dashboard_metrics)
kubectl apply -f k8s/jobs/postgres-load-data.yaml
kubectl -n agile-assistant wait --for=condition=complete job/postgres-load-data --timeout=120s
kubectl -n agile-assistant logs job/postgres-load-data

# Alembic миграции (создаёт таблицу tasks для API)
kubectl apply -f k8s/jobs/migrate.yaml
kubectl -n agile-assistant wait --for=condition=complete job/migrate --timeout=120s
kubectl -n agile-assistant logs job/migrate

# Загрузка базы знаний в Qdrant (embedding-модель ~500MB + индексация 82 чанков, ~3-4 мин)
kubectl apply -f k8s/jobs/qdrant-ingest.yaml
kubectl -n agile-assistant wait --for=condition=complete job/qdrant-ingest --timeout=300s
kubectl -n agile-assistant logs job/qdrant-ingest
```

> Все Job-ы имеют `backoffLimit` для автоповтора при ошибках и
> `ttlSecondsAfterFinished: 3600` — автоудаление через 1 час после завершения.

**3.6. Приложение:**

```bash
kubectl apply -f k8s/deployments/api.yaml            # FastAPI (2 реплики, gunicorn + uvicorn)
kubectl apply -f k8s/deployments/celery-worker.yaml   # Celery main worker (embedding-модель, 4Gi RAM)
kubectl apply -f k8s/deployments/celery-judge.yaml    # Celery judge worker (queue=judge, vsellm)
kubectl apply -f k8s/deployments/celery-beat.yaml     # Celery Beat (replicas:1 + Recreate, периодический sync)
kubectl apply -f k8s/deployments/streamlit.yaml        # Streamlit UI
```

**3.7. Ingress** — маршрутизация внешнего трафика:

```bash
kubectl apply -f k8s/ingress.yaml    # 4 Ingress-ресурса: API, static-svg, static-css, UI
```

**3.8. Monitoring (опционально, Prometheus Operator)** — заменяет
`static_configs` из `monitoring/prometheus/prometheus.yml` на ServiceMonitor /
PodMonitor / PrometheusRule CRD. Подробности в
[k8s/monitoring/README.md](k8s/monitoring/README.md):

```bash
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  -f k8s/monitoring/kube-prometheus-stack-values.yaml
kubectl apply -f k8s/monitoring/    # 7 ServiceMonitor + 1 PodMonitor + 1 PrometheusRule
```

### Шаг 4: Проверка

```bash
# Статус всех подов
kubectl -n agile-assistant get pods

# Ожидаемый результат: все поды Running, READY 1/1
# PostgreSQL HA: 3 пода (postgres-cluster-1, -2, -3)
# vLLM может загружаться 10-15 минут при первом запуске (скачивание модели ~6.5GB + компиляция CUDA-графов)
# При последующих запусках модель берётся из PVC (vllm-cache)

# Статус PostgreSQL HA кластера
kubectl -n agile-assistant get cluster postgres-cluster
```

### Шаг 5: Использование

Узнайте IP minikube:

```bash
minikube ip
```

Откройте в браузере (при первом входе появится окно Basic Auth — логин `admin`):

```bash
# Streamlit UI
open http://$(minikube ip)

# Swagger UI
open http://$(minikube ip)/docs
```

Создайте задачу через API (Basic Auth обязателен):

```bash
# Создание задачи
curl -s -u admin:<пароль> -X POST http://$(minikube ip)/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"query": "Расскажи о задаче AL-38787"}'

# Проверка статуса (подставьте task_id)
curl -s -u admin:<пароль> http://$(minikube ip)/api/tasks/<task_id> | python3 -m json.tool
```

### Маршруты Ingress

| Путь              | Сервис           | Особенности                                         |
| ----------------- | ---------------- | --------------------------------------------------- |
| `/api/*`          | `api:8080`       | Prefix `/api` удаляется (rewrite-target)            |
| `/static/*.svg`   | `streamlit:8501` | Content-Type: image/svg+xml (configuration-snippet) |
| `/static/*.css`   | `streamlit:8501` | Content-Type: text/css (configuration-snippet)      |
| `/docs`, `/redoc` | `api:8080`       | Swagger UI                                          |
| `/_stcore`        | `streamlit:8501` | WebSocket-эндпоинт Streamlit                        |
| `/` (default)     | `streamlit:8501` | Streamlit UI                                        |

### Полезные команды

```bash
# Логи конкретного сервиса
kubectl -n agile-assistant logs -l app=vllm --tail=50
kubectl -n agile-assistant logs -l app=celery-worker --tail=50

# Перезапуск деплоймента
kubectl -n agile-assistant rollout restart deployment/api

# Масштабирование API
kubectl -n agile-assistant scale deployment/api --replicas=3

# PostgreSQL HA: проверка кластера
kubectl -n agile-assistant get cluster postgres-cluster

# PostgreSQL HA: тест failover (удалить primary — standby промоутится)
kubectl -n agile-assistant delete pod postgres-cluster-1

# Остановка всего
minikube stop

# Удаление кластера
minikube delete
```

## Guardrails

Трёхуровневая система защиты: regex-фильтр на входе (L1), валидация SQL перед
выполнением (L2), пост-обработка финального ответа (L3). Все три уровня работают
без LLM-вызовов (<1 ms). Детекция off-topic делегирована Supervisor'у через
отдельный `query_type=off_topic` — он и так делает LLM-вызов для классификации
intent, отдельный embedding/NLI-слой был нестабилен на русском и удалён.

Модуль: `hse_prom_prog/agents/guardrails/`. Включение/выключение —
`GUARDRAIL_ENABLED` в `.env` (по умолчанию `True`; при `False` оба
guardrail-узла в workflow пропускаются).

### L1 — TopicGuard (input)

Файл: [topic_guard.py](hse_prom_prog/agents/guardrails/topic_guard.py).
Срабатывает в узле `_input_guardrail_node` до Supervisor'а. Чисто regex — без
LLM, эмбеддингов, порогов.

- **Prompt injection block** — 2 паттерна ловят попытки перехвата роли («ignore
  instructions», «ты теперь не ассистент», «pretend you are …»). При
  срабатывании — `final_response = OFF_TOPIC_RESPONSE`, workflow прекращает
  обработку
- **Whitelist fast-path** — 3 паттерна помечают очевидно-безопасные запросы
  (issue key `AL-123`, приветствие, «что умеешь / help»). Просто ускоряет путь,
  не меняет маршрутизацию
- Остальное → `reason=pass`, Supervisor принимает решение (включая off_topic)

Off-topic (еда, погода, стихи, анекдоты, финансы, перевод…) определяется
Supervisor'ом. Если он классифицирует запрос как `off_topic`, workflow идёт в
`_off_topic_node` → возвращает `OFF_TOPIC_RESPONSE` напрямую, пропуская SQL /
RAG / Response Agent / L3.

### L2 — SQLGuard (tool-level)

Файл: [sql_guard.py](hse_prom_prog/agents/guardrails/sql_guard.py). Вызывается
из `run_query()` — единственного tool'а SQL Agent'а — перед каждым выполнением
SQL. Заменяет наивный `startswith("SELECT")`, который обходится через CTE с
побочными эффектами, подзапросы, stacked queries и скрытые комментарии.

Четыре слоя (fail-fast, от дешёвого к дорогому):

1. **Limits** — пустые и слишком длинные (>2000 символов) запросы
2. **Regex blacklist** — DDL (`DROP/CREATE/ALTER TABLE|INDEX|...`), DML
   (`INSERT INTO`, `UPDATE ... SET`, `DELETE FROM`, `MERGE INTO`), DCL (`GRANT`,
   `REVOKE`), опасные функции (`pg_sleep(`, `dblink(`, `lo_import(`), `COPY` /
   `SET` / `DO $` statement-ы, SQL-комментарии (`--`, `/* */`)
3. **AST (sqlglot)** — парсит запрос, требует корень `SELECT`, проверяет что в
   дереве нет mutation-узлов
   (`Insert/Delete/Update/Drop/Create/Alter/Merge/ TruncateTable`) и что
   упомянуты только whitelist-таблицы (`report_agile_dashboard`,
   `report_agile_dashboard_metrics`). Ловит stacked queries через
   `len(statements) > 1`
4. **Complexity** — максимум 5 JOIN на запрос

Graceful degradation: если `sqlglot` не импортируется, AST-слой пропускается —
работает regex-only режим. `run_query()` при блокировке возвращает модели
сообщение об ошибке с указанием слоя и причины, SQL Agent может попробовать
переписать запрос.

### L3 — ResponseGuard (output)

Файл: [response_guard.py](hse_prom_prog/agents/guardrails/response_guard.py).
Срабатывает в `_output_guardrail_node` после Response Agent'а. Работает в двух
режимах:

- **BLOCK** (критичные нарушения) → весь ответ заменяется на `BLOCKED_RESPONSE`
  («Извините, не удалось сформировать корректный ответ…»)
- **sanitize** → проблемный фрагмент удаляется in-place с маркером-заменой

Шесть проверок:

| Проверка            | Режим    | Что ловит                                                                                                                         |
| ------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `length_empty`      | BLOCK    | ответ короче 10 символов                                                                                                          |
| `length_overflow`   | sanitize | ответ длиннее 5000 символов                                                                                                       |
| `language`          | sanitize | доля кириллицы < 25% (англоязычный ответ на русский запрос)                                                                       |
| `sql_leak`          | sanitize | SQL-keyword + whitelist-таблица в тексте → `[SQL запрос скрыт]`                                                                   |
| `traceback`         | BLOCK    | Python traceback / `raise ...Error` / `File "...", line ...`                                                                      |
| `hallucinated_urls` | sanitize | URL / email, которых нет в переданном RAG-контексте → `[ссылка удалена]`                                                          |
| `internal_leak`     | sanitize | connection strings (`qdrant://qdrant:6333`), `pg_*`, env vars (`QDRANT_URL`), имена внутренних таблиц → `[внутренняя информация]` |

Для `OFF_TOPIC_RESPONSE` проверки пропускаются — заготовленный текст уже
заведомо чист.

### Тестирование guards

Отдельного eval-пайплайна под guards нет — они покрываются существующими
golden-датасетами (частично, через регрессионные сценарии) и прямыми
unit-снипетами.

**Supervisor eval (81 кейс) — покрытие L1 / off-topic:**

| Категория            | Кейсов | Что проверяет                                                                                                                                       |
| -------------------- | -----: | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task_regex`         |     10 | fast-path по issue key (часть whitelist TopicGuard через pass-through)                                                                              |
| `tasks_filter`       |     10 | SQL-маршрутизация, множественное число → `tasks_filter`                                                                                             |
| `metric`             |     10 | метрики команд / спринтов                                                                                                                           |
| `rag`                |     10 | теоретические вопросы без привязки к команде                                                                                                        |
| `hybrid`             |      8 | данные + рекомендации                                                                                                                               |
| `simple`             |      6 | приветствия / мета-вопросы — проверка carve-out от off_topic                                                                                        |
| `adversarial`        |     12 | prompt-injection, `DROP TABLE ...`, пустой запрос, неоднозначности (defense-in-depth для L1 + L2 через Supervisor's `_post_process_classification`) |
| `off_topic`          |      9 | детекция off-topic (анекдоты, погода, стихи, еда, финансы, перевод, здоровье)                                                                       |
| `off_topic_boundary` |      6 | «почти off_topic», но относится к Agile (мотивация, планирование спринта) — НЕ должно блокироваться                                                 |

Deploy gate в eval runner'е (формализован в коде):

- off_topic caught: ≥ 8/9 (допустим 1 miss)
- boundary NOT off_topic: = 6/6 (zero tolerance на false positive)
- baseline non-off-topic routing: ≥ 65/66 (допустима 1 регрессия)

**Response eval (40 кейсов) — регрессионное покрытие L3:**

Eval вызывает `ResponseAgent.process()` напрямую и имеет свои rule-based
проверки (language: ≥30% кириллицы, must_contain / must_not_contain, длина,
наличие источников) — они частично перекрываются с L3 и работают как регрессия
на случай, если L3 начнёт ложно блочить / санитизировать валидные ответы.
Полноценное тестирование L3 — через E2E запуск workflow (см. ниже).

**SQL eval (46 кейсов) — регрессионное покрытие L2:**

Категория `negative` (3) тестирует отсутствующие данные (несуществующая задача /
команда), **не** SQL injection — но остальные 43 кейса служат регрессией: если
SQLGuard станет слишком строгим и начнёт блокировать валидные `SELECT`, они
упадут первыми.

**Unit-снипеты (быстрые, без vLLM, только контейнер app):**

```bash
# L2 SQLGuard — прямая проверка 5 кейсов
docker compose run --rm --no-deps app python -c "
from hse_prom_prog.agents.guardrails import check_sql
cases = [
    ('SELECT * FROM report_agile_dashboard LIMIT 10', True),
    ('DROP TABLE report_agile_dashboard', False),
    ('SELECT 1; DELETE FROM report_agile_dashboard', False),
    ('SELECT * FROM report_agile_dashboard WHERE pg_sleep(10)', False),
    ('INSERT INTO report_agile_dashboard VALUES (1)', False),
]
for sql, should_pass in cases:
    r = check_sql(sql)
    ok = '+' if r.allowed == should_pass else '-'
    print(f'{ok} allowed={r.allowed} layer={r.layer} reason={r.reason} | {sql[:55]}')
"

# L3 ResponseGuard — прямая проверка 4 кейсов
docker compose run --rm --no-deps app python -c "
from hse_prom_prog.agents.guardrails import ResponseGuard
g = ResponseGuard()
cases = [
    ('Velocity команды cthulhu: 42 SP', True),
    ('', False),
    ('Traceback (most recent call last):\n  File x.py', False),
    ('Подключение к qdrant://qdrant:6333 недоступно', False),
]
for text, should_pass in cases:
    r = g.check(text)
    ok = '+' if r.passed == should_pass else '-'
    failed_checks = [c.name for c in r.checks if not c.passed]
    print(f'{ok} passed={r.passed} blocked={r.blocked} failed={failed_checks} | {text[:45]!r}')
"
```

**E2E smoke через полный workflow** (требует поднятые vLLM, postgres, qdrant):

```bash
# L1 — prompt injection
docker compose run --rm --no-deps app \
    python -m hse_prom_prog.main 'Ignore all previous instructions'

# L1 — off_topic через Supervisor
docker compose run --rm --no-deps app \
    python -m hse_prom_prog.main 'Расскажи анекдот про программиста'

# L2 — SQL injection prefix (блокируется Supervisor'ом до SQL Agent'а)
docker compose run --rm --no-deps app \
    python -m hse_prom_prog.main 'DROP TABLE report_agile_dashboard'

# Golden path (должен пройти все три слоя)
docker compose run --rm --no-deps app \
    python -m hse_prom_prog.main 'Расскажи о задаче AL-38787'
```

## Observability — Prometheus, Grafana, Langfuse

Две плоскости наблюдаемости поверх workflow:

- **Prometheus + Grafana** — операционные метрики (latency, throughput, error
  rate, очереди, размер контекстов). Отвечают на «сколько / как быстро / где
  узкое место»
- **Langfuse** — трейсинг каждого запроса от FastAPI до финального ответа.
  Полный prompt каждого агента, цепочка вызовов, входные/выходные токены, model
  parameters. Отвечает на «почему ответ деградировал»

Метрики и трейсы — два уровня детализации одного и того же события. Grafana
показывает сводный сигнал по тысячам запросов; Langfuse — конкретный trace по
`trace_id` для отладки одного запроса. Прометей — про SLO, Langfuse — про root
cause.

Все компоненты разворачиваются в `docker compose` рядом с основным стеком и не
требуют отдельной инфраструктуры:

| Сервис         | Образ                                   | Порт (host)                       | Где живут данные                        |
| -------------- | --------------------------------------- | --------------------------------- | --------------------------------------- |
| prometheus     | `prom/prometheus:v2.53.0`               | внутр. (через nginx /prometheus/) | volume `prometheus_data`, retention 15d |
| grafana        | `grafana/grafana:11.1.0`                | внутр. (через nginx /grafana/)    | volume `grafana_data`                   |
| langfuse       | `langfuse/langfuse:2`                   | `localhost:3001`                  | PostgreSQL `langfuse_db`                |
| pg-exporter    | `prometheuscommunity/postgres-exporter` | внутренний                        | —                                       |
| redis-exporter | `oliver006/redis_exporter`              | внутренний                        | —                                       |

> Prometheus и Grafana проксируются через nginx с gzip-сжатием. Прямые порты
> `9090` / `3000` намеренно закрыты от внешнего мира — публичная точка одна
> (порт 80). Langfuse временно открыт на `:3001`, потому что его OAuth-эндпоинт
> требует совпадения `NEXTAUTH_URL` с тем, что вводит пользователь в браузере.

### Метрики Prometheus

Реестр метрик собран в [hse_prom_prog/metrics.py](hse_prom_prog/metrics.py). Все
имена живут в неймспейсе `agile_assistant_*`. ~50 кастомных метрик разделены на
12 секций.

**Pipeline (workflow_task)**:

| Метрика                       | Тип       | Лейблы       | Что показывает                                     |
| ----------------------------- | --------- | ------------ | -------------------------------------------------- |
| `pipeline_duration_seconds`   | Histogram | `query_type` | E2E latency Celery-задачи от старта до записи в DB |
| `pipeline_queue_wait_seconds` | Histogram | —            | Время ожидания в Redis-очереди (только attempt 0)  |
| `tasks_total`                 | Counter   | `status`     | Терминальные исходы: COMPLETED / FAILED            |
| `tasks_in_progress`           | Gauge     | —            | Сколько задач сейчас в работе                      |

**Celery worker** (через сигналы `task_prerun` / `task_postrun` / `task_failure`
/ `task_retry`):

| Метрика                        | Тип       | Лейблы              | Что показывает                                        |
| ------------------------------ | --------- | ------------------- | ----------------------------------------------------- |
| `celery_task_duration_seconds` | Histogram | `task_name`         | Длительность Celery-задачи (включая memory_tasks)     |
| `celery_tasks_total`           | Counter   | `task_name, status` | success / failure / retry per task                    |
| `celery_active_tasks`          | Gauge     | —                   | Активные задачи воркера                               |
| `celery_queue_length`          | Gauge     | —                   | LLEN `celery` в Redis (custom collector с lazy-Redis) |

**Per-agent durations**:

| Метрика                            | Тип       | Лейблы               | Что показывает                                               |
| ---------------------------------- | --------- | -------------------- | ------------------------------------------------------------ |
| `supervisor_duration_seconds`      | Histogram | `path`               | fast / slow path — даёт понять долю LLM-вызовов              |
| `supervisor_classifications_total` | Counter   | `intent, query_type` | confusion-матрица для роутинга                               |
| `supervisor_fast_path_total`       | Counter   | —                    | сколько запросов поймал regex до LLM                         |
| `supervisor_llm_calls_total`       | Counter   | —                    | slow-path вызовы                                             |
| `supervisor_parse_errors_total`    | Counter   | —                    | LLM вернул невалидный JSON                                   |
| `sql_agent_duration_seconds`       | Histogram | `intent`             | E2E LangGraph SQL-цикл                                       |
| `sql_query_duration_seconds`       | Histogram | —                    | Время одного `db.execute_query`                              |
| `sql_queries_total`                | Counter   | `status`             | success / error / blocked (через L2 SQLGuard)                |
| `sql_result_rows`                  | Histogram | —                    | Распределение размера результата                             |
| `sql_retries_total`                | Counter   | —                    | retries из-за SQL error или semantic mismatch                |
| `sql_empty_results_total`          | Counter   | `intent`             | row_count = 0 без error — индикатор плохого фильтра          |
| `rag_agent_duration_seconds`       | Histogram | —                    | E2E RAG (retrieve → rerank → LLM)                            |
| `rag_retrieval_duration_seconds`   | Histogram | `search_type`        | Только Qdrant query                                          |
| `rag_chunks_retrieved`             | Histogram | `search_type`        | Сколько кандидатов вернул retriever                          |
| `rag_top_score`                    | Histogram | `search_type`        | Score лучшего чанка — proxy на качество retrieval            |
| `rag_chunks_after_reranker`        | Histogram | —                    | Сколько чанков выжило после фильтра по threshold             |
| `rag_reranker_duration_seconds`    | Histogram | —                    | Время cross-encoder reranking                                |
| `rag_requests_total`               | Counter   | `search_type`        | dense / sparse / hybrid                                      |
| `rag_fallbacks_total`              | Counter   | `from_mode, to_mode` | hybrid → dense fallback при отсутствии sparse vectors        |
| `validator_results_total`          | Counter   | `use_sql, use_rag`   | Куда уходит payload в Response Agent                         |
| `validator_data_missing_total`     | Counter   | `source`             | sql / rag / both — какой источник пустой                     |
| `response_duration_seconds`        | Histogram | `branch`             | error/simple/sql_task/sql_tasks_filter/sql_metric/rag/hybrid |
| `response_length_tokens`           | Histogram | `branch`             | Размер финального ответа в токенах                           |
| `response_truncated_total`         | Counter   | —                    | Список задач обрезан (>20 строк)                             |
| `response_llm_timeouts_total`      | Counter   | —                    | Таймауты LLM в любой ветке Response Agent                    |

**Guardrails L1/L2/L3**:

| Метрика                            | Тип     | Лейблы            | Что показывает                                           |
| ---------------------------------- | ------- | ----------------- | -------------------------------------------------------- |
| `guardrail_l1_results_total`       | Counter | `reason`          | injection_blocked / whitelist_fast_path / pass           |
| `guardrail_l2_results_total`       | Counter | `allowed, layer`  | Какой слой SQLGuard сработал (limits / regex / ast / ok) |
| `guardrail_l3_results_total`       | Counter | `passed, blocked` | Прошёл / заблокирован / санитизирован                    |
| `guardrail_l3_checks_failed_total` | Counter | `check_name`      | sql_leak / traceback / hallucinated_urls / internal_leak |

**Entity sanitizer**:

| Метрика                                | Тип     | Лейблы        | Что показывает                                              |
| -------------------------------------- | ------- | ------------- | ----------------------------------------------------------- |
| `sanitizer_corrections_total`          | Counter | `layer`       | Какой слой 1-7 что-то исправил                              |
| `sanitizer_anaphora_carries_total`     | Counter | `entity_type` | Какие поля карри-форвардятся чаще (team/sprint/cluster/...) |
| `sanitizer_fallback_extractions_total` | Counter | `field`       | Какое enum-поле LLM пропускает (layer 7 ловит)              |

**Memory layer**:

| Метрика                          | Тип       | Лейблы   | Что показывает                                            |
| -------------------------------- | --------- | -------- | --------------------------------------------------------- |
| `memory_context_tokens`          | Histogram | —        | Сколько токенов история + summary внесли в промпт         |
| `memory_context_turns`           | Histogram | —        | Сколько ходов вошло в окно (после sliding window)         |
| `memory_session_rotations_total` | Counter   | `reason` | inactivity / explicit_close — частота auto/manual ротации |
| `memory_summarizations_total`    | Counter   | —        | Запуски Celery-таски `summarize_session`                  |
| `memory_profile_updates_total`   | Counter   | —        | Запуски Celery-таски `update_profile_async`               |

**LLM-as-a-Judge (Phase 4)**:

| Метрика                   | Тип     | Лейблы      | Что показывает                                                                                                                                             |
| ------------------------- | ------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `judge_criterion_score`   | Gauge   | `criterion` | Score последнего evaluation per criterion (0/1): `practicality`, `language_quality`, `text_cleanliness`, `agile_correctness`, `completeness`, `politeness` |
| `judge_weighted_total`    | Gauge   | —           | Weighted total score последнего evaluation (0.0–1.0)                                                                                                       |
| `judge_evaluations_total` | Counter | `status`    | Терминальные исходы судьи: `success` / `parse_error` / `api_error`                                                                                         |

**Data sync (Phase 5, Celery Beat)**:

| Метрика                       | Тип       | Лейблы           | Что показывает                                                              |
| ----------------------------- | --------- | ---------------- | --------------------------------------------------------------------------- |
| `data_sync_timestamp_seconds` | Gauge     | `source`         | Unix-timestamp последнего успешного запуска (`jira_csv` / `knowledge_base`) |
| `data_sync_total`             | Counter   | `source, status` | Запуски по источнику и исходу (`success` / `failure`)                       |
| `data_sync_duration_seconds`  | Histogram | `source`         | Длительность одного прогона (CSV: секунды, KB: минуты)                      |
| `data_sync_rows`              | Gauge     | `source`         | Кол-во строк / чанков после успешного sync                                  |

**FastAPI HTTP** (через `prometheus-fastapi-instrumentator`): автоматические
метрики `http_requests_total`, `http_request_duration_seconds` с лейблами
`method`, `handler`, `status`. `/metrics` исключён из инструментирования, чтобы
Prometheus сам себя не считал.

### Дашборды Grafana

Пять JSON-дашбордов в
[monitoring/grafana/dashboards/](monitoring/grafana/dashboards/), автоматически
провижионятся при запуске Grafana через `provisioning/dashboards/` (см.
[monitoring/grafana/provisioning/](monitoring/grafana/provisioning/)).

| Дашборд                                        | UID                    | Что показывает                                                                                                                               |
| ---------------------------------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Agile Assistant — Overview**                 | `agile-overview`       | E2E latency p50/p95/p99, throughput RPS, error rate, queue wait, in-progress, query_type breakdown, data freshness (Jira CSV / KB last sync) |
| **Agile Assistant — Infrastructure**           | `agile-infrastructure` | PostgreSQL connections / TPS / replication lag, Redis ops/sec, Qdrant healthcheck, vLLM KV-cache, GPU                                        |
| **Agile Assistant — Agents Deep Dive**         | `agile-agents`         | Per-agent latency, SQL retries / empty results, RAG retrieval / reranker / fallbacks, response branches                                      |
| **Agile Assistant — Guardrails & Safety**      | `agile-guardrails`     | L1 injection blocks, L2 layer breakdown, L3 failed checks, sanitizer corrections by layer, memory rotations                                  |
| **Agile Assistant — Quality (LLM-as-a-Judge)** | `agile-quality`        | Weighted total quality score, per-criterion scores (1h / 24h avg), judge-evaluation outcomes, API errors                                     |

Все дашборды настроены на datasource `prometheus`. Разворачиваются автоматически
после `docker compose up -d grafana` — провижионинг провалидирует dashboards и
зальёт их в папку «Agile Assistant».

### Алерты Prometheus

Все правила в
[monitoring/prometheus/alerts.yml](monitoring/prometheus/alerts.yml). 18 правил
в четырёх группах (severity: warning / critical):

**`agile_assistant_pipeline`** — общая работоспособность pipeline-а:

| Alert                | Условие                                          | Severity |
| -------------------- | ------------------------------------------------ | -------- |
| `HighE2ELatency`     | p95 `pipeline_duration_seconds` > 30s, `for: 5m` | warning  |
| `HighErrorRate`      | доля FAILED задач > 10% за 5 мин, `for: 5m`      | critical |
| `CeleryQueueBacklog` | `celery_queue_length` > 20, `for: 5m`            | warning  |

**`infrastructure`** — состояние внешних зависимостей:

| Alert                      | Условие                                                                           | Severity |
| -------------------------- | --------------------------------------------------------------------------------- | -------- |
| `VLLMMainQueueOverflow`    | `vllm:num_requests_waiting{job="vllm-main"}` > 5, `for: 3m`                       | warning  |
| `VLLMSQLQueueOverflow`     | `vllm:num_requests_waiting{job="vllm-sql"}` > 3, `for: 3m`                        | warning  |
| `VLLMKVCacheExhausted`     | `vllm:gpu_cache_usage_perc` > 0.95, `for: 2m`                                     | critical |
| `PostgreSQLReplicationLag` | `pg_replication_lag_seconds` > 5 (k8s/CNPG: `cnpg_pg_replication_lag`), `for: 1m` | critical |
| `QdrantDown`               | `up{job="qdrant"} == 0`, `for: 1m`                                                | critical |
| `SanitizerFallbackHigh`    | `rate(sanitizer_corrections_total{layer="7_fallback"}[5m])` > 0.5, `for: 10m`     | warning  |

**`agile_assistant_quality`** — LLM-as-a-Judge (Phase 4); 0/1 gauges усредняются
через `avg_over_time` за час, `for: 15m` глушит одиночные выбросы:

| Alert                       | Условие                                                                         | Severity |
| --------------------------- | ------------------------------------------------------------------------------- | -------- |
| `JudgeQualityDrop`          | `avg_over_time(judge_weighted_total[1h])` < 0.7, `for: 15m`                     | warning  |
| `JudgeAgileCorrectnessDrop` | `avg_over_time(judge_criterion_score{criterion="agile_correctness"}[1h])` < 0.6 | warning  |
| `JudgePracticalityDrop`     | `avg_over_time(judge_criterion_score{criterion="practicality"}[1h])` < 0.6      | warning  |
| `JudgeLanguageQualityDrop`  | `avg_over_time(judge_criterion_score{criterion="language_quality"}[1h])` < 0.6  | warning  |
| `JudgeUnavailable`          | доля `judge_evaluations_total{status="api_error"}` за 30 мин > 50%, `for: 10m`  | warning  |

**`agile_assistant_data_freshness`** — Phase 5, Celery Beat sync:

| Alert                | Условие                                                                                | Severity |
| -------------------- | -------------------------------------------------------------------------------------- | -------- |
| `JiraDataStale`      | `time() - max(data_sync_timestamp_seconds{source="jira_csv"})` > 12h, `for: 10m`       | warning  |
| `KnowledgeBaseStale` | `time() - max(data_sync_timestamp_seconds{source="knowledge_base"})` > 48h, `for: 30m` | warning  |
| `DataSyncFailing`    | `increase(data_sync_total{status="failure"}[1h])` >= 2                                 | warning  |
| `CeleryBeatDown`     | `time() - max(data_sync_timestamp_seconds)` > 30h (нет успехов ни по одному источнику) | critical |

Алерты не отправляются в Alertmanager — он не входит в Phase 1 деплой.
Срабатывания видны на странице `/prometheus/alerts` и могут быть экспортированы
в Slack/Telegram при подключении Alertmanager. В k8s те же 18 правил живут как
PrometheusRule CRD в
[k8s/monitoring/prometheus-rules.yaml](k8s/monitoring/prometheus-rules.yaml).

### Трейсинг Langfuse

Реализация — [hse_prom_prog/tracing.py](hse_prom_prog/tracing.py). Один
singleton-клиент инициализируется при импорте, остальные модули используют
реэкспорт `observe` / `langfuse_context` оттуда. Если SDK не установлен или
выключен через `LANGFUSE_ENABLED=false` — `observe` становится no-op
декоратором, `langfuse_context` — заглушкой с пустыми методами; код агентов
работает идентично.

**Корневой trace** создаётся в `tasks/workflow_task.py` императивно (не через
`@observe`), потому что нужно явно установить `user_id`, `session_id`,
`input/output` и сделать `flush()` в `finally` блоке:

```
trace = langfuse_client.trace(
    name="workflow",
    user_id=external_id,
    session_id=conversation_id,
    input={"query": query},
    metadata={"task_id": task_id, "celery_retry": self.request.retries},
)
langfuse_context.update_current_trace(trace_id=trace.id)
```

После этого все `@observe(...)` в агентах автоматически становятся child-spans
через Python `contextvars`. Celery worker запущен с `--pool=threads` — каждая
задача в своём потоке, контексты не смешиваются.

**Покрытие spans / generations**:

| Точка                             | Тип        | Что захватывается                                                |
| --------------------------------- | ---------- | ---------------------------------------------------------------- |
| `workflow` (root)                 | trace      | user_id, session_id, input query, output final_response          |
| `guardrail_l1`                    | span       | input query, output reason (pass/whitelist/injection_blocked)    |
| `memory_context_build`            | span       | conversation_id, token_budget → tokens_used, turns_included      |
| `supervisor`                      | span       | input query, has_history, has_profile → intent, query_type, path |
| `entity_sanitizer`                | span       | raw_entities → sanitized_entities                                |
| `sql_agent`                       | span       | query, intent, entities → sql, rows_count, error                 |
| `sql_llm_call` (внутри sql_agent) | generation | model=Qwen3-8B-AWQ, tools, prompt → tool_calls + usage           |
| `run_query` (внутри sql_agent)    | span       | sql → status (success/error/blocked), rows_count, duration_ms    |
| `guardrail_l2` (внутри run_query) | span       | sql → allowed, layer, reason                                     |
| `rag_agent`                       | span       | query → sources, response_length                                 |
| `retrieval` (внутри rag_agent)    | span       | query, search_type, k → chunks_count, preview_sources            |
| `reranker` (внутри rag_agent)     | span       | chunks_count, threshold → chunks_after, scores                   |
| `validator`                       | span       | query_type, sql_ok, rag_ok → use_sql, use_rag, note              |
| `response_agent`                  | span       | branch, intent → response_preview (500 chars), response_length   |
| `llm_call` (LLMClient.invoke)     | generation | model=avibe-gptq-8bit, prompt, params → output + usage           |
| `guardrail_l3`                    | span       | response_length → passed, blocked, failed_checks                 |

LLM-вызовы помечены как `as_type="generation"` — Langfuse выделяет их в
отдельную сущность с usage tokens, model parameters и поддержкой LLM-as-a-judge
scoring. Phase 4 использует это: `judge_task` пишет 6 score'ов (practicality /
language_quality / text_cleanliness / agile_correctness / completeness /
politeness) обратно в исходный trace через `langfuse_client.score`, после чего
они видны и в Langfuse, и в Grafana (дашборд Quality).

**Что не пишется в Langfuse**:

- Полный текст RAG-чанков и SQL-результатов — только id / source / row_count.
  Иначе storage раздувается на длинных диалогах
- `final_response` дублируется только preview (500 символов) на span'е Response
  Agent — полный текст уже на root trace
- Async memory tasks (`summarize_session`, `update_profile_async`) — они
  выполняются после ответа пользователю и не являются частью основного trace.
  Если понадобится их трейсить отдельно — нужно создавать новый trace
- HTTP-вызовы FastAPI — за это отвечает Prometheus (HTTP middleware), Langfuse
  на этом уровне переплюс

### Запуск observability-стека

Phase 1+2 (Prometheus + Grafana) запускаются вместе с основным стеком
автоматически:

```bash
docker compose up -d prometheus grafana pg-exporter redis-exporter
```

После этого:

- Prometheus UI: `http://localhost/prometheus/` (через nginx, gzip)
- Grafana UI: `http://localhost/grafana/` (логин: `admin`, пароль из
  `GRAFANA_PASSWORD`)
- Метрики FastAPI: `http://localhost/api/metrics` — заблокировано nginx'ом
  (404), Prometheus скрейпит напрямую через docker network
- Метрики Celery worker: `celery-worker:9100/metrics` — internal-only

**Phase 3 (Langfuse)** — отдельный шаг, потому что нужны API-ключи из UI:

```bash
# 1. Если postgres_data уже создан — создать БД для Langfuse вручную
docker compose exec postgres \
    psql -U hse_user -c "CREATE DATABASE langfuse_db;"

# 2. Запустить Langfuse
docker compose up -d langfuse
docker compose ps langfuse  # ждём healthy (60-90 сек)

# 3. Открыть http://localhost:3001 → зарегистрировать admin аккаунт
#    (первый пользователь автоматически админ)

# 4. Settings → Projects → New Project → "Agile Assistant"

# 5. Settings → API Keys → Create New Key
#    Скопировать pk-lf-... и sk-lf-... в .env:
#      LANGFUSE_PUBLIC_KEY=pk-lf-...
#      LANGFUSE_SECRET_KEY=sk-lf-...

# 6. Перезапустить api и celery-worker, чтобы tracing.py подхватил ключи
docker compose restart api celery-worker

# 7. Отправить тестовый запрос
curl -s -X POST http://localhost/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"query": "Расскажи о задаче AL-38787"}'

# 8. В Langfuse UI → Traces должен появиться trace с деревом spans
```

> При первом запуске volume `postgres_data` ещё пуст — `init-langfuse.sql` в
> `database/init-langfuse.sql` выполнится автоматически и создаст `langfuse_db`.
> Шаг 1 нужен только если volume уже существовал до Phase 3.

**Отключение Langfuse без удаления кода**:

```bash
# В .env:
LANGFUSE_ENABLED=false

docker compose restart api celery-worker
```

Все `@observe`-декораторы становятся no-op, метрики Prometheus продолжают
писаться. Полезно при отладке проблем с Langfuse-сервером или при локальной
разработке без отдельного контейнера.

## Оценка агентов

Модуль `eval/` содержит четыре независимых eval-пайплайна — по одному для
каждого LLM-агента. Каждый использует свой golden dataset и свои rule-based или
LLM-based метрики, результаты пишутся в
`eval/results/{experiment}_{timestamp}.json`.

| Eval           | Dataset (кейсов)                      | Метрики                                                                                                                        |
| -------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| RAG-пайплайн   | `golden_dataset.json` (41)            | RAGAS (GPT-5.2 as judge): context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness                |
| SQL Agent      | `sql_golden_dataset.json` (46)        | Rule-based: exact_match_fields, row_count_exact, value_exact, exact_match_grouped                                              |
| Supervisor     | `supervisor_golden_dataset.json` (81) | Routing accuracy, intent match, entity match (soft substring), confusion matrix, deploy gate (off_topic / boundary / baseline) |
| Response Agent | `response_golden_dataset.json` (40)   | Rule-based: must_contain / must_contain_any / must_not_contain / language / length / sources                                   |

### RAG-пайплайн (RAGAS)

Оценка качества RAG-пайплайна через [RAGAS](https://docs.ragas.io/). В качестве
LLM-as-judge используется GPT-5.2 через OpenAI-compatible API (vsellm).

#### Метрики

| Метрика              | Что оценивает                                | Категория  |
| -------------------- | -------------------------------------------- | ---------- |
| `context_precision`  | Точность найденных чанков                    | Retrieval  |
| `context_recall`     | Полнота найденных чанков                     | Retrieval  |
| `faithfulness`       | Верность ответа контексту (без галлюцинаций) | Generation |
| `answer_relevancy`   | Релевантность ответа вопросу                 | Generation |
| `answer_correctness` | Корректность ответа (vs ground truth)        | End-to-end |

#### Golden Dataset

Файл `eval/golden_dataset.json` содержит 41 вопрос с эталонными ответами,
составленными строго по документам из `knowledge_base/`:

| Категория | Вопросов | Описание                                                    |
| --------- | -------- | ----------------------------------------------------------- |
| metrics   | 25       | Вопросы по описаниям метрик (done_total, scope_drop и т.д.) |
| agile     | 6        | Agile-практики, дашборды                                    |
| cross_doc | 5        | Вопросы, требующие информации из нескольких документов      |
| negative  | 5        | Вопросы, на которые в базе знаний **нет** ответа            |

Негативные примеры прогоняются через пайплайн, но исключаются из RAGAS-оценки
(нет ground truth для сравнения).

#### Запуск оценки

**Через Docker Compose** (рекомендуемый способ — всё поднимается автоматически):

```bash
# 1. Убедитесь, что инфраструктура запущена и база знаний загружена
docker compose up -d postgres qdrant redis vllm
docker compose run --rm app python -m hse_prom_prog.rag.ingest

# 2. Запуск baseline-оценки
docker compose run --rm \
  -e VSELLM_API_KEY=${VSELLM_API_KEY} \
  -e VSELLM_BASE_URL=${VSELLM_BASE_URL} \
  app python -m eval.run_eval --experiment baseline

# 3. Результат сохранится в eval/results/baseline_<timestamp>.json
```

**Локально** (Poetry, если инфраструктура запущена отдельно):

```bash
# Запуск с дефолтным именем эксперимента (baseline)
poetry run python -m eval.run_eval

# Запуск с пользовательским именем
poetry run python -m eval.run_eval --experiment semantic_v2
```

> **Baseline**: первый запуск `--experiment baseline` фиксирует текущие
> параметры пайплайна (модель, chunk_size, overlap, top_k) и метрики. Все
> последующие эксперименты сравниваются с ним через `compare.py`.

Скрипт выполняет:

1. Загружает `golden_dataset.json`
2. Прогоняет каждый вопрос через RAG-пайплайн (retrieve → rerank → generate)
3. Вычисляет RAGAS-метрики (LLM-as-judge: GPT-5.2 через vsellm)
4. Сохраняет результаты в `eval/results/{experiment}_{timestamp}.json`
5. Выводит сводную таблицу в консоль

**Примеры запуска экспериментов с разными режимами поиска:**

```bash
# Dense search (по умолчанию)
docker compose run --rm \
  -e RERANKER_ENABLED=false \
  app python -m eval.run_eval --experiment dense_baseline

# Sparse search (BM25 only)
docker compose run --rm \
  -e SEARCH_TYPE=sparse \
  -e RERANKER_ENABLED=false \
  app python -m eval.run_eval --experiment sparse_bm25

# Hybrid search (dense + BM25, RRF fusion)
docker compose run --rm \
  -e SEARCH_TYPE=hybrid \
  -e RERANKER_ENABLED=false \
  app python -m eval.run_eval --experiment hybrid_rrf

# Hybrid + reranker
docker compose run --rm \
  -e SEARCH_TYPE=hybrid \
  app python -m eval.run_eval --experiment hybrid_reranker
```

Формат результата:

```json
{
  "experiment": "hybrid_rrf",
  "timestamp": "20260328_120000",
  "config": {
    "vllm_model": "/models/avibe-gptq-8bit",
    "embedding_model": "intfloat/multilingual-e5-base",
    "chunk_size": 500,
    "chunk_overlap": 200,
    "search_type": "hybrid",
    "rrf_k": 60,
    "retriever_initial_k": 20,
    "retriever_top_k": 4,
    "reranker_enabled": true,
    "reranker_model": "BAAI/bge-reranker-v2-m3",
    "reranker_threshold": 0.01,
    "reranker_top_n": 4,
    "total_chunks": 82,
    "avg_chunk_chars": 650,
    "avg_retrieval_time_s": 0.05
  },
  "aggregate": {
    "context_precision": 0.82,
    "context_recall": 0.72,
    "faithfulness": 0.91,
    "answer_relevancy": 0.87,
    "answer_correctness": 0.75,
    "latency_avg_s": 2.5,
    "latency_median_s": 2.1,
    "latency_p95_s": 4.3
  },
  "per_question": [...]
}
```

**Требования**: переменные `VSELLM_API_KEY` и `VSELLM_BASE_URL` должны быть
заданы в `.env` (см. [Конфигурация](#конфигурация)). Qdrant должен быть запущен
с загруженной базой знаний.

### SQL Agent

Оценивает качество text-to-SQL: сгенерированный запрос, количество строк,
ключевые поля.

**Датасет** `eval/sql_golden_dataset.json` (46 кейсов):

| Категория      | Кейсов | Пример                                      |
| -------------- | ------ | ------------------------------------------- |
| `single_task`  | 2      | «Расскажи о задаче AL-38787»                |
| `tasks_filter` | 7      | «Все задачи команды cthulhu»                |
| `metric`       | 6      | «Velocity команды linehaul»                 |
| `aggregation`  | 20     | «Сколько story points у команды cthulhu?»   |
| `comparative`  | 7      | «У какой команды самый большой scope drop?» |
| `cross_table`  | 1      | «Scope drop + Cancelled задачи» (2 таблицы) |
| `negative`     | 3      | Задача / команда / фильтр без результатов   |

**Стратегии проверки** (per-case `eval_strategy`): `exact_match_fields`,
`row_count_exact`, `value_exact`, `value_approx`, `exact_match_rows`,
`exact_match_grouped`, `composite`.

**Запуск:**

```bash
docker compose up -d postgres vllm-sql
docker compose run --rm --no-deps app \
  python -m eval.run_sql_eval --experiment sql_v1
```

Сохраняет `eval/results/sql_v1_<timestamp>.json`. В консоль — per-category
pass-rate, latency p50/p95, список failures с actual vs expected.

### Supervisor Agent

Оценивает маршрутизацию + извлечение сущностей. Реальный LLM через vLLM + entity
sanitizer с DB-валидацией.

**Датасет** `eval/supervisor_golden_dataset.json` (81 кейс):

| Категория            | Кейсов |
| -------------------- | ------ |
| `task_regex`         | 10     |
| `tasks_filter`       | 10     |
| `metric`             | 10     |
| `rag`                | 10     |
| `hybrid`             | 8      |
| `simple`             | 6      |
| `adversarial`        | 12     |
| `off_topic`          | 9      |
| `off_topic_boundary` | 6      |

**Метрики:**

- **Routing accuracy** — совпал ли `query_type` (sql/rag/hybrid/simple)
- **Intent match** — совпал ли `intent` (task/tasks_filter/metric/general)
- **Entity match** — soft substring по ключевым полям (issue_key, team_name,
  sprint_name, metric_name); список поддерживается для multi-team кейсов
- **Confusion matrix** 4×4 по `query_type`
- **Fast-path rate** — доля запросов с issue_key, где regex поймал до LLM
- **Latency p50/p95** — отдельно для fast-path и slow-path

**Запуск:**

```bash
docker compose up -d postgres vllm
docker compose run --rm --no-deps app \
  python -m eval.run_supervisor_eval --experiment supervisor_v1
```

Текущий baseline: ~98.5% routing, ~93.9% exact match при `temperature=0.0`.

### Response Agent

Оценивает формат финального ответа: обязательные/запрещённые термины, язык,
длина, источники. Не вызывает Supervisor / SQL / RAG — весь входной `state`
(включая `rag_response`, `sql_result`) зафиксирован в датасете.

**Датасет** `eval/response_golden_dataset.json` (40 кейсов):

| Категория          | Кейсов | Покрытие                                                     |
| ------------------ | ------ | ------------------------------------------------------------ |
| `sql_task`         | 6      | одна задача, NULL-поля, 0 rows, длинный summary, Bug, Epic   |
| `sql_tasks_filter` | 6      | малый список, 70+ задач (обрезка), пустой, mixed status      |
| `sql_metric`       | 6      | одна метрика, динамика по 3 спринтам, агрегат, топ-3, пустой |
| `rag`              | 5      | с источниками / без, длинный / короткий, markdown            |
| `hybrid`           | 6      | SQL+RAG, пустой RAG, пустой SQL, оба упали                   |
| `simple`           | 4      | приветствие, «что умеешь», спасибо, мета                     |
| `error`            | 3      | classifier error, SQL execution error, hybrid timeout        |
| `edge`             | 4      | все NULL, 50 задач, пустой query, спецсимволы                |

**Метрики (per-case):**

- `must_contain` — AND-список обязательных подстрок (case-insensitive)
- `must_contain_any` — ИЛИ-группы синонимов (например,
  `[["In Progress", "в процессе", "в работе"]]`)
- `must_not_contain` — запрещённые подстроки (raw SQL, `SELECT`, Traceback)
- `language: ru` — ≥30% букв кириллицы
- `min_length` / `max_length` — границы длины ответа
- `must_have_sources` — наличие/отсутствие блока `**Источники:**`

**Плейсхолдеры** `<<GENERATE: team=cthulhu, limit=70>>` в `sql_result`
раскрываются во время загрузки через SQL-запрос к БД, чтобы кейсы с 70-118
строками не раздували JSON-датасет.

**Запуск:**

```bash
docker compose up -d postgres vllm
docker compose run --rm --no-deps app \
  python -m eval.run_response_eval --experiment response_v1
```

### Сравнение RAG-экспериментов

```bash
# Сравнение двух экспериментов
poetry run python -m eval.compare \
  eval/results/baseline_20260310_143000.json \
  eval/results/semantic_v2_20260310_150000.json

# Сравнение трёх и более
poetry run python -m eval.compare eval/results/*.json
```

Выводит:

- Таблицу метрик с дельтами (зелёный — улучшение, красный — деградация)
- Различия в конфигурациях пайплайна между экспериментами

```
Metric             dense_baseline     hybrid_rrf         delta (last − first)
-----------------  -----------------  -----------------  --------------------
context_precision  0.8200             0.8900             +0.0700
context_recall     0.7200             0.7800             +0.0600
faithfulness       0.9100             0.9300             +0.0200
answer_correctness 0.7500             0.7200             -0.0300

Config differences:
param        dense_baseline    hybrid_rrf
-----------  ----------------  ----------------
search_type  dense             hybrid
rrf_k        None              60
```

## Разработка

### Установка dev-зависимостей

Для разработки необходимо установить дополнительные зависимости:

```bash
poetry install --with dev
```

Это установит:

- `ruff` - линтер и форматтер кода
- `pre-commit` - хуки для git
- `pytest` и `pytest-asyncio` - тестирование
- `pytest-cov` - покрытие кода тестами

### Code Quality

Проект использует `ruff` для линтинга и форматирования:

```bash
# Проверка кода
poetry run ruff check .

# Автоматическое исправление
poetry run ruff check . --fix

# Форматирование
poetry run ruff format .
```

### Pre-commit hooks

```bash
# Установите pre-commit hooks
poetry run pre-commit install

# Запустите вручную
poetry run pre-commit run --all-files
```

### Тестирование

Тесты разнесены на три модуля по слоям:

| Модуль                     | Тестов | Что покрыто                                                                                                                                                                                                      |
| -------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/test_memory.py`     | 63     | TokenEstimator, Truncator, ContextBuilder, ProfileExtractor, MemoryManager, format_history, EntitySanitizer (carry-forward + fallback), Supervisor + context, ResponseAgent + history, лемматизированная анафора |
| `tests/test_workflow.py`   | 69     | LangGraph-агенты и роутинг (Supervisor / SQL / RAG / Validator / Response / Workflow); ряд кейсов сейчас зафиксирован как известно-сломанный baseline и чинится точечно                                          |
| `tests/test_api_memory.py` | —      | FastAPI `/conversations` + inactivity rotation (требует `langchain_qdrant` для импорта workflow_task — запускается из Docker)                                                                                    |

```bash
# Запустите тесты
poetry run pytest tests/

# С подробным выводом
poetry run pytest tests/ -v

# С покрытием кода
poetry run pytest tests/ --cov=hse_prom_prog
```

## Конфигурация

Все настройки управляются через переменные окружения (см.
[.env.example](.env.example)):

### vLLM Configuration

| Переменная                | Описание                           | По умолчанию               |
| ------------------------- | ---------------------------------- | -------------------------- |
| `VLLM_BASE_URL`           | URL vLLM API endpoint              | `http://localhost:8000/v1` |
| `VLLM_MODEL`              | Название модели                    | `/models/avibe-gptq-8bit`  |
| `VLLM_API_KEY`            | API ключ для vLLM                  | `EMPTY`                    |
| `VLLM_TEMPERATURE`        | Temperature для LLM                | `0.0`                      |
| `VLLM_MAX_TOKENS`         | Максимум токенов                   | `600`                      |
| `VLLM_REPETITION_PENALTY` | Штраф за повторы (vLLM extra_body) | `1.1`                      |

### PostgreSQL Configuration

| Переменная          | Описание        | По умолчанию   |
| ------------------- | --------------- | -------------- |
| `POSTGRES_HOST`     | Хост PostgreSQL | `localhost`    |
| `POSTGRES_PORT`     | Порт PostgreSQL | `5432`         |
| `POSTGRES_USER`     | Пользователь БД | `hse_user`     |
| `POSTGRES_PASSWORD` | Пароль БД       | `hse_password` |
| `POSTGRES_DB`       | Название БД     | `hse_jira_db`  |

### Qdrant Configuration

| Переменная               | Описание           | По умолчанию                    |
| ------------------------ | ------------------ | ------------------------------- |
| `QDRANT_URL`             | URL Qdrant сервера | `http://localhost:6333`         |
| `QDRANT_COLLECTION_NAME` | Название коллекции | `business_docs`                 |
| `EMBEDDING_MODEL`        | Модель эмбеддингов | `intfloat/multilingual-e5-base` |

### S3 Configuration

| Переменная              | Описание                    | По умолчанию                      |
| ----------------------- | --------------------------- | --------------------------------- |
| `S3_ENDPOINT`           | S3 endpoint URL             | `https://storage.yandexcloud.net` |
| `S3_KB_BUCKET`          | S3 bucket для базы знаний   | `knowledge-base`                  |
| `S3_KB_PATH`            | Путь внутри bucket для KB   | `knowledge_base`                  |
| `S3_DATA_BUCKET`        | S3 bucket для CSV данных    | `database-agile`                  |
| `S3_DATA_PATH`          | Путь внутри bucket для CSV  | `data`                            |
| `AWS_ACCESS_KEY_ID`     | Ключ доступа Yandex Cloud   | —                                 |
| `AWS_SECRET_ACCESS_KEY` | Секретный ключ Yandex Cloud | —                                 |

### Embedding Configuration

| Переменная               | Описание                                        | По умолчанию |
| ------------------------ | ----------------------------------------------- | ------------ |
| `EMBEDDING_SPARSE_MODEL` | Sparse модель: `None` (BM25) или `BAAI/bge-m3`  | —            |
| `EMBEDDING_DIMENSION`    | Matryoshka-truncation (64/128/256/512/768/1024) | —            |

### Chunking Configuration

| Переменная          | Описание                          | По умолчанию |
| ------------------- | --------------------------------- | ------------ |
| `CHUNK_SIZE`        | Размер чанка для разбиения текста | `500`        |
| `CHUNK_OVERLAP`     | Перекрытие между чанками          | `200`        |
| `MAX_CONTEXT_CHARS` | Макс. символов контекста для LLM  | `4000`       |

### Search Configuration

| Переменная    | Описание                                    | По умолчанию |
| ------------- | ------------------------------------------- | ------------ |
| `SEARCH_TYPE` | Режим поиска: `dense`, `sparse`, `hybrid`   | `dense`      |
| `RRF_K`       | Параметр k для RRF fusion (только `hybrid`) | `60`         |

### Retriever Configuration

| Переменная            | Описание                               | По умолчанию |
| --------------------- | -------------------------------------- | ------------ |
| `RETRIEVER_TOP_K`     | Финальное число чанков (без реранкера) | `4`          |
| `RETRIEVER_INITIAL_K` | Число чанков для первичного извлечения | `20`         |

### Reranker Configuration

| Переменная           | Описание                               | По умолчанию              |
| -------------------- | -------------------------------------- | ------------------------- |
| `RERANKER_ENABLED`   | Включить cross-encoder reranking       | `true`                    |
| `RERANKER_MODEL`     | Модель реранкера                       | `BAAI/bge-reranker-v2-m3` |
| `RERANKER_THRESHOLD` | Минимальный скор для фильтрации чанков | `0.01`                    |
| `RERANKER_TOP_N`     | Число чанков после реранкинга          | `4`                       |

### Redis Configuration

| Переменная       | Описание         | По умолчанию |
| ---------------- | ---------------- | ------------ |
| `REDIS_HOST`     | Хост Redis       | `localhost`  |
| `REDIS_PORT`     | Порт Redis       | `6379`       |
| `REDIS_DB`       | Номер базы Redis | `0`          |
| `REDIS_PASSWORD` | Пароль Redis     | —            |

### Celery Configuration

| Переменная                    | Описание                    | По умолчанию |
| ----------------------------- | --------------------------- | ------------ |
| `CELERY_BROKER_URL`           | URL брокера (авто из Redis) | —            |
| `CELERY_TASK_TIME_LIMIT`      | Hard timeout задачи (сек)   | `600`        |
| `CELERY_TASK_SOFT_TIME_LIMIT` | Soft timeout задачи (сек)   | `300`        |

### FastAPI Configuration

| Переменная     | Описание                 | По умолчанию |
| -------------- | ------------------------ | ------------ |
| `FASTAPI_HOST` | Хост FastAPI сервера     | `0.0.0.0`    |
| `FASTAPI_PORT` | Порт FastAPI сервера     | `8080`       |
| `CORS_ORIGINS` | Разрешённые CORS origins | `*`          |

### VSELLM Configuration (LLM-as-judge)

| Переменная        | Описание                    | По умолчанию               |
| ----------------- | --------------------------- | -------------------------- |
| `VSELLM_API_KEY`  | API ключ для vsellm (RAGAS) | —                          |
| `VSELLM_BASE_URL` | URL vsellm API endpoint     | `https://api.vsellm.ru/v1` |

### Memory Layer Configuration

| Переменная                | Описание                                              | По умолчанию |
| ------------------------- | ----------------------------------------------------- | ------------ |
| `HISTORY_TOKEN_BUDGET`    | Бюджет токенов на `<conversation_history>` в промптах | `1200`       |
| `SESSION_TIMEOUT_MINUTES` | Idle-таймаут для auto-rotation диалога                | `30`         |
| `MAX_CONVERSATION_TURNS`  | Максимум хранимых ходов на один диалог                | `50`         |

### Monitoring (Phase 1)

| Переменная         | Описание                                             | По умолчанию |
| ------------------ | ---------------------------------------------------- | ------------ |
| `GRAFANA_PASSWORD` | Пароль admin-пользователя в Grafana (логин: `admin`) | `admin`      |

### Langfuse Tracing (Phase 3)

| Переменная                 | Описание                                                                   | По умолчанию             |
| -------------------------- | -------------------------------------------------------------------------- | ------------------------ |
| `LANGFUSE_PUBLIC_KEY`      | Public key проекта Langfuse (`pk-lf-...`). Пусто = не отправлять spans     | —                        |
| `LANGFUSE_SECRET_KEY`      | Secret key проекта Langfuse (`sk-lf-...`)                                  | —                        |
| `LANGFUSE_HOST`            | URL Langfuse-сервера (внутри docker-network)                               | `http://langfuse:3000`   |
| `LANGFUSE_ENABLED`         | Master kill-switch для SDK. `false` → `@observe` становится no-op          | `true`                   |
| `LANGFUSE_NEXTAUTH_SECRET` | Секрет NextAuth для UI Langfuse-сервера (для prod: `openssl rand -hex 32`) | `changeme-in-production` |
| `LANGFUSE_SALT`            | Salt для шифрования API-ключей в БД Langfuse                               | `changeme-in-production` |

### Other

| Переменная  | Описание                              | По умолчанию |
| ----------- | ------------------------------------- | ------------ |
| `DEBUG`     | Debug-режим (hot-reload, verbose log) | `false`      |
| `LOG_LEVEL` | Уровень логирования                   | `INFO`       |

## Текущая лучшая конфигурация RAG

Все эксперименты сравниваются с этой конфигурацией:

| Параметр               | Значение                                      |
| ---------------------- | --------------------------------------------- |
| `SEARCH_TYPE`          | `dense` (multilingual-e5-base)                |
| `chunk_size / overlap` | 500 / 200                                     |
| `RETRIEVER_INITIAL_K`  | 20                                            |
| Reranker               | `bge-reranker-v2-m3`, top_n=4, threshold=0.01 |
| Промпт                 | Prompt v2 (основные факты, 2-6 предложений)   |
| `VLLM_MAX_TOKENS`      | 600                                           |

## Лицензия

MIT
