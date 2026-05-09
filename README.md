# Agile AI Assistant

Multi-agent система для анализа Jira-задач и Agile-практик: LangGraph-роутер из
пяти агентов (Supervisor / SQL / RAG / Validator / Response), трёхуровневые
guardrails, двухуровневая память (окно диалога + профиль пользователя),
полноценный observability-стек и асинхронный API. Работает на двух vLLM-серверах
поверх PostgreSQL, Qdrant, Celery, Redis, FastAPI, Streamlit, nginx; есть
deployment в Docker Compose и Kubernetes.

## Содержание

- [Описание ассистента](#описание-ассистента)
- [Архитектура](#архитектура)
- [Возможности (типы запросов)](#возможности-типы-запросов)
- [Memory Layer](#memory-layer)
- [Guardrails](#guardrails)
- [Observability](#observability)
- [Структура проекта](#структура-проекта)
- [Быстрый старт с Docker Compose](#быстрый-старт-с-docker-compose)
- [Использование (CLI)](#использование-cli)
- [API](#api)
- [Streamlit UI](#streamlit-ui)
- [Деплой](#деплой)
- [Оценка качества](#оценка-качества)
- [Конфигурация](#конфигурация)
- [Разработка](#разработка)
- [Лицензия](#лицензия)

## Описание ассистента

Agile AI Assistant — ассистент, который отвечает на вопросы по данным Jira (одна
конкретная задача, фильтры по командам/спринтам, метрики команд) и по
методическим документам компании (Agile-практики, описания метрик, регламенты),
а также объединяет оба источника в гибридных запросах.

Под капотом — LangGraph-граф из пяти агентов с детерминированной маршрутизацией
по `query_type`, тремя слоями guardrails (input regex / SQL AST / output
sanitization) и двухуровневой памятью (sliding window + профиль). Сверху —
FastAPI/Celery-стек для асинхронной обработки и Streamlit-чат, наблюдаемые через
Prometheus, Grafana и Langfuse.

## Архитектура

Приложение построено на основе LangGraph и использует пять агентов (Supervisor /
SQL / RAG / Validator / Response) плюс трёхуровневую систему
[guardrails](#guardrails): regex-фильтр на входе, валидация SQL перед
выполнением, пост-обработка ответа. Supervisor классифицирует запрос
пользователя, определяет `intent`, `entities` и `query_type`, затем
маршрутизирует по одному из пяти путей (включая `off_topic` — запрос не по теме,
ответ выдаётся заготовленным текстом без вызова остальных агентов). Между
сообщениями память слоя [Memory Layer](#memory-layer) сохраняет историю диалога
и профиль пользователя — Supervisor и Response Agent читают короткоживущий
контекст + долгоживущий профиль из блока `MemoryManager`. Поверх этого работает
[observability-стек](#observability): Prometheus + Grafana собирают метрики
latency / throughput / error rate / очередей, а Langfuse получает трейсы каждого
запроса с полными промптами и токенами:

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

**Стек**: LangGraph (роутер) · vLLM × 2 (avibe-gptq-8bit + Qwen3-8B-AWQ) ·
PostgreSQL 16 (Jira-данные + memory layer + audit) · Qdrant 1.13 (RAG-индекс
хранит dense+sparse векторы; режим поиска управляется `SEARCH_TYPE`, по
умолчанию `dense`) · Celery + Redis (асинхронная очередь) · FastAPI (HTTP API) ·
Streamlit (чат-UI) · nginx (reverse proxy) · Prometheus + Grafana + Langfuse
(observability) · Alembic (миграции).

Подробности по компонентам:

- [docs/architecture.md](docs/architecture.md) — пять агентов, LLM Backend, БД,
  Qdrant
- [docs/memory.md](docs/memory.md) — Memory Layer (полная версия)
- [docs/guardrails.md](docs/guardrails.md) — L1/L2/L3 + тестирование
- [docs/observability.md](docs/observability.md) — метрики, дашборды, алерты,
  трейсы

## Возможности (типы запросов)

Supervisor определяет `query_type` и маршрутизирует запрос по одной из веток:

| Тип         | Что делает                                         | Пример                                                    |
| ----------- | -------------------------------------------------- | --------------------------------------------------------- |
| `sql`       | Достаёт данные из PostgreSQL через SQL Agent       | «Расскажи о задаче AL-38787», «Velocity команды lpop»     |
| `rag`       | Ищет ответ в базе знаний через Qdrant              | «Как рассчитывается scope drop?», «Что такое Done Total?» |
| `hybrid`    | Комбинирует данные из БД + контекст из базы знаний | «Метрики lpop и что можно улучшить»                       |
| `simple`    | Прямой ответ Response Agent (без БД и RAG)         | «Привет», «Что ты умеешь?»                                |
| `off_topic` | Заготовленный отказ через `_off_topic_node`        | «Расскажи анекдот», «Какая погода?»                       |

Полные CLI-примеры по каждому типу — в разделе
[Использование (CLI)](#использование-cli).

## Memory Layer

Двухуровневая память, целиком в PostgreSQL, без отдельных контейнеров:

- **Окно диалога** (short-term): sliding window по токен-бюджету
  (`HISTORY_TOKEN_BUDGET=800`) + rolling summary для всего, что выпало;
  inactivity rotation после 30 минут простоя; анафорический carry-forward через
  лемматизацию.
- **Профиль пользователя** (long-term): детерминированный rule-based
  `default_team` / `frequent_metrics` / `recent_sprints`. Гейт ≥ 6 сообщений с
  одной командой — иначе профиль остаётся пустым.
- **Async refresh**: суммаризация и обновление профиля идут через Celery,
  никогда не блокируя путь ответа.

Подробнее: [docs/memory.md](docs/memory.md) (поведение и конфигурация) ·
[docs/database.md](docs/database.md) (схема + psql-инспекция своих данных).

## Guardrails

Трёхуровневая защита, **без** LLM-вызовов (<1 ms на каждый слой):

- **L1 — TopicGuard** (input, regex): блокирует prompt-injection, помечает
  whitelist fast-path. Off-topic классифицирует Supervisor через отдельный
  `query_type=off_topic`.
- **L2 — SQLGuard** (tool-level): 4 слоя fail-fast — limits → regex blacklist →
  AST через `sqlglot` → complexity (≤5 JOIN). Блокирует всё, что не read-only
  `SELECT` по whitelist-таблицам.
- **L3 — ResponseGuard** (output): 6 проверок — пустой/слишком длинный ответ,
  доля кириллицы, sql_leak, traceback, hallucinated_urls, internal_leak. Режимы
  BLOCK / sanitize.

Включается флагом `GUARDRAIL_ENABLED` в `.env` (по умолчанию `True`).

Подробнее: [docs/guardrails.md](docs/guardrails.md).

## Observability

Три плоскости наблюдаемости поверх workflow:

- **Prometheus + Grafana** — ~50 кастомных метрик в неймспейсе
  `agile_assistant_*` (pipeline / Celery / per-agent / guardrails / sanitizer /
  memory / quality / data sync); 5 дашбордов в папке «Agile Assistant»; 18
  алертов в 4 группах.
- **Langfuse** — трейс каждого запроса с полными промптами LLM-вызовов,
  входными/выходными токенами и model parameters.
- **Loki + Promtail** — централизованная агрегация логов всех сервисов.
  Приложение пишет одновременно в stdout (для `docker compose logs`) и в
  ротирующиеся файлы `./logs/<service>.log` (50 МБ × 5 backup); Promtail тейлит
  файлы и шлёт в Loki, retention 7 дней. Запросы — через Grafana → Explore →
  datasource `Loki` (например, `{service="api", level="ERROR"}`).

UI:

- Grafana — `http://195.209.218.21/grafana/` (метрики + логи через Explore)
- Prometheus — `http://195.209.218.21/prometheus/`
- Langfuse — `http://195.209.218.21:3001/`

Подробнее: [docs/observability.md](docs/observability.md).

## Структура проекта

```
agile-assistant/
├── agile_assistant/
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
│   │   └── workflow.py                # LangGraph StateGraph (5 агентов + 2 guardrail-узла + off_topic)
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
│   │   ├── embeddings.py              # Shared embedding utils (truncation, Matryoshka)
│   │   ├── ingest.py                  # Ingestion pipeline (S3 → pdfplumber → Qdrant)
│   │   ├── reranker.py                # Cross-encoder reranker (bge-reranker-v2-m3)
│   │   ├── retriever.py               # Multi-mode retriever (dense/sparse/hybrid)
│   │   └── sparse.py                  # Sparse embeddings (fastembed BM25 / BGE-M3)
│   └── tasks/
│       ├── __init__.py
│       ├── celery_app.py              # Celery app factory + Beat schedule (Jira: каждые 6 ч UTC, KB: 03:00 UTC ежесуточно)
│       ├── workflow_task.py           # Celery task (wraps workflow + inactivity rotation)
│       ├── memory_tasks.py            # Celery: summarize_session, update_profile_async, _refresh_rolling_summary
│       ├── judge_task.py              # LLM-as-a-Judge async scoring (queue=judge, vsellm)
│       └── sync_tasks.py              # sync_jira_data (S3→PostgreSQL, каждые 6 ч) + sync_knowledge_base (S3→Qdrant, daily 03:00 UTC)
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
├── docs/                              # Подробная документация по разделам
│   ├── architecture.md
│   ├── memory.md
│   ├── database.md
│   ├── knowledge-base.md
│   ├── api.md
│   ├── ui.md
│   ├── deployment.md
│   ├── kubernetes.md
│   ├── local-development.md
│   ├── guardrails.md
│   ├── observability.md
│   ├── evaluation.md
│   └── configuration.md
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml             # 9 scrape targets (api, celery-worker, celery-judge, vllm-main, vllm-sql, pg, redis, qdrant, self)
│   │   └── alerts.yml                 # 18 правил в 4 группах (pipeline / infrastructure / quality / data-freshness)
│   └── grafana/
│       ├── provisioning/              # Datasource (Prometheus) + dashboard provider
│       └── dashboards/                # 5 JSON-дашбордов: overview, infrastructure, agents, guardrails, quality
├── scripts/
│   ├── download_model.sh              # Download avibe-gptq-8bit from S3
│   └── verify_memory.sh               # End-to-end memory layer verification
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
├── k8s/                               # Kubernetes-манифесты (см. docs/kubernetes.md)
├── tests/                             # 756 тестов (см. раздел «Разработка → Тестирование»)
│   ├── __init__.py
│   ├── conftest.py
│   ├── contract/                      # Связки агентов и маршрутизация workflow (41)
│   └── unit/                          # Per-module unit тесты: agents/, memory/, tasks/, api/, database/, observability/, llm/, rag/, agents/guardrails/ (715)
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

Пошаговая инструкция для подъёма полного стека с нуля: PostgreSQL + Redis +
Qdrant + vLLM (×2) + FastAPI + Celery (worker / beat / judge) + Streamlit +
Nginx + Prometheus + Grafana + Langfuse.

> Проект разворачивается на сервере `195.209.218.21`. Везде, где в командах
> встречается `localhost`, имеется в виду сам сервер. Все шаги выполняются из
> корня репозитория и используют `docker compose` v2.

### Шаг 0: Клонирование и `.env`

```bash
git clone <repository-url> agile-assistant
cd agile-assistant
git checkout main

cp .env.example .env
```

Обязательные секреты в `.env`:

- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` — Yandex Cloud S3 (модели + база
  знаний)
- `VSELLM_API_KEY` — judge / RAGAS
- При желании `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`

Полный список переменных и их значения по умолчанию —
[docs/configuration.md](docs/configuration.md).

### Шаг 1: Сборка общего образа приложения

```bash
docker compose build app
```

Образ `app` шарится между `api`, `celery-worker`, `celery-beat`, `celery-judge`,
`streamlit`, `migrate` — собираем один раз.

### Шаг 2: Инфраструктура

```bash
docker compose up -d postgres redis qdrant
```

Дождитесь `healthy`:

```bash
docker compose ps postgres redis qdrant
```

### Шаг 3: Скачивание моделей из S3

```bash
docker compose up download-model download-sql-model download-embedding-model download-reranker-model
```

Эти job-контейнеры заливают `avibe-gptq-8bit`, `qwen3-8b-awq-4bit`,
`multilingual-e5-base` и `bge-reranker-v2-m3` в общие Docker volumes. Каждый job
сам проверяет, что модель уже на диске, и пропускает скачивание — повторный
запуск идемпотентен.

### Шаг 4: vLLM-серверы

```bash
docker compose up -d vllm vllm-sql
```

vLLM при старте загружает модель в GPU — это занимает несколько минут.
Healthcheck опирается на `/v1/models`. Прогресс старта:

```bash
docker compose logs -f vllm | grep -E "Application startup|Uvicorn running"
```

### Шаг 5: Миграции

```bash
docker compose run --rm migrate
```

Накатывает Alembic 001–005: `tasks`, `conversations`, `messages`,
`user_profiles`, `conversation_summaries`. Быстрый sanity:

```bash
docker compose exec postgres psql -U hse_user -d hse_jira_db -c "\dt"
```

Должны быть все пять таблиц. Подробное описание схемы и инспекция через psql —
[docs/database.md](docs/database.md).

### Шаг 6: Загрузка данных

```bash
# 6a. CSV из S3 → PostgreSQL (Jira-задачи, метрики)
docker compose up load-data

# 6b. PDF / Markdown из S3 → Qdrant (база знаний для RAG)
docker compose run --rm app python -m agile_assistant.rag.ingest
```

`load-data` скачивает CSV из `S3_DATA_BUCKET` и загружает в PostgreSQL. `ingest`
скачивает документы из `S3_KB_BUCKET` и индексирует в Qdrant. Без этих шагов
SQL-запросы и RAG-запросы не будут работать.

При первом запуске `ingest` (а также при первом RAG-запросе у поднятого
celery-worker) дополнительно подтянутся embedding- и reranker-модели из
`s3://${S3_MODELS_BUCKET}/${S3_MODELS_PATH}/${EMBEDDING_MODEL}/` и
`.../${RERANKER_MODEL}/` в локальный кэш (`EMBEDDING_MODEL_CACHE_DIR`, по
умолчанию `/app/models/`). Это runtime-страховка к compose-job'ам
`download-embedding-model` и `download-reranker-model` (Шаг 3) — если volume
пустой, модели всё равно появятся перед началом работы. HuggingFace Hub в проде
не дёргается.

### Шаг 7: API + воркеры + UI

```bash
docker compose up -d api celery-worker celery-beat celery-judge streamlit
```

`celery-beat` сразу начнёт планировать периодический sync данных:
`sync_jira_data` каждые 6 часов (S3 → PostgreSQL) и `sync_knowledge_base`
ежесуточно в 03:00 UTC (S3 → Qdrant). Полное расписание и метрики свежести
данных —
[docs/observability.md → Data sync (Celery Beat)](docs/observability.md#метрики-prometheus).

### Шаг 8: Nginx и мониторинг

```bash
docker compose up -d nginx prometheus grafana pg-exporter redis-exporter \
                     langfuse loki promtail
```

`loki` хранит логи в volume `loki_data` (retention 7 дней), `promtail` тейлит
файлы из bind-mount `./logs` (директория есть в репозитории, `*.log`-файлы
создаются Python'ом при старте api/celery-\*). Внешних портов у обоих нет —
Grafana ходит к Loki по docker-сети.

### Шаг 9: Smoke-проверка

```bash
docker compose ps
# все сервисы в Up / healthy

curl -fsS http://localhost/health
# {"status":"ok"}
```

В браузере:

- Streamlit UI — `http://195.209.218.21/`
- Grafana — `http://195.209.218.21/grafana/`
- Prometheus — `http://195.209.218.21/prometheus/`
- Langfuse — `http://195.209.218.21:3001/`
- Swagger API — `http://195.209.218.21/api/docs`

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
генерирует ответ на основе найденного контекста.

### Остановка

```bash
docker compose down
```

## Использование (CLI)

### Примеры запросов

```bash
# Через Docker Compose
docker compose run --rm app python -m agile_assistant.main "Выведи данные по задаче AL-38787"

# Локально
poetry run python -m agile_assistant.main "Выведи данные по задаче AL-38787"

# ── query_type=sql, intent=task (конкретная задача по ключу) ──
poetry run python -m agile_assistant.main "Информация о задаче AL-38799"
poetry run python -m agile_assistant.main "Покажи мне что там с AL-39041"
poetry run python -m agile_assistant.main "AL-39043"

# ── query_type=sql, intent=tasks_filter (поиск задач по фильтрам) ──
poetry run python -m agile_assistant.main "Все задачи команды cthulhu"
poetry run python -m agile_assistant.main "Задачи команды lpop в спринте #1 Q1'26"
poetry run python -m agile_assistant.main "Баги в кластере Logistics"
poetry run python -m agile_assistant.main "Задачи со статусом In Progress"

# ── query_type=sql, intent=metric (метрики команд/спринтов) ──
poetry run python -m agile_assistant.main "Done total из спринта Мандариновый рывок"
poetry run python -m agile_assistant.main "Какой scope drop у команды cthulhu"
poetry run python -m agile_assistant.main "Метрики команды lpop"
poetry run python -m agile_assistant.main "Velocity команды linehaul"

# ── query_type=rag (вопросы о практиках и метриках из базы знаний) ──
poetry run python -m agile_assistant.main "Что такое Groomed Backlog?"
poetry run python -m agile_assistant.main "Как снизить Scope Drop?"
poetry run python -m agile_assistant.main "Какие бейзлайновые значения метрик?"
poetry run python -m agile_assistant.main "Как рассчитывается velocity?"

# ── query_type=hybrid (данные из БД + рекомендации из базы знаний) ──
poetry run python -m agile_assistant.main "Покажи scope drop команды cthulhu и дай рекомендации"
poetry run python -m agile_assistant.main "Метрики lpop и что можно улучшить"

# ── query_type=simple (общие вопросы, без обращения к БД и RAG) ──
poetry run python -m agile_assistant.main "Что такое спринт в Agile?"
poetry run python -m agile_assistant.main "Как рассчитываются story points?"
poetry run python -m agile_assistant.main "Привет"
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

## API

Помимо CLI, приложение выставляет асинхронный REST API. Клиент делает POST,
получает `task_id` и забирает результат поллингом:

```bash
# Создать задачу
curl -s -X POST http://localhost/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"query": "Расскажи о задаче AL-38787"}'

# Забрать результат
curl -s http://localhost/api/tasks/<task_id> | python3 -m json.tool
```

Задача проходит статусы `PENDING` → `PROCESSING` → `COMPLETED` / `FAILED`.

Подробнее (поллинг, параллельная обработка, мониторинг, scaling) —
[docs/api.md](docs/api.md).

## Streamlit UI

Streamlit-чат на `http://localhost/`. Тонкий клиент, ходит **только** в FastAPI
по HTTP. Поддерживает все 5 типов запросов, прогресс-индикатор, историю диалогов
в сайдбаре, восстановление сессии при F5, авто-ротацию диалога после 30 минут
простоя.

Подробнее: [docs/ui.md](docs/ui.md).

## Деплой

| Способ                                       | Когда использовать                                                                      | Документация                                           |
| -------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **Docker Compose**                           | Основной путь, dev/staging/prod                                                         | [Шаги 0–9 выше](#быстрый-старт-с-docker-compose)       |
| **Production (nginx + gunicorn + Langfuse)** | docker-compose, но с nginx как единой точкой входа и упорядоченным запуском мониторинга | [docs/deployment.md](docs/deployment.md)               |
| **Kubernetes (minikube)**                    | k8s + CloudNativePG HA + Prometheus Operator                                            | [docs/kubernetes.md](docs/kubernetes.md)               |
| **Локально (Poetry)**                        | разработка под CLI, без Docker                                                          | [docs/local-development.md](docs/local-development.md) |

## Оценка качества

Четыре независимых eval-пайплайна — по одному для каждого агента. Результаты
пишутся в `eval/results/{experiment}_{timestamp}.json`.

| Eval           | Dataset (кейсов)                      | Метрики                                                                                                                        |
| -------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| RAG-пайплайн   | `golden_dataset.json` (41)            | RAGAS (GPT-5.2 as judge): context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness                |
| SQL Agent      | `sql_golden_dataset.json` (46)        | Rule-based: exact_match_fields, row_count_exact, value_exact, exact_match_grouped                                              |
| Supervisor     | `supervisor_golden_dataset.json` (81) | Routing accuracy, intent match, entity match (soft substring), confusion matrix, deploy gate (off_topic / boundary / baseline) |
| Response Agent | `response_golden_dataset.json` (40)   | Rule-based: must_contain / must_contain_any / must_not_contain / language / length / sources                                   |

Подробнее: [docs/evaluation.md](docs/evaluation.md).

## Конфигурация

Все настройки управляются через переменные окружения (см.
[.env.example](.env.example)). Полный справочник по 17 группам переменных —
[docs/configuration.md](docs/configuration.md). Самые важные:

- **vLLM**: `VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_TEMPERATURE`, `VLLM_MAX_TOKENS`
- **Search/RAG**: `SEARCH_TYPE` (`dense`/`sparse`/`hybrid`), `EMBEDDING_MODEL`,
  `RERANKER_ENABLED`
- **Memory Layer**: `HISTORY_TOKEN_BUDGET`, `SESSION_TIMEOUT_MINUTES`
- **Langfuse**: `LANGFUSE_ENABLED`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`

### Текущая лучшая конфигурация RAG

Все эксперименты сравниваются с этой конфигурацией:

| Параметр                          | Значение                                      |
| --------------------------------- | --------------------------------------------- |
| LLM (Supervisor / RAG / Response) | avibe-8b, GPTQ 8-bit                          |
| LLM (SQL Agent)                   | qwen3-8b, AWQ 4-bit                           |
| `SEARCH_TYPE`                     | `dense` (multilingual-e5-base)                |
| `chunk_size / overlap`            | 500 / 200                                     |
| `RETRIEVER_INITIAL_K`             | 20                                            |
| Reranker                          | `bge-reranker-v2-m3`, top_n=4, threshold=0.01 |
| `VLLM_MAX_TOKENS`                 | 600                                           |

## Разработка

### Установка dev-зависимостей

Для разработки необходимо установить дополнительные зависимости:

```bash
poetry install --with dev
```

Это установит:

- `ruff` — линтер и форматтер кода
- `pre-commit` — хуки для git
- `pytest` и `pytest-asyncio` — тестирование
- `pytest-cov` — покрытие кода тестами

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

Тесты разнесены по слою (unit/contract) и по модулю — всего **756 тестов**:

| Каталог                             | Тестов | Что покрыто                                                                                  |
| ----------------------------------- | -----: | -------------------------------------------------------------------------------------------- |
| `tests/contract/`                   |     41 | Связки агентов: Supervisor → SQL/RAG, SQL/RAG → Validator → Response, маршрутизация workflow |
| `tests/unit/agents/`                |    191 | Supervisor, SQL Agent, RAG Agent, Validator, Response Agent, EntitySanitizer                 |
| `tests/unit/agents/guardrails/`     |     58 | TopicGuard (L1), SQLGuard (L2), ResponseGuard (L3)                                           |
| `tests/unit/memory/`                |     74 | ContextBuilder, ProfileExtractor, MemoryManager, token_estimator, truncator                  |
| `tests/unit/tasks/`                 |     58 | workflow_task (Celery + inactivity rotation), memory_tasks, sync_tasks                       |
| `tests/unit/api/`                   |     28 | Роутеры `/tasks` и `/conversations`                                                          |
| `tests/unit/database/`              |     33 | Подключение к PostgreSQL, load_csv                                                           |
| `tests/unit/observability/`         |     46 | Реестр Prometheus + Langfuse tracing (no-op + real SDK)                                      |
| `tests/unit/llm/`                   |     26 | OpenAI-совместимый клиент vLLM                                                               |
| `tests/unit/rag/`                   |     20 | Embeddings utilities (truncation, normalisation)                                             |
| `tests/unit/test_conftest_smoke.py` |      7 | Smoke-тесты на работоспособность фикстур                                                     |

```bash
# Запустите тесты
poetry run pytest tests/

# С подробным выводом
poetry run pytest tests/ -v

# С покрытием кода
poetry run pytest tests/ --cov=agile_assistant
```

## Лицензия

MIT
