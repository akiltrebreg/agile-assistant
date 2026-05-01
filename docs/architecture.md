[← README](../README.md) · Раздел: Архитектура (детали)

# Архитектура — детали компонентов

Полный разбор пяти LLM-агентов, LLM Backend, БД и векторного хранилища. Краткое
введение и общая ASCII-диаграмма потока живут в [корневом README](../README.md).

## Компоненты

### 1. Supervisor Agent (классификатор запросов)

- Классифицирует запрос на один из 4 интентов:
  - `task` — запрос о конкретной задаче по ключу (например, «AL-38787»)
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
- **Entity Sanitizer**
  ([entity_sanitizer.py](../hse_prom_prog/agents/entity_sanitizer.py)):
  пост-обработка выхода LLM в 7 слоёв — (1) нормализация синонимов
  (enum-значения через `_SYNONYM_MAPS`), (2) валидация по БД (реальные
  issue_type / status), (3) фильтр галлюцинаций (шаблонные имена, плейсхолдеры),
  (4) enum query-presence check, (5) Russian morphology prefix match (4 символа)
  для sprint_name / cluster, (6) анафорический carry-forward с
  pymorphy3-лемматизацией маркеров, (7) fallback-экстрактор: тот же
  `_SYNONYM_MAPS` применяется как обратный словарь — если LLM не извлёк
  enum-поле, оно подставляется по совпадению лемм запроса с леммами синонимов
- **DB enum cache**: реальные значения `issue_type`, `status` подтягиваются из
  БД с TTL 1 час — сравнение с тем, что вернула LLM
- **Error intent fallback**: при parse failure возвращает
  `intent=error, query_type=error` вместо silent fallback — workflow корректно
  маршрутизирует в Response Agent с заготовленным сообщением

### 2. SQL Agent (LangGraph text-to-SQL с tool calling)

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
- **Безопасность**: `run_query` блокирует всё кроме `SELECT` через
  [SQLGuard](guardrails.md#l2--sqlguard-tool-level)

### 3. RAG Agent (ответ на основе базы знаний)

- Использует Qdrant как векторное хранилище для документов из
  [knowledge_base/](knowledge-base.md)
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
- Генерирует ответ через LLM строго на основе найденного контекста
- Возвращает ответ с указанием источников (category/filename)
- Graceful degradation: если Qdrant недоступен, workflow продолжает работать для
  SQL-запросов. Если sparse vectors отсутствуют в коллекции, hybrid/sparse
  режимы автоматически откатываются на dense search

### 4. Validator Agent (валидация результатов)

- Проверяет выходы SQL Agent и RAG Agent перед передачей в Response Agent
- Для каждого `query_type` определяет, какие данные доступны:
  - `sql` — есть ли результат из БД
  - `rag` — есть ли ответ из базы знаний
  - `hybrid` — какая комбинация данных доступна (оба, только SQL, только RAG)
- Формирует `validation_result` с флагами `use_sql`, `use_rag` и `note`
  (описание ошибки при отсутствии данных)

### 5. Response Agent (генерация ответа)

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
  - **rag**: passthrough `rag_response` (без повторного LLM-вызова)
  - **hybrid**: SQL-данные + RAG-контекст → LLM строит ответ в два блока:
    `ДАННЫЕ` (числа/команда/спринт из БД) и `АНАЛИЗ` (рекомендации из
    переданного контекста, без галлюцинаций URL)
- Обрабатывает ошибки, пустые результаты («Задача AL-99999 не найдена») и
  таймауты LLM

### 6. Memory Layer (short-term диалог + long-term профиль)

См. [memory.md](memory.md) — полный разбор. Здесь — краткий обзор для контекста
архитектуры:

- **MemoryManager** — фасад над `ConversationRepository`, `ProfileRepository`,
  `SummaryRepository` + `ContextBuilder` + `ProfileExtractor`. Единственная
  точка входа из workflow и Celery
- **ContextBuilder** — sliding window по токен-бюджету (`HISTORY_TOKEN_BUDGET`,
  1200 токенов): последние полные ходы, всё выпавшее покрывается rolling summary
- **ProfileExtractor** — детерминированный rule-based анализ, без LLM:
  `default_team`, `frequent_metrics`, `recent_sprints`, `dominant_query_types`
- **Анафорический carry-forward** — слой 6 в
  [Entity Sanitizer](#1-supervisor-agent-классификатор-запросов)
- **Inactivity rotation** — диалог idle > `SESSION_TIMEOUT_MINUTES` (30 мин)
  закрывается автоматически
- **Persistence** — таблицы `conversations`, `messages`, `user_profiles`,
  `conversation_summaries` ([детали](database.md))

### 7. Observability (Prometheus + Grafana + Langfuse)

См. [observability.md](observability.md) — полные таблицы метрик / дашбордов /
алертов / трейсов. В архитектурном контексте:

- **Prometheus метрики** ([metrics.py](../hse_prom_prog/metrics.py)) — единый
  реестр из ~50 метрик в неймспейсе `agile_assistant_*`. Скрейпятся с 9 целей:
  api, celery-worker, celery-judge, vllm-main, vllm-sql, postgresql, redis,
  qdrant, prometheus self
- **Grafana дашборды** — пять JSON в папке «Agile Assistant»: Overview,
  Infrastructure, Agents Deep Dive, Guardrails & Safety, Quality
  (LLM-as-a-Judge)
- **Langfuse трейсинг** ([tracing.py](../hse_prom_prog/tracing.py)) —
  singleton-клиент с graceful degradation. На каждый Celery-таск создаётся root
  trace + child-spans на каждого агента и каждый LLM-вызов
- **Kill-switch**: `LANGFUSE_ENABLED=false` отключает SDK полностью — `@observe`
  становится no-op, приложение работает идентично без трейсов

## LLM Backend

Два раздельных vLLM-контейнера на одной GPU (совместный шаринг памяти):

- **Основной LLM** (`vllm-server`, порт 8000): avibe-gptq-8bit (GPTQ 8-bit),
  используется Supervisor, RAG Agent, Validator, Response Agent
- **SQL LLM** (`vllm-sql-server`, порт 8001): Qwen3-8B-AWQ (4-bit,
  compressed-tensors), используется только SQL Agent. Запущен с флагами
  `--enable-auto-tool-choice --tool-call-parser=hermes` для поддержки tool
  calling, `--max-model-len=6144`, `--gpu-memory-utilization=0.38`
- **GPU-бюджет** на RTX 4090 (24 GB): vllm (avibe) — `max-model-len=6144`,
  `gpu-memory-utilization=0.55`; vllm-sql (qwen3) — `max-model-len=6144`,
  `gpu-memory-utilization=0.38`. Итого ~22.3 GB из 24 GB. vllm-sql стартует
  первым (`depends_on: service_healthy`), чтобы избежать OOM-race
- **Модели** скачиваются из Yandex Cloud S3 при первом старте контейнера

## База данных

- **СУБД**: PostgreSQL 16 (Alpine)
- **Таблицы** (обе доступны SQL Agent-у, модель сама выбирает нужную):
  - `report_agile_dashboard` — задачи Jira (58 полей): issue_key, feature_teams,
    sprint_name, issue_status_act, storypoints_act, issue_type, assignee_name и
    др. Используется для запросов по задачам
  - `report_agile_dashboard_metrics` — агрегированные метрики команд по спринтам
    (35 полей): done_total, scope_drop, complete_sp (velocity), sprint_goal,
    cancel_rate и др. Используется для запросов по метрикам
- **Описания для LLM**: `COMMENT ON TABLE/COLUMN` в
  [database/init.sql](../database/init.sql) объясняют назначение таблиц и
  колонок — LLM SQL Agent читает их через `pg_description` и использует для
  выбора правильной таблицы и колонки
- **Индексы**: issue_key, jirasprint_id, sprint_state, assignee_name,
  feature_teams
- **Данные**: загружаются из CSV файлов с использованием команды COPY

Полная схема всех таблиц (включая memory layer и аудит-таблицу `tasks`) —
[database.md](database.md).

## Векторное хранилище (Qdrant)

- **Версия**: Qdrant v1.13.2
- **Коллекция**: `business_docs` (настраивается через `QDRANT_COLLECTION_NAME`)
- **Dense vectors**: модель `intfloat/multilingual-e5-base` (768-мерные, cosine
  similarity, мультиязычные — поддержка русского языка). В `.env` это
  `EMBEDDING_MODEL=multilingual-e5-base` — имя папки в S3 / локальном кэше,
  **не** Hub ID с org-префиксом
- **Sparse vectors**: BM25 через `fastembed` (по умолчанию) или BGE-M3 learned
  sparse (`EMBEDDING_SPARSE_MODEL=BAAI/bge-m3`)
- **Загрузка модели**: snapshot подтягивается из
  `s3://${S3_MODELS_BUCKET}/${S3_MODELS_PATH}/${EMBEDDING_MODEL}/` в кэш
  `EMBEDDING_MODEL_CACHE_DIR` (по умолчанию `/app/models/`). Сначала это делает
  compose-job `download-embedding-model`, при пустом volume — runtime-страховка
  в `embeddings.ensure_embedding_model_downloaded()`. HuggingFace Hub в
  продакшене не дёргается
- **Документы**: PDF и Markdown из [knowledge_base/](knowledge-base.md)
  (Agile-практики, описания метрик, внутренние регламенты)
- **Ingestion pipeline**: загрузка из S3 (или локальной `knowledge_base/`) →
  pdfplumber (текст + таблицы) → chunking (500 символов, overlap 200) → prepend
  metadata (заголовок документа + секция) → dense embedding + sparse embedding →
  загрузка обоих типов в Qdrant

## Связанные разделы

- [Memory Layer](memory.md) — двухуровневая память: окно диалога +
  долговременный профиль
- [База знаний и RAG](knowledge-base.md) — структура документов и ingestion
  pipeline
- [Guardrails](guardrails.md) — L1/L2/L3 защита, работающие без LLM
- [Observability](observability.md) — метрики Prometheus, дашборды, трейсы
  Langfuse
