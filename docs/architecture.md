[← README](../README.md) · Раздел: Архитектура (детали)

# Архитектура — детали компонентов

Полный разбор пяти агентов, LLM Backend, БД и векторного хранилища. Краткое
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
  ([entity_sanitizer.py](../agile_assistant/agents/entity_sanitizer.py)) — слой
  нормализации сущностей между Supervisor'ом и SQL / RAG-агентами. Supervisor
  извлекает сущности «как сказал человек» («баги», «в работе», «у них»,
  «velocity»), а в БД и в SQL-промпте используются канонические значения (`Bug`,
  `In Progress`, конкретное имя команды, `velocity`). Без нормализации SELECT
  либо ничего не находит, либо падает. Sanitizer делает четыре вещи:
  1. **Синонимы → канон** — статические словари русских / разговорных форм
     (`баги/багов/bug → Bug`, `сделана/готово → Done`,
     `улучшение → Improvement`).
  2. **Валидация по БД** — реальные команды, спринты, `issue_type`, `status`
     подтягиваются из Postgres (TTL-кэш 1 час). Если LLM выдала несуществующую
     команду — она дропается, а не уходит в SQL.
  3. **Anaphora carry-forward** — на «а у них velocity?» подтягивает `team_name`
     из предыдущего хода. Маркеры («у них», «этой команды», «в том же», ...)
     распознаются через `pymorphy3` лемматизацию, чтобы «командой cthulhu» и
     «команды cthulhu» считались одинаково.
  4. **Drop > hallucinate** — всё, что не нашлось ни в синонимах, ни в БД,
     превращается в `None`. Лучше уронить запрос или ответить «не нашёл», чем
     выдумать значение.
- **Error intent fallback**: при parse failure возвращает
  `intent=error, query_type=error` вместо silent fallback — workflow корректно
  маршрутизирует в Response Agent с заготовленным сообщением

### 2. SQL Agent (LangGraph text-to-SQL с tool calling)

SQL Agent отвечает за всё, что требует чисел из Postgres: задачи, фильтры по
командам / спринтам, метрики. Превращает запрос пользователя в SELECT, выполняет
его, отдаёт результат дальше. Реализован как мини-цикл на LangGraph: «LLM
предлагает SQL → tool выполняет → если ошибка, LLM пробует ещё раз → когда
успех, нода `extract` забирает полные данные».

- **Какая модель.** Qwen3-8B-AWQ (4-bit квантизация) в отдельном vLLM-контейнере
  `vllm-sql` на порту 8001. Это **отдельный** vLLM от основного (avibe-8b на
  порту 8000) — Qwen лучше справляется с tool calling, поэтому держим под SQL
  его, а под Supervisor / Response — avibe.
- **Схема БД сама подтягивается.** Перед каждым запуском агент идёт в
  `information_schema` PostgreSQL, собирает DDL разрешённых таблиц вместе с
  описаниями колонок (`COMMENT ON COLUMN ...`) и вставляет это в system prompt.
  Кэш на 10 минут. Какие таблицы разрешено показывать модели — задаётся
  whitelist'ом `ALLOWED_TABLES` в
  [schema_loader.py:22](../agile_assistant/agents/schema_loader.py#L22) (сейчас
  это `report_agile_dashboard` и `report_agile_dashboard_metrics`). Это
  сознательное ограничение поверхности атаки: модель не должна видеть
  `pg_catalog`, схему Langfuse и пр. Из этого следует:
  - **новые колонки** в уже разрешённых таблицах подхватываются автоматически
    после обновления кэша, промпт править не надо;
  - **новая таблица** появится в промпте только после явного добавления её имени
    в `ALLOWED_TABLES` и пересборки контейнера.
- **Один tool — `run_query(sql)`.** Это единственный инструмент, доступный
  модели. Выполняет SELECT, возвращает модели **sample**: 3 строки × 10 колонок.
  Этого достаточно, чтобы модель поняла «такой запрос вернул осмысленные данные»
  и не тратил токены на простыни ответа.
- **Модель обязана начать с SQL.** До первого успешного `run_query` мы передаём
  в vLLM `tool_choice="any"` — модель технически не может ответить текстом
  «нужно подумать», она вынуждена сразу позвать tool. После успешного запроса
  режим переключается на свободный, чтобы LLM могла нормально завершить диалог.
- **До 3 попыток на исправление SQL.** Retry'и могут сработать по двум причинам:
  - **SQL упал** (синтаксис, отсутствие колонки, и т. п.) → сообщение об ошибке
    возвращается модели, она пробует снова. Если на следующей попытке модель
    генерирует **тот же самый SQL** — добавляем явную подсказку «этот запрос уже
    не сработал, нужна другая формулировка». Это спасает от петель.
  - **SQL отработал, но семантически не подходит** под вопрос —
    `_semantic_check` в
    [sql_agent.py:532-544](../agile_assistant/agents/sql_agent.py#L532-L544)
    через regex-сигналы ловит mismatch (например, на «у какой команды самый
    большой scope drop?» модель сделала просто `WHERE team_name='cthulhu'` без
    `ORDER BY` / `MAX`). В таком случае модели тоже отдаётся подсказка «query
    вернул данные, но они неверные — перепиши».
- **Полные данные — отдельной нодой.** Модель видит только sample, чтобы
  экономить токены, но в финальный ответ нужны все строки. Поэтому когда модель
  закончила, нода `extract` перевыполняет **все успешные SQL** из истории и
  склеивает результаты — уже без лимита 3×10. Это нужно для cross-table
  запросов: модель может в одном диалоге сделать несколько параллельных
  tool-calls (например, «scope drop И отменённые задачи» — две таблицы, два
  SELECT'а), и в финал должны попасть строки из всех. Эти полные данные уходят в
  Validator и Response Agent.
- **Чистка `<think>`-блоков.** Qwen3 — reasoning-модель, она пишет рассуждения
  внутри тегов `<think>...</think>`. Они нужны модели на момент генерации, но в
  истории диалога только засоряют контекст. Перед каждым следующим вызовом vLLM
  эти блоки вырезаются, иначе на 2-3-й итерации упирались бы в 6144 токена —
  лимит `max-model-len` для vllm-sql.
- **SQLGuard блокирует всё, кроме SELECT.** Прежде чем `run_query` отправит
  запрос в Postgres, SQL проходит через
  [SQLGuard](guardrails.md#l2--sqlguard-tool-level): отлавливаются
  `INSERT / UPDATE / DELETE / DROP`, подозрительные конструкции, слишком большое
  число JOIN'ов. Если модель попытается сделать что-то деструктивное — запрос
  даже не дойдёт до БД.

### 3. RAG Agent (ответ на основе базы знаний)

RAG Agent отвечает за вопросы, ответ на которые лежит **в документах**, а не в
БД: что такое scope drop, как считается velocity и т. п. Документы хранятся в
[knowledge_base/](knowledge-base.md) (PDF + Markdown), при ingestion'е они
режутся на куски (~500 символов) и складываются в Qdrant. Когда приходит вопрос,
агент находит самые релевантные куски и просит LLM сформулировать ответ строго
по ним.

Полный путь запроса:

```
вопрос пользователя
   ↓
[1] поиск кандидатов в Qdrant       — top-20 чанков
   ↓
[2] cross-encoder реранкер          — top-4 после переоценки
   ↓
[3] склейка в контекст (≤4000 симв.)
   ↓
[4] LLM генерирует ответ            — строго по контексту
   ↓
ответ + список источников
```

- **Хранилище — Qdrant.** Это векторная БД, оптимизированная под поиск по
  смыслу. Каждый кусок текста (chunk) превращается в вектор-эмбеддинг —
  несколько сотен чисел, которые описывают «о чём этот текст». Похожие по смыслу
  куски имеют близкие векторы. Поиск — это «найди в коллекции 20 чанков, чьи
  векторы ближе всего к вектору вопроса».
- **Три режима поиска** (выбирается через `SEARCH_TYPE` в `.env`):
  - **`dense`** (по умолчанию) — поиск по смыслу. Вопрос и чанки кодируются
    моделью `multilingual-e5-base` (768 чисел на текст). Близость считается как
    cosine similarity. Хорошо ловит перефразирования: «как уменьшить
    отвалившиеся задачи?» найдёт документ про scope drop, даже если этого слова
    в вопросе нет.
  - **`sparse`** — поиск по словам (BM25). Это «улучшенный TF-IDF»: чем реже
    слово в коллекции и чем чаще оно в чанке, тем выше score. Хорошо работает с
    точными терминами и редкими словами (имена команд, аббревиатуры типа
    «AL-38787»), но не понимает синонимов. Управляется через
    `EMBEDDING_SPARSE_MODEL`: пусто = fastembed BM25, `BAAI/bge-m3` = learned
    sparse (что-то промежуточное между BM25 и dense).
  - **`hybrid`** — оба варианта одновременно. Qdrant сам прогоняет dense и
    sparse поиск параллельно (prefetch API) и сливает результаты алгоритмом RRF
    (Reciprocal Rank Fusion): каждому чанку даётся score, обратно
    пропорциональный его рангу в обоих списках, и итоговый рейтинг — сумма.
    Балансирует «по смыслу» и «по словам».
- **Двухэтапный retrieval с реранкером.** Первый этап (dense / sparse / hybrid)
  отдаёт **20** кандидатов. Это широкая сеть — мы не уверены, что лучший чанк
  попал в top-3. Второй этап — **cross-encoder реранкер** `bge-reranker-v2-m3`:
  для каждого из 20 кандидатов он смотрит пару «(вопрос, чанк)» целиком и выдаёт
  точечный score релевантности. Это существенно медленнее, чем dense-поиск
  (потому и не используется на первом этапе для всей коллекции), зато заметно
  точнее. Из 20 кандидатов он отбирает **top-4**. Реранкер можно отключить через
  `RERANKER_ENABLED=false` — полезно для A/B-экспериментов, чтобы сравнить
  качество с ним и без.
- **Контекст ≤ 4000 символов.** После top-4 чанки склеиваются в один блок
  текста, который передаётся LLM как «вот выдержки из базы знаний, отвечай
  строго по ним». 4000 символов — компромисс: больше → дольше генерация и выше
  шанс «потерять» правильный ответ среди шума; меньше → можем не поместить
  нужные детали.
- **LLM генерирует ответ строго по контексту.** Используется тот же avibe-8b,
  что и в Response Agent.
- **Источники сохраняются для трейсинга.** Каждый чанк в Qdrant хранит metadata
  `{category, filename}` (например, `metrics/scope_drop.md`). После ответа агент
  кладёт список использованных источников в state как `rag_sources` — это видно
  в Langfuse-трейсах и в eval-датасетах, но пользователю в Streamlit UI не
  показывается.
- **Graceful degradation — что если что-то сломалось:**
  - Qdrant недоступен (упал, сетевой сбой) → `_get_retriever_safe()` ловит
    исключение, RAG-узел возвращает `rag_response=None`. Validator видит, что
    источник пустой, и Response Agent через LLM формулирует ответ «по этому
    вопросу нет данных в базе знаний». Сообщение не хардкод — генерируется
    каждый раз. Главное: **SQL-запросы продолжают работать**, workflow не
    падает.
  - В коллекции нет sparse-векторов (например, ingestion прошёл по старому коду
    без BM25) → hybrid и sparse автоматически **откатываются на dense**. Метрика
    `rag_fallbacks_total{from_mode="hybrid", to_mode="dense"}` покажет, как
    часто это происходит.

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
  800 токенов): последние полные ходы, всё выпавшее покрывается rolling summary
- **ProfileExtractor** — детерминированный rule-based анализ, без LLM:
  `default_team`, `frequent_metrics`, `recent_sprints`, `dominant_query_types`
- **Анафорический carry-forward** — реализован в
  [Entity Sanitizer](#1-supervisor-agent-классификатор-запросов): на «а у них
  velocity?» `team_name` подтягивается из предыдущего хода через
  `pymorphy3`-лемматизацию маркеров
- **Inactivity rotation** — диалог idle > `SESSION_TIMEOUT_MINUTES` (30 мин)
  закрывается автоматически
- **Persistence** — таблицы `conversations`, `messages`, `user_profiles`,
  `conversation_summaries` ([детали](database.md))

### 7. Observability (Prometheus + Grafana + Langfuse)

См. [observability.md](observability.md) — полные таблицы метрик / дашбордов /
алертов / трейсов. В архитектурном контексте:

- **Prometheus метрики** ([metrics.py](../agile_assistant/metrics.py)) — единый
  реестр из ~50 метрик в неймспейсе `agile_assistant_*`. Скрейпятся с 9 целей:
  api, celery-worker, celery-judge, vllm-main, vllm-sql, postgresql, redis,
  qdrant, prometheus self
- **Grafana дашборды** — пять JSON в папке «Agile Assistant»: Overview,
  Infrastructure, Agents Deep Dive, Guardrails & Safety, Quality
  (LLM-as-a-Judge)
- **Langfuse трейсинг** ([tracing.py](../agile_assistant/tracing.py)) —
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
