[← README](../README.md) · Раздел: Observability

# Observability — Prometheus, Grafana, Langfuse

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

## Метрики Prometheus

Реестр метрик собран в [hse_prom_prog/metrics.py](../hse_prom_prog/metrics.py).
Все имена живут в неймспейсе `agile_assistant_*`. ~50 кастомных метрик разделены
на 12 секций.

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

**LLM-as-a-Judge**:

| Метрика                   | Тип     | Лейблы      | Что показывает                                                                                                                                             |
| ------------------------- | ------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `judge_criterion_score`   | Gauge   | `criterion` | Score последнего evaluation per criterion (0/1): `practicality`, `language_quality`, `text_cleanliness`, `agile_correctness`, `completeness`, `politeness` |
| `judge_weighted_total`    | Gauge   | —           | Weighted total score последнего evaluation (0.0–1.0)                                                                                                       |
| `judge_evaluations_total` | Counter | `status`    | Терминальные исходы судьи: `success` / `parse_error` / `api_error`                                                                                         |

**Data sync (Celery Beat)**:

| Метрика                       | Тип       | Лейблы           | Что показывает                                                              |
| ----------------------------- | --------- | ---------------- | --------------------------------------------------------------------------- |
| `data_sync_timestamp_seconds` | Gauge     | `source`         | Unix-timestamp последнего успешного запуска (`jira_csv` / `knowledge_base`) |
| `data_sync_total`             | Counter   | `source, status` | Запуски по источнику и исходу (`success` / `failure`)                       |
| `data_sync_duration_seconds`  | Histogram | `source`         | Длительность одного прогона (CSV: секунды, база знаний: минуты)             |
| `data_sync_rows`              | Gauge     | `source`         | Кол-во строк / чанков после успешного sync                                  |

**FastAPI HTTP** (через `prometheus-fastapi-instrumentator`): автоматические
метрики `http_requests_total`, `http_request_duration_seconds` с лейблами
`method`, `handler`, `status`. `/metrics` исключён из инструментирования, чтобы
Prometheus сам себя не считал.

## Дашборды Grafana

Пять JSON-дашбордов в
[monitoring/grafana/dashboards/](../monitoring/grafana/dashboards/),
автоматически провижионятся при запуске Grafana через `provisioning/dashboards/`
(см. [monitoring/grafana/provisioning/](../monitoring/grafana/provisioning/)).

| Дашборд                                        | UID                    | Что показывает                                                                                                                                        |
| ---------------------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Agile Assistant — Overview**                 | `agile-overview`       | E2E latency p50/p95/p99, throughput RPS, error rate, queue wait, in-progress, query_type breakdown, data freshness (Jira CSV / база знаний last sync) |
| **Agile Assistant — Infrastructure**           | `agile-infrastructure` | PostgreSQL connections / TPS / replication lag, Redis ops/sec, Qdrant healthcheck, vLLM KV-cache, GPU                                                 |
| **Agile Assistant — Agents Deep Dive**         | `agile-agents`         | Per-agent latency, SQL retries / empty results, RAG retrieval / reranker / fallbacks, response branches                                               |
| **Agile Assistant — Guardrails & Safety**      | `agile-guardrails`     | L1 injection blocks, L2 layer breakdown, L3 failed checks, sanitizer corrections by layer, memory rotations                                           |
| **Agile Assistant — Quality (LLM-as-a-Judge)** | `agile-quality`        | Weighted total quality score, per-criterion scores (1h / 24h avg), judge-evaluation outcomes, API errors                                              |

Все дашборды настроены на datasource `prometheus`. Разворачиваются автоматически
после `docker compose up -d grafana` — провижионинг провалидирует dashboards и
зальёт их в папку «Agile Assistant».

### Overview — общий обзор системы

| Панель                                 | Что трекает                                                 | Зачем                                                                    |
| -------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------ |
| **E2E Latency (p50/p95/p99)**          | Полное время обработки запроса от создания задачи до ответа | Главный SLA-показатель: тренды по «обычному» (p50) и «худшему» (p95/p99) |
| **E2E Latency by Query Type (p95)**    | То же, но разбито по типу запроса (sql/rag/hybrid/simple)   | Сразу видно, какой тип запроса тормозит весь пайплайн                    |
| **Throughput (req/min)**               | Сколько задач в минуту обрабатывает система                 | Базовая метрика нагрузки                                                 |
| **Error Rate**                         | Доля задач со статусом FAILED ко всем задачам               | Если >10% — что-то системно сломалось                                    |
| **Tasks In Progress**                  | Сколько задач прямо сейчас в обработке                      | При >4 — очередь начинает копиться (concurrency=1 на воркер)             |
| **Queue Wait Time (p95)**              | Сколько задача ждёт в очереди до старта                     | Если растёт — нужны ещё воркеры                                          |
| **Task Status Distribution (1h)**      | Pie chart: сколько SUCCESS / FAILED / PENDING за час        | Быстрый health-check                                                     |
| **Celery Queue Length**                | Длина очереди в Redis                                       | Индикатор backpressure                                                   |
| **Celery Active Tasks**                | Сколько Celery-воркеров реально что-то делают               | Мониторинг занятости воркеров                                            |
| **Jira CSV — Last Sync**               | Сколько прошло с последней синхронизации Jira (cadence: 6ч) | Жёлтый при 6ч, красный при 12ч — данные устаревают                       |
| **Knowledge Base — Last Sync**         | Сколько прошло с последнего ре-индекса базы знаний в Qdrant | Жёлтый 24ч, красный 48ч                                                  |
| **Data Sync Runs (success / failure)** | Сколько успешных/упавших sync-запусков за 15 минут          | Красная линия = воркер падает                                            |

### Agents Deep Dive — внутренности агентов

**Supervisor**:

| Панель                                   | Что трекает                                                    | Зачем                                      |
| ---------------------------------------- | -------------------------------------------------------------- | ------------------------------------------ |
| **Intent Distribution (1h)**             | Pie chart намерений (greeting / help / agile_question / ...)   | Понять, что именно спрашивают пользователи |
| **Query Type Distribution (1h)**         | Pie chart типов запроса (sql / rag / hybrid / simple / error)  | Куда роутится трафик                       |
| **Parse Errors (1h)**                    | Сколько раз LLM вернул невалидный JSON при классификации       | Если >0 — модель «галлюцинирует» формат    |
| **Fast Path vs Slow Path (rate)**        | Сколько запросов решились regex-ом (fast) vs ушли в LLM (slow) | Чем больше fast — тем дешевле и быстрее    |
| **Classification Latency by Path (p95)** | Время классификации отдельно для fast и slow путей             | Fast должен быть <10ms, slow ~1-2s         |

**SQL Agent**:

| Панель                                 | Что трекает                                                             | Зачем                                                        |
| -------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------ |
| **SQL Agent Latency by Intent (p95)**  | Полное время работы SQL-агента, разбитое по намерению                   | Видно, какие интенты дороже (например, complex aggregations) |
| **SQL Query Execution Time (p50/p95)** | Чисто время выполнения SQL в Postgres                                   | Если растёт — индексы или медленные запросы                  |
| **SQL Query Status (rate, stacked)**   | success / error / empty per second                                      | Если много error — модель пишет невалидный SQL               |
| **SQL Retries**                        | Сколько раз агент перезапускал SQL после ошибки или semantic_check fail | Высокий retry rate = модель часто промахивается              |
| **Empty Results by Intent (1h)**       | Сколько SQL-запросов вернули 0 строк, по типу намерения                 | Указывает на проблемы с пониманием entity (название команды) |
| **Result Row Distribution**            | Гистограмма: сколько строк возвращают SQL-запросы                       | Понимание типичного размера результата                       |

**RAG Agent**:

| Панель                                           | Что трекает                                                               | Зачем                                                     |
| ------------------------------------------------ | ------------------------------------------------------------------------- | --------------------------------------------------------- |
| **RAG Retrieval Latency by Search Type (p95)**   | Время dense / sparse / hybrid поиска в Qdrant                             | Сравнить, какой режим быстрее                             |
| **Reranker Latency (p50/p95)**                   | Время cross-encoder реранкинга (bge-reranker-v2-m3)                       | Reranker — самый дорогой шаг RAG                          |
| **Full RAG Agent Duration (p95)**                | Полное время работы RAG-агента (retrieval + rerank + LLM)                 | Главный SLA для RAG-веток                                 |
| **Top Chunk Score (median)**                     | Медианная релевантность лучшего чанка                                     | Если падает — база знаний устарела или вопросы не покрыты |
| **Chunks: Retrieved vs After Reranker (median)** | Сколько чанков нашли изначально и сколько осталось после реранкинга       | Контроль воронки top-20 → top-4                           |
| **RAG Fallbacks**                                | Сколько раз пришлось переключаться между режимами (hybrid → dense и т.п.) | Если есть — основной режим неустойчив                     |

**Response Agent**:

| Панель                                    | Что трекает                                                        | Зачем                                                                  |
| ----------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| **Response Latency by Branch (p95)**      | Время финального ответа по ветке (sql\_\*/rag/hybrid/simple/error) | `rag` должен быть <100ms (passthrough), остальные — реальный LLM-вызов |
| **Response Length Distribution (tokens)** | Heatmap длины ответа в токенах                                     | Видно, попадает ли модель в `VLLM_MAX_TOKENS`                          |
| **LLM Timeouts (1h)**                     | Сколько раз вышел timeout LLM-вызова                               | Любое значение >0 — критично                                           |
| **Truncated Responses (1h)**              | Сколько ответов обрезано из-за лимита токенов                      | Если много — лимит занижен                                             |

### Guardrails & Safety — защитные слои

**Guardrails (L1/L2/L3)**:

| Панель                                | Что трекает                                                                     | Зачем                                          |
| ------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------- |
| **L1: TopicGuard Results (1h)**       | Pie chart исходов первого слоя (passed / off_topic_blocked / injection_blocked) | Сколько запросов отсеивается на входе          |
| **L1: Injection Blocks Over Time**    | Только prompt-injection блокировки во времени                                   | Любой всплеск = атака                          |
| **L2: SQLGuard Blocks by Layer (1h)** | Сколько SQL заблокировано по типу проверки (regex / AST / complexity)           | Куда чаще всего «попадает» сгенерированный SQL |
| **L3: ResponseGuard Results (1h)**    | Сколько финальных ответов прошло / заблокировано                                | Третий контур качества                         |
| **L3: Failed Check Types (1h)**       | По какой именно проверке упал ответ (PII, hallucination, off-topic)             | Прицельная диагностика                         |
| **Off-Topic Rate**                    | Скорость прихода off-topic запросов                                             | Если растёт — пользователи не понимают scope   |

**Entity Sanitizer**:

| Панель                                    | Что трекает                                                      | Зачем                                        |
| ----------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------- |
| **Corrections by Layer (1h)**             | Сколько исправлений сделал каждый слой санитайзера               | Видно, какой слой реально работает           |
| **Anaphora Carry-Forward by Entity (1h)** | Сколько раз «эта команда» / «этот спринт» подтянулись из истории | Метрика работы памяти диалога                |
| **Fallback Extractions by Field (1h)**    | Сколько раз pymorphy3 вытаскивал entity, когда LLM их пропустил  | Индикатор качества первичного извлечения LLM |

**Memory**:

| Панель                                 | Что трекает                                                   | Зачем                                               |
| -------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------- |
| **Context Window Tokens (median/p95)** | Сколько токенов истории попадает в контекст                   | Должно держаться около `HISTORY_TOKEN_BUDGET=800`   |
| **Context Turns (median)**             | Сколько ходов истории попадает в контекст                     | Дополнение к токенам — сколько реально «помещается» |
| **Session Rotations by Reason (1h)**   | Почему сессия закрывается (inactivity_30min / explicit_close) | Видно, естественно ли завершаются диалоги           |
| **Profile Updates & Summarizations**   | Скорость обновлений профиля и суммаризаций сессий             | Memory-фоновые задачи живы                          |

### Infrastructure — внешние сервисы

**vLLM** (две инстанции — main и sql):

| Панель                                    | Что трекает                                             | Зачем                                        |
| ----------------------------------------- | ------------------------------------------------------- | -------------------------------------------- |
| **vLLM Main: Running / Waiting Requests** | Сколько запросов сейчас в работе и в очереди у avibe-8b | Если waiting >0 постоянно — нужно мощнее GPU |
| **vLLM SQL: Running / Waiting Requests**  | То же для qwen3-8b (SQL-инстанс)                        | Аналогично                                   |
| **vLLM: Time to First Token (p95)**       | Сколько ждать первого токена от обоих инстансов         | Главная UX-метрика для streaming-сценариев   |
| **vLLM: GPU KV-Cache Usage**              | Загруженность KV-кэша на GPU                            | Жёлтый 80%, красный 95% — близко к OOM       |
| **vLLM: Tokens/sec (generation)**         | Скорость генерации токенов                              | Throughput модели                            |

**PostgreSQL**:

| Панель                 | Что трекает                                    | Зачем                                     |
| ---------------------- | ---------------------------------------------- | ----------------------------------------- |
| **Active Connections** | active / idle / idle in transaction соединения | «idle in transaction» >0 долго = leak     |
| **Replication Lag**    | Отставание реплики от мастера                  | Важно при масштабировании read-реплик     |
| **Database Size**      | Размер БД на диске                             | Контроль роста, особенно messages-таблицы |

**Redis**:

| Панель                    | Что трекает                    | Зачем                    |
| ------------------------- | ------------------------------ | ------------------------ |
| **Queue Length (celery)** | Длина Celery-очереди в Redis   | Дубль панели из Overview |
| **Memory Usage**          | Занятая Redis-ом память        | Контроль eviction        |
| **Connected Clients**     | Сколько коннектов держит Redis | Аномальные пики = leak   |

**Qdrant**:

| Панель                    | Что трекает                         | Зачем                                          |
| ------------------------- | ----------------------------------- | ---------------------------------------------- |
| **Search Latency (p95)**  | Скорость поиска в Qdrant (REST API) | Если растёт — возможно, нужна оптимизация HNSW |
| **Vectors in Collection** | Сколько векторов в коллекции        | 0 = база знаний не загружена → красный         |

### Quality (LLM-as-a-Judge) — оценка качества ответов судьёй

| Панель                                   | Что трекает                                                                                                                 | Зачем                                       |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **Weighted Score (current)**             | Текущая взвешенная оценка ответа (0–1)                                                                                      | Свежий «балл» последнего ответа             |
| **Weighted Score (1h rolling average)**  | Скользящее среднее за 1ч и 24ч + текущее значение                                                                           | Тренд качества: жёлтый <0.6, зелёный ≥0.8   |
| **Evaluation Rate by Outcome**           | Сколько оценок прошло success / parse_error / api_error / skipped в минуту                                                  | Здоровье самого судьи                       |
| **Evaluation Outcomes (1h)**             | Pie chart исходов оценок за час                                                                                             | Быстрый снапшот: судья работает или сыпется |
| **Scores by Criterion (1h average)**     | Средняя оценка по каждому критерию (Agile-корректность, Практичность, Контекст, Вежливость, Чистота текста, Качество языка) | Видно, по какому критерию модель проседает  |
| **Criterion Trends (24h)**               | То же, но во времени за 24ч                                                                                                 | Регрессия по критерию (например, языку)     |
| **Low Score Snapshots (weighted < 0.5)** | Таблица плохих ответов                                                                                                      | Кандидаты на ручной анализ                  |

## Алерты Prometheus

Все правила в
[monitoring/prometheus/alerts.yml](../monitoring/prometheus/alerts.yml). 18
правил в четырёх группах (severity: warning / critical):

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

**`agile_assistant_quality`** — LLM-as-a-Judge; 0/1 gauges усредняются через
`avg_over_time` за час, `for: 15m` глушит одиночные выбросы:

| Alert                       | Условие                                                                         | Severity |
| --------------------------- | ------------------------------------------------------------------------------- | -------- |
| `JudgeQualityDrop`          | `avg_over_time(judge_weighted_total[1h])` < 0.7, `for: 15m`                     | warning  |
| `JudgeAgileCorrectnessDrop` | `avg_over_time(judge_criterion_score{criterion="agile_correctness"}[1h])` < 0.6 | warning  |
| `JudgePracticalityDrop`     | `avg_over_time(judge_criterion_score{criterion="practicality"}[1h])` < 0.6      | warning  |
| `JudgeLanguageQualityDrop`  | `avg_over_time(judge_criterion_score{criterion="language_quality"}[1h])` < 0.6  | warning  |
| `JudgeUnavailable`          | доля `judge_evaluations_total{status="api_error"}` за 30 мин > 50%, `for: 10m`  | warning  |

**`agile_assistant_data_freshness`** — Celery Beat sync:

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
[k8s/monitoring/prometheus-rules.yaml](../k8s/monitoring/prometheus-rules.yaml).

## Трейсинг Langfuse

Реализация — [hse_prom_prog/tracing.py](../hse_prom_prog/tracing.py). Один
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
scoring. Этим пользуется `judge_task`: пишет 6 score'ов (practicality /
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

## Запуск observability-стека

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
> [database/init-langfuse.sql](../database/init-langfuse.sql) выполнится
> автоматически и создаст `langfuse_db`. Шаг 1 нужен только если volume уже
> существовал до Phase 3.

**Отключение Langfuse без удаления кода**:

```bash
# В .env:
LANGFUSE_ENABLED=false

docker compose restart api celery-worker
```

Все `@observe`-декораторы становятся no-op, метрики Prometheus продолжают
писаться. Полезно при отладке проблем с Langfuse-сервером или при локальной
разработке без отдельного контейнера.

## Связанные разделы

- [Конфигурация → Langfuse Tracing](configuration.md#langfuse-tracing-phase-3) —
  env-переменные SDK и сервера
- [Оценка → RAGAS](evaluation.md#rag-пайплайн-ragas) — feedback-петля «judge →
  trace → дашборд Quality»
