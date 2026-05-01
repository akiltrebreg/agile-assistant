[← README](../README.md) · Раздел: Оценка агентов

# Оценка агентов

Модуль [eval/](../eval/) содержит четыре независимых eval-пайплайна — по одному
для каждого LLM-агента. Каждый использует свой golden dataset и свои rule-based
или LLM-based метрики, результаты пишутся в
`eval/results/{experiment}_{timestamp}.json`.

| Eval           | Dataset (кейсов)                      | Метрики                                                                                                                        |
| -------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| RAG-пайплайн   | `golden_dataset.json` (41)            | RAGAS (GPT-5.2 as judge): context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness                |
| SQL Agent      | `sql_golden_dataset.json` (46)        | Rule-based: exact_match_fields, row_count_exact, value_exact, exact_match_grouped                                              |
| Supervisor     | `supervisor_golden_dataset.json` (81) | Routing accuracy, intent match, entity match (soft substring), confusion matrix, deploy gate (off_topic / boundary / baseline) |
| Response Agent | `response_golden_dataset.json` (40)   | Rule-based: must_contain / must_contain_any / must_not_contain / language / length / sources                                   |

## RAG-пайплайн (RAGAS)

Оценка качества RAG-пайплайна через [RAGAS](https://docs.ragas.io/). В качестве
LLM-as-judge используется GPT-5.2 через OpenAI-compatible API (vsellm).

### Метрики

| Метрика              | Что оценивает                                | Категория  |
| -------------------- | -------------------------------------------- | ---------- |
| `context_precision`  | Точность найденных чанков                    | Retrieval  |
| `context_recall`     | Полнота найденных чанков                     | Retrieval  |
| `faithfulness`       | Верность ответа контексту (без галлюцинаций) | Generation |
| `answer_relevancy`   | Релевантность ответа вопросу                 | Generation |
| `answer_correctness` | Корректность ответа (vs ground truth)        | End-to-end |

### Golden Dataset

Файл [eval/golden_dataset.json](../eval/golden_dataset.json) содержит 41 вопрос
с эталонными ответами, составленными строго по документам из
[knowledge_base/](knowledge-base.md):

| Категория | Вопросов | Описание                                                    |
| --------- | -------- | ----------------------------------------------------------- |
| metrics   | 25       | Вопросы по описаниям метрик (done_total, scope_drop и т.д.) |
| agile     | 6        | Agile-практики, дашборды                                    |
| cross_doc | 5        | Вопросы, требующие информации из нескольких документов      |
| negative  | 5        | Вопросы, на которые в базе знаний **нет** ответа            |

Негативные примеры прогоняются через пайплайн, но исключаются из RAGAS-оценки
(нет ground truth для сравнения).

### Запуск оценки

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
заданы в `.env` (см. [Конфигурация](configuration.md)). Qdrant должен быть
запущен с загруженной базой знаний.

## SQL Agent

Оценивает качество text-to-SQL: сгенерированный запрос, количество строк,
ключевые поля.

**Датасет** [eval/sql_golden_dataset.json](../eval/sql_golden_dataset.json) (46
кейсов):

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

## Supervisor Agent

Оценивает маршрутизацию + извлечение сущностей. Реальный LLM через vLLM + entity
sanitizer с DB-валидацией.

**Датасет**
[eval/supervisor_golden_dataset.json](../eval/supervisor_golden_dataset.json)
(81 кейс):

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

## Response Agent

Оценивает формат финального ответа: обязательные/запрещённые термины, язык,
длина, источники. Не вызывает Supervisor / SQL / RAG — весь входной `state`
(включая `rag_response`, `sql_result`) зафиксирован в датасете.

**Датасет**
[eval/response_golden_dataset.json](../eval/response_golden_dataset.json) (40
кейсов):

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

**Метрики (per-case):** часть rule-based проверок (language ≥30% кириллицы,
must_contain / must_not_contain, длина, наличие источников) перекрывается с L3
[ResponseGuard](guardrails.md#l3--responseguard-output) и работает как
регрессия. Поэтому отдельной таблицы здесь нет — детали проверок:

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

## Сравнение RAG-экспериментов

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

## Связанные разделы

- [База знаний](knowledge-base.md) — данные, на которых работает RAG-eval
- [Guardrails](guardrails.md) — какие проверки L1/L2/L3 покрывают eval'ы
  регрессионно
