# Agile AI Assistant

Multi-agent система для анализа Jira-задач с использованием LangGraph, vLLM,
PostgreSQL, Qdrant, Celery, Redis, nginx, k8s.

## Содержание

- [Архитектура](#архитектура)
  - [Компоненты](#компоненты)
  - [LLM Backend](#llm-backend)
  - [База данных](#база-данных)
  - [Векторное хранилище (Qdrant)](#векторное-хранилище-qdrant)
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
- [Оценка RAG-пайплайна (RAGAS)](#оценка-rag-пайплайна-ragas)
  - [Golden Dataset](#golden-dataset)
  - [Запуск оценки](#запуск-оценки)
  - [Сравнение экспериментов](#сравнение-экспериментов)
- [Разработка](#разработка)
  - [Установка dev-зависимостей](#установка-dev-зависимостей)
  - [Code Quality](#code-quality)
  - [Тестирование](#тестирование)
- [Конфигурация](#конфигурация)
- [Текущая лучшая конфигурация RAG](#текущая-лучшая-конфигурация-rag)
- [Лицензия](#лицензия)

## Архитектура

Приложение построено на основе LangGraph и использует пять агентов с условным
ветвлением. Supervisor классифицирует запрос пользователя, определяет `intent`,
`entities` и `query_type`, затем маршрутизирует по одному из четырёх путей:

```
  Supervisor ──► (conditional routing by query_type)
    │
    ├─ sql     ──► SQL Agent ────────────────────► Validator ──► Response Agent
    │
    ├─ rag     ──► RAG Agent ────────────────────► Validator ──► Response Agent
    │
    ├─ hybrid  ──► SQL Agent ──┐
    │              RAG Agent ──┴─► Validator ──► Response Agent
    │
    └─ simple  ──► Response Agent (прямой ответ через LLM)
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
  entities + query_type

**2. SQL Agent** (шаблонный SQL)

- Получает `intent` + `entities` от Supervisor
- Строит SQL-запросы **программно из шаблонов** (не генерирует SQL через LLM):
  - `task` → `SELECT * FROM report_agile_dashboard WHERE issue_key = :key`
  - `tasks_filter` → динамический `WHERE` с `ILIKE` по entities
  - `metric` → `SELECT` из `report_agile_dashboard_metrics` с whitelist метрик
- Автоматический `LIMIT 100` на все запросы
- Whitelist допустимых метрик (защита от невалидных колонок)
- Параметризованные запросы через SQLAlchemy `text()` (защита от SQL-инъекций)

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

- Шесть режимов генерации ответа в зависимости от `query_type` и
  `validation_result`:
  - **task**: форматирует одну задачу с русскими лейблами → LLM генерирует ответ
  - **tasks_filter**: компактный список задач → LLM описывает результат
  - **metric**: метрики в JSON → LLM анализирует динамику
  - **rag**: ответ из базы знаний с указанием источников
  - **hybrid**: данные из БД + контекст из базы знаний → LLM объединяет
  - **simple (прямой)**: отвечает на общие вопросы без внешних данных
- Формирует контекстуальный ответ на русском языке
- Обрабатывает ошибки, пустые результаты и таймауты LLM

### LLM Backend

- **Модель**: avibe-gptq-8bit (GPTQ 8-bit квантизация, скачивается из Yandex
  Cloud S3)
- **Backend**: vLLM с OpenAI-compatible API, `--quantization=gptq`
- **URL**: `http://localhost:8000/v1`

### База данных

- **СУБД**: PostgreSQL 16 (Alpine)
- **Таблицы** (обе активно используются SQL Agent-ом):
  - `report_agile_dashboard` — задачи Jira (58 полей): issue_key, feature_teams,
    sprint_name, issue_status_act, storypoints_act и др. Используется для
    интентов `task` и `tasks_filter`
  - `report_agile_dashboard_metrics` — агрегированные метрики команд по спринтам
    (35 полей): done_total, scope_drop, velocity, sprint_goal и др. Используется
    для интента `metric`
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

## Структура проекта

```
hse-prom-prog/
├── hse_prom_prog/
│   ├── __init__.py
│   ├── config.py                      # Pydantic settings
│   ├── main.py                        # CLI entry point
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py              # Supervisor (intent + entities + query_type)
│   │   ├── sql_agent.py               # SQL agent (шаблонный SQL)
│   │   ├── rag_agent.py               # RAG agent (Qdrant + LLM)
│   │   ├── validator_agent.py         # Validator (проверка результатов)
│   │   ├── response_agent.py          # Response agent (LLM)
│   │   └── schema_description.py      # Описание схемы БД для LLM
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                     # FastAPI application
│   │   ├── dependencies.py            # DI (DB, repo)
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   └── tasks.py               # POST/GET /tasks endpoints
│   │   └── schemas/
│   │       ├── __init__.py
│   │       └── task.py                # Pydantic request/response
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py              # PostgreSQL connection manager
│   │   ├── load_csv.py                # Load CSV data from S3 into PostgreSQL
│   │   └── task_repository.py         # Task CRUD (raw SQL)
│   ├── graph/
│   │   ├── __init__.py
│   │   └── workflow.py                # LangGraph StateGraph (5 агентов)
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py                  # OpenAI client для vLLM
│   ├── models/
│   │   ├── __init__.py
│   │   └── task.py                    # TaskStatus enum, Task model
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
│       ├── celery_app.py              # Celery application factory
│       └── workflow_task.py           # Celery task (wraps workflow)
├── knowledge_base/                    # Загружается из S3 (S3_KB_BUCKET)
│   ├── agile/                         # Agile-практики, дашборды
│   ├── metrics/                       # Описания метрик
│   └── internal/                      # Внутренние регламенты
├── alembic/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       ├── 001_add_tasks_table.py
│       └── 002_add_cleanup_function.py
├── database/
│   └── init.sql                       # PostgreSQL schema (данные загружаются из S3)
├── scripts/
│   └── download_model.sh              # Download avibe-gptq-8bit from S3
├── streamlit_app/
│   ├── app.py                         # Streamlit entrypoint (chat UI)
│   ├── api_client.py                  # HTTP client for FastAPI
│   ├── config.py                      # Env-based settings
│   └── components/
│       ├── __init__.py
│       ├── sidebar.py                 # Sidebar (status, controls)
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
│   ├── golden_dataset.json            # 41 вопрос для оценки RAG
│   ├── metrics.py                     # RAGAS-метрики (GPT-5.2 as judge)
│   ├── run_eval.py                    # CLI: запуск оценки
│   ├── compare.py                     # CLI: сравнение экспериментов
│   ├── spot_check.py                  # Точечная проверка retrieval без RAGAS
│   └── results/                       # Результаты (gitignored)
├── tests/
│   ├── __init__.py
│   └── test_workflow.py               # 65 тестов (все агенты + workflow)
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
git checkout checkpoint_13

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

Supervisor извлекает ключ `AL-38787`, SQL Agent выполняет
`SELECT * FROM report_agile_dashboard WHERE issue_key = 'AL-38787'`, Response
Agent формирует ответ с описанием задачи, статусом, исполнителем, командой,
story points и другими полями.

**2. Запрос метрик по спринту** (таблица `report_agile_dashboard_metrics`):

> Напиши мне Done Total по спринту 26Q1.1 Конь не валялся

Supervisor определяет intent=`metric`, SQL Agent выполняет запрос к таблице
`report_agile_dashboard_metrics` с фильтром по `sprint_name`, Response Agent
возвращает значение метрики `done_total` для указанного спринта.

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

[SQL Agent] Выполняю запрос к PostgreSQL...
[SQL Agent] Данные успешно получены

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
- Индикатор статуса API в сайдбаре (Online / Offline)
- Обработка ошибок и таймаутов с понятными сообщениями
- Детали выполнения (timestamps) в раскрывающемся блоке
- Кнопка очистки чата

## Nginx + Production

Nginx выступает единой точкой входа (порт 80). FastAPI и Streamlit не имеют
внешних портов и доступны только через reverse proxy.

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

| Контейнер  | Образ                 | Роль                                            |
| ---------- | --------------------- | ----------------------------------------------- |
| **nginx**  | `nginx:1.27-alpine`   | Reverse proxy, единственный открытый порт (80)  |
| **api**    | Dockerfile + gunicorn | FastAPI в production (gunicorn + UvicornWorker) |
| **static** | `busybox` + volume    | Хранит статические файлы, шарит volume с nginx  |
| **qdrant** | `qdrant:v1.13.2`      | Векторное хранилище для RAG Agent               |

### Маршруты nginx

| Путь              | Куда проксирует  | Особенности                                        |
| ----------------- | ---------------- | -------------------------------------------------- |
| `/api/*`          | `api:8080`       | Prefix `/api` удаляется (`/api/tasks` -> `/tasks`) |
| `/static/*`       | Диск / Streamlit | Сначала volume, затем fallback на Streamlit        |
| `/docs`, `/redoc` | `api:8080`       | Swagger UI                                         |
| `/_stcore/stream` | `streamlit:8501` | WebSocket (Upgrade + Connection headers)           |
| `/` (default)     | `streamlit:8501` | Streamlit UI с WebSocket-поддержкой                |

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
│   ├── vllm.yaml                 # vLLM (avibe-gptq-8bit, GPU, S3 init)
│   ├── api.yaml                  # FastAPI (gunicorn, 2 replicas)
│   ├── celery-worker.yaml        # Celery worker (threads, concurrency=4)
│   └── streamlit.yaml            # Streamlit UI
└── services/
    ├── qdrant-svc.yaml           # ClusterIP :6333, :6334
    ├── redis-svc.yaml            # ClusterIP :6379
    ├── vllm-svc.yaml             # ClusterIP :8000 (name: vllm-server)
    ├── api-svc.yaml              # ClusterIP :8080
    └── streamlit-svc.yaml        # ClusterIP :8501
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
kubectl apply -f k8s/services/               # qdrant, redis, vllm-server, api, streamlit
```

> Service-ы применяются до Deployment-ов, чтобы DNS-имена были доступны при
> старте контейнеров. PostgreSQL Service создаётся автоматически оператором
> CloudNativePG.

**3.3. Инфраструктура** — БД, векторное хранилище, кеш, LLM:

```bash
kubectl apply -f k8s/statefulsets/postgres-cluster.yaml   # CloudNativePG: 1 primary + 2 standby
kubectl apply -f k8s/deployments/qdrant.yaml              # Qdrant v1.13.2
kubectl apply -f k8s/deployments/redis.yaml               # Redis 7 (брокер Celery)
kubectl apply -f k8s/deployments/vllm.yaml                # vLLM + avibe-gptq-8bit (GPU, S3 download)
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
kubectl apply -f k8s/deployments/celery-worker.yaml   # Celery worker (embedding-модель, 4Gi RAM)
kubectl apply -f k8s/deployments/streamlit.yaml        # Streamlit UI
```

**3.7. Ingress** — маршрутизация внешнего трафика:

```bash
kubectl apply -f k8s/ingress.yaml    # 4 Ingress-ресурса: API, static-svg, static-css, UI
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

## Оценка RAG-пайплайна (RAGAS)

Модуль `eval/` реализует автоматическую оценку качества RAG-пайплайна с помощью
фреймворка [RAGAS](https://docs.ragas.io/). В качестве LLM-as-judge используется
GPT-5.2 через OpenAI-compatible API (vsellm).

### Метрики

| Метрика              | Что оценивает                                | Категория  |
| -------------------- | -------------------------------------------- | ---------- |
| `context_precision`  | Точность найденных чанков                    | Retrieval  |
| `context_recall`     | Полнота найденных чанков                     | Retrieval  |
| `faithfulness`       | Верность ответа контексту (без галлюцинаций) | Generation |
| `answer_relevancy`   | Релевантность ответа вопросу                 | Generation |
| `answer_correctness` | Корректность ответа (vs ground truth)        | End-to-end |

### Golden Dataset

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
заданы в `.env` (см. [Конфигурация](#конфигурация)). Qdrant должен быть запущен
с загруженной базой знаний.

### Сравнение экспериментов

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

65 тестов покрывают все компоненты системы:

| Компонент      | Тестов | Что покрыто                                       |
| -------------- | ------ | ------------------------------------------------- |
| Supervisor     | 14     | regex, LLM-классификация, query_type, fallback    |
| SQL Agent      | 11     | шаблоны SQL, ошибки БД, пустые результаты         |
| RAG Agent      | 6      | retrieval, генерация, ошибки, truncation, sources |
| Validator      | 9      | sql/rag/hybrid/both-failed сценарии               |
| Response Agent | 12     | direct, sql, rag, hybrid, ошибки LLM              |
| Workflow       | 5      | все 4 маршрута + workflow без retriever           |
| Параметризация | 8      | issue key extraction, general queries             |

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
