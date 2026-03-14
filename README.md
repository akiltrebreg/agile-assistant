# HSE Prom Prog - Agile AI Assistant (Задание 4)

Multi-agent система для анализа Jira-задач с использованием LangGraph, vLLM,
PostgreSQL, Qdrant, Celery, Redis, nginx.

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

**3. RAG Agent** (ответы на основе базы знаний)

- Использует Qdrant как векторное хранилище для документов из `knowledge_base/`
- Поиск релевантных фрагментов через HuggingFace-эмбеддинги
  (`intfloat/multilingual-e5-base`)
- Ограничение контекста: до 4000 символов из top-4 релевантных чанков
- Генерирует ответ через LLM строго на основе найденного контекста
- Возвращает ответ с указанием источников (category/filename)
- Graceful degradation: если Qdrant недоступен, workflow продолжает работать для
  SQL-запросов

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

- **Модель**: Qwen/Qwen2.5-3B-Instruct
- **Backend**: vLLM с OpenAI-compatible API
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
- **Эмбеддинги**: `intfloat/multilingual-e5-base` (768-мерные, мультиязычные —
  поддержка русского языка)
- **Метрика**: Cosine similarity
- **Документы**: PDF и Markdown из `knowledge_base/` (Agile-практики, описания
  метрик, внутренние регламенты)
- **Ingestion pipeline**: загрузка → chunking (1000 символов, overlap 200) →
  эмбеддинг → загрузка в Qdrant

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
│   │   ├── ingest.py                  # Ingestion pipeline (load → chunk → embed → Qdrant)
│   │   └── retriever.py               # Qdrant retriever (singleton)
│   └── tasks/
│       ├── __init__.py
│       ├── celery_app.py              # Celery application factory
│       └── workflow_task.py           # Celery task (wraps workflow)
├── knowledge_base/
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
│   ├── init.sql                       # PostgreSQL schema
│   └── data/
│       ├── report_agile_dashboard.csv
│       └── report_agile_dashboard_metrics.csv
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

# Переключитесь на ветку checkpoint_5
git checkout checkpoint_5

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

### Шаг 4: Загрузка базы знаний в Qdrant

```bash
docker compose run --rm app python -m hse_prom_prog.rag.ingest
```

Загружает PDF и Markdown документы из `knowledge_base/` в Qdrant. Без этого шага
RAG-запросы (вопросы о метриках, практиках, регламентах) не будут работать.

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
VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct
VLLM_API_KEY=EMPTY
VLLM_TEMPERATURE=0.7
VLLM_MAX_TOKENS=512

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
регламентах. Документы хранятся в `knowledge_base/` и загружаются в Qdrant через
ingestion pipeline.

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

После добавления или обновления документов в `knowledge_base/` необходимо
запустить ingestion pipeline для загрузки в Qdrant:

```bash
# Через Docker Compose (Qdrant должен быть запущен)
docker compose run --rm app python -m hse_prom_prog.rag.ingest

# Локально (Qdrant на localhost:6333)
poetry run python -m hse_prom_prog.rag.ingest

# С указанием пользовательского пути к документам
poetry run python -m hse_prom_prog.rag.ingest /path/to/docs
```

Pipeline выполняет:

1. **Загрузка** — читает .pdf (PyPDFLoader) и .md (TextLoader)
2. **Chunking** — разбивает на фрагменты (1000 символов, overlap 200)
3. **Эмбеддинг** — `intfloat/multilingual-e5-base` (CPU)
4. **Загрузка в Qdrant** — пересоздаёт коллекцию и загружает все чанки

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

### Шаг 3: Запуск vLLM

```bash
docker run -d --gpus all --name vllm-server -p 8000:8000 \
    vllm/vllm-openai:v0.8.5 \
    --model Qwen/Qwen2.5-3B-Instruct
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

# 3. (Опционально) Загрузить документы в Qdrant
docker compose run --rm app python -m hse_prom_prog.rag.ingest

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

# 2. (Опционально) Загрузить документы в Qdrant для RAG
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

### Структура манифестов

```
k8s/
├── namespace.yaml                # Namespace agile-assistant
├── Dockerfile.postgres           # Custom Postgres image (CSV data baked in)
├── ingress.yaml                  # Ingress (API rewrite + Streamlit/WebSocket)
├── configmaps/
│   ├── app-config.yaml           # Env vars (DNS names, credentials)
│   └── postgres-init.yaml        # init.sql (schema + COPY)
├── storage/
│   ├── postgres-pvc.yaml         # PVC 2Gi
│   ├── qdrant-pvc.yaml           # PVC 2Gi
│   └── vllm-cache-pvc.yaml      # PVC 10Gi (HuggingFace model cache)
├── deployments/
│   ├── postgres.yaml             # PostgreSQL 16 (custom image)
│   ├── qdrant.yaml               # Qdrant v1.13.2
│   ├── redis.yaml                # Redis 7 (ephemeral)
│   ├── vllm.yaml                 # vLLM (Qwen2.5-3B, GPU)
│   ├── api.yaml                  # FastAPI (gunicorn, 2 replicas)
│   ├── celery-worker.yaml        # Celery worker (threads, concurrency=4)
│   └── streamlit.yaml            # Streamlit UI
└── services/
    ├── postgres-svc.yaml         # ClusterIP :5432
    ├── qdrant-svc.yaml           # ClusterIP :6333, :6334
    ├── redis-svc.yaml            # ClusterIP :6379
    ├── vllm-svc.yaml             # ClusterIP :8000 (name: vllm-server)
    ├── api-svc.yaml              # ClusterIP :8080
    └── streamlit-svc.yaml        # ClusterIP :8501
```

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

### Шаг 2: Сборка образов в minikube

Переключите Docker-клиент на Docker daemon внутри minikube:

```bash
eval $(minikube docker-env)
```

Соберите образы:

```bash
# Основной образ приложения (API, Celery, Streamlit)
docker build -t agile-assistant:latest .

# Custom Postgres с CSV-данными
docker build -t agile-assistant-postgres:latest -f k8s/Dockerfile.postgres .
```

### Шаг 3: Развёртывание

Применяйте манифесты в порядке зависимостей:

```bash
# 1. Namespace + ConfigMaps + Storage
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmaps/
kubectl apply -f k8s/storage/

# 2. Инфраструктурные сервисы (БД, кеш, LLM)
kubectl apply -f k8s/services/
kubectl apply -f k8s/deployments/postgres.yaml
kubectl apply -f k8s/deployments/qdrant.yaml
kubectl apply -f k8s/deployments/redis.yaml
kubectl apply -f k8s/deployments/vllm.yaml

# 3. Дождитесь готовности инфраструктуры
kubectl -n agile-assistant wait --for=condition=ready pod -l app=postgres --timeout=120s
kubectl -n agile-assistant wait --for=condition=ready pod -l app=qdrant --timeout=120s
kubectl -n agile-assistant wait --for=condition=ready pod -l app=redis --timeout=60s

# 4. Миграции Alembic (таблица tasks)
cat <<'MIGRATE' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: migrate
  namespace: agile-assistant
spec:
  restartPolicy: Never
  containers:
    - name: migrate
      image: docker.io/library/agile-assistant:latest
      imagePullPolicy: Never
      command: ["alembic", "upgrade", "head"]
      envFrom:
        - configMapRef:
            name: app-config
MIGRATE
kubectl -n agile-assistant wait --for=jsonpath='{.status.phase}'=Succeeded pod/migrate --timeout=120s
kubectl -n agile-assistant logs migrate
kubectl -n agile-assistant delete pod migrate

# 5. Загрузка базы знаний в Qdrant
cat <<'INGEST' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: qdrant-ingest
  namespace: agile-assistant
spec:
  restartPolicy: Never
  containers:
    - name: qdrant-ingest
      image: docker.io/library/agile-assistant:latest
      imagePullPolicy: Never
      command: ["python", "-m", "hse_prom_prog.rag.ingest"]
      envFrom:
        - configMapRef:
            name: app-config
INGEST
# Ждите завершения (загрузка модели + индексация ~3-4 мин)
kubectl -n agile-assistant wait --for=jsonpath='{.status.phase}'=Succeeded pod/qdrant-ingest --timeout=300s
kubectl -n agile-assistant logs qdrant-ingest
kubectl -n agile-assistant delete pod qdrant-ingest

# 6. Приложение
kubectl apply -f k8s/deployments/api.yaml
kubectl apply -f k8s/deployments/celery-worker.yaml
kubectl apply -f k8s/deployments/streamlit.yaml

# 7. Настройка snippet-аннотаций и Ingress
kubectl -n ingress-nginx wait --for=condition=ready pod -l app.kubernetes.io/component=controller --timeout=120s
kubectl -n ingress-nginx patch configmap ingress-nginx-controller \
  --type merge \
  -p '{"data":{"allow-snippet-annotations":"true","annotations-risk-level":"Critical"}}'
kubectl -n ingress-nginx rollout restart deployment ingress-nginx-controller
kubectl -n ingress-nginx rollout status deployment ingress-nginx-controller
kubectl apply -f k8s/ingress.yaml
```

### Шаг 4: Проверка

```bash
# Статус всех подов
kubectl -n agile-assistant get pods

# Ожидаемый результат: все поды Running, READY 1/1
# vLLM может загружаться 10-15 минут при первом запуске (скачивание модели ~6.5GB + компиляция CUDA-графов)
# При последующих запусках модель берётся из PVC (vllm-cache)
```

### Шаг 5: Использование

Узнайте IP minikube:

```bash
minikube ip
```

Откройте в браузере:

```bash
# Streamlit UI
open http://$(minikube ip)

# Swagger UI
open http://$(minikube ip)/docs
```

Создайте задачу через API:

```bash
# Создание задачи
curl -s -X POST http://$(minikube ip)/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"query": "Расскажи о задаче AL-38787"}'

# Проверка статуса (подставьте task_id)
curl -s http://$(minikube ip)/api/tasks/<task_id> | python3 -m json.tool
```

### Маршруты Ingress

| Путь              | Сервис           | Особенности                              |
| ----------------- | ---------------- | ---------------------------------------- |
| `/api/*`          | `api:8080`       | Prefix `/api` удаляется (rewrite-target) |
| `/docs`, `/redoc` | `api:8080`       | Swagger UI                               |
| `/_stcore`        | `streamlit:8501` | WebSocket-эндпоинт Streamlit             |
| `/` (default)     | `streamlit:8501` | Streamlit UI                             |

### Полезные команды

```bash
# Логи конкретного сервиса
kubectl -n agile-assistant logs -l app=vllm --tail=50
kubectl -n agile-assistant logs -l app=celery-worker --tail=50

# Перезапуск деплоймента
kubectl -n agile-assistant rollout restart deployment/api

# Масштабирование API
kubectl -n agile-assistant scale deployment/api --replicas=3

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

```bash
# Запуск с дефолтным именем эксперимента (baseline)
poetry run python -m eval.run_eval

# Запуск с пользовательским именем
poetry run python -m eval.run_eval --experiment semantic_v2
```

Скрипт выполняет:

1. Загружает `golden_dataset.json`
2. Прогоняет каждый вопрос через RAG-пайплайн (retrieve → generate)
3. Вычисляет RAGAS-метрики (LLM-as-judge: GPT-5.2 через vsellm)
4. Сохраняет результаты в `eval/results/{experiment}_{timestamp}.json`
5. Выводит сводную таблицу в консоль

Формат результата:

```json
{
  "experiment": "baseline",
  "timestamp": "20260310_143000",
  "config": {
    "vllm_model": "Qwen/Qwen2.5-3B-Instruct",
    "embedding_model": "intfloat/multilingual-e5-base",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "retriever_top_k": 4
  },
  "aggregate": {
    "context_precision": 0.82,
    "faithfulness": 0.91,
    "answer_correctness": 0.75
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
Metric             baseline           semantic_v2        delta (last − first)
-----------------  -----------------  -----------------  --------------------
context_precision  0.8200             0.8900             +0.0700
faithfulness       0.9100             0.9300             +0.0200
answer_correctness 0.7500             0.7200             -0.0300

Config differences
param         baseline                 semantic_v2
------------  -----------------------  -----------------------
chunk_size    1000                     512
chunk_overlap 200                      100
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

| Переменная         | Описание              | По умолчанию               |
| ------------------ | --------------------- | -------------------------- |
| `VLLM_BASE_URL`    | URL vLLM API endpoint | `http://localhost:8000/v1` |
| `VLLM_MODEL`       | Название модели       | `Qwen/Qwen2.5-3B-Instruct` |
| `VLLM_API_KEY`     | API ключ для vLLM     | `EMPTY`                    |
| `VLLM_TEMPERATURE` | Temperature для LLM   | `0.7`                      |
| `VLLM_MAX_TOKENS`  | Максимум токенов      | `512`                      |

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

## Лицензия

MIT
