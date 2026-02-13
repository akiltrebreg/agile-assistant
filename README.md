# HSE Prom Prog - Agile AI Assistant (Задание 3)

Multi-agent система для анализа Jira-задач с использованием LangGraph, vLLM и
PostgreSQL, Celery и Redis.

## Содержание

- [Архитектура](#архитектура)
  - [Компоненты](#компоненты)
  - [LLM Backend](#llm-backend)
  - [База данных](#база-данных)
- [Структура проекта](#структура-проекта)
- [Быстрый старт с Docker Compose](#быстрый-старт-с-docker-compose)
  - [Настройка переменных окружения](#настройка-переменных-окружения)
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
- [Разработка](#разработка)
  - [Установка dev-зависимостей](#установка-dev-зависимостей)
  - [Code Quality](#code-quality)
  - [Тестирование](#тестирование)
- [Конфигурация](#конфигурация)
- [Лицензия](#лицензия)

## Архитектура

Приложение построено на основе LangGraph и использует три последовательных
агента для обработки пользовательских запросов о Jira-задачах:

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Supervisor    │────▶│  SQL Agent   │────▶│ Response Agent  │
│  (извлекает     │     │ (PostgreSQL) │     │  (генерирует    │
│   issue_key)    │     │              │     │  ответ с LLM)   │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

### Компоненты

**1. Supervisor Agent**

- Извлекает ключ Jira-задачи (например, "ABC-123") из пользовательского запроса
- Использует regex + LLM для надежного извлечения
- Передает issue_key следующему агенту

**2. SQL Agent**

- Принимает issue_key от Supervisor
- Выполняет SQL-запросы к PostgreSQL базе данных
- Использует SQLAlchemy для безопасной работы с БД
- Возвращает полную информацию о Jira-задаче

**3. Response Agent**

- Использует LLM для генерации естественноязычных ответов
- Анализирует данные из БД и оригинальный запрос пользователя
- Формирует контекстуальный ответ, релевантный запросу
- Фокусируется на конкретной информации, запрошенной пользователем
- Обрабатывает ошибки и отсутствующие данные

### LLM Backend

- **Модель**: Qwen/Qwen2.5-3B-Instruct
- **Backend**: vLLM с OpenAI-compatible API
- **URL**: `http://localhost:8000/v1`

### База данных

- **СУБД**: PostgreSQL 16 (Alpine)
- **Таблицы**:
  - `report_agile_dashboard` - основная таблица с данными по задачам (58 полей)
  - `report_agile_dashboard_metrics` - метрики спринтов (35 полей)
- **Индексы**: issue_key, jirasprint_id, sprint_state, assignee_name,
  feature_teams
- **Данные**: Загружаются из CSV файлов с использованием команды COPY

## Структура проекта

```
hse-prom-prog/
├── hse_prom_prog/
│   ├── __init__.py
│   ├── config.py                      # Pydantic settings
│   ├── main.py                        # CLI entry point
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py              # Supervisor agent
│   │   ├── sql_agent.py              # SQL agent (PostgreSQL)
│   │   └── response_agent.py         # Response agent (LLM)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                    # FastAPI application
│   │   ├── dependencies.py           # DI (DB, repo)
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   └── tasks.py              # POST/GET /tasks endpoints
│   │   └── schemas/
│   │       ├── __init__.py
│   │       └── task.py               # Pydantic request/response
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py             # PostgreSQL connection manager
│   │   └── task_repository.py        # Task CRUD (raw SQL)
│   ├── graph/
│   │   ├── __init__.py
│   │   └── workflow.py               # LangGraph StateGraph
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py                 # OpenAI client для vLLM
│   ├── models/
│   │   ├── __init__.py
│   │   └── task.py                   # TaskStatus enum, Task model
│   └── tasks/
│       ├── __init__.py
│       ├── celery_app.py             # Celery application factory
│       └── workflow_task.py          # Celery task (wraps workflow)
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
├── tests/
│   ├── __init__.py
│   └── test_workflow.py
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

Самый простой способ запустить весь стек (PostgreSQL + vLLM + Redis + FastAPI +
Celery + приложение):

```bash
# Клонируйте репозиторий
git clone <repository-url>
cd hse-prom-prog

# Скачайте все ветки из удаленного репозитория
git fetch --all

# Переключитесь на ветку checkpoint_3
git checkout checkpoint_3

# Скопируйте example файл окружения
cp .env.example .env

# Запустите все сервисы
docker compose up -d

# Проверьте статус сервисов
docker compose ps

# Запустите приложение (в отдельном терминале после запуска сервисов)
docker compose run --rm app python -m hse_prom_prog.main "Выведи данные по задаче AL-38787"

# Остановите сервисы
docker compose down
```

**Что запускается:**

- PostgreSQL на порту 5432 с тестовыми данными
- vLLM сервер на порту 8000
- Redis на порту 6380 (брокер Celery)
- FastAPI API на порту 8080
- Celery worker (обработка задач)
- Alembic миграции (таблица tasks)
- CLI-приложение готово к использованию

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

# Logging
LOG_LEVEL=INFO
```

**Примечания:**

- Для Docker Compose используйте `VLLM_BASE_URL=http://vllm:8000/v1` и
  `POSTGRES_HOST=postgres`
- Для локальной разработки используйте `localhost` для обоих сервисов
- Значения по умолчанию подходят для большинства случаев использования

## Локальная разработка

### Требования

- Python 3.12+
- Poetry
- PostgreSQL 16 (или Docker для PostgreSQL)
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

# Другие примеры
poetry run python -m hse_prom_prog.main "Информация о задаче AL-38799"
poetry run python -m hse_prom_prog.main "Покажи мне что там с AL-39041"
poetry run python -m hse_prom_prog.main "AL-39043"
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

### Запуск async-стека

```bash
# 1. Поднять инфраструктуру
docker compose up -d postgres redis vllm

# 2. Дождаться готовности и применить миграции Alembic
docker compose run --rm migrate

# 3. Запустить API и Celery-воркер
docker compose up -d api celery-worker

# 4. Проверить, что все сервисы healthy
docker compose ps
```

### Создание задачи

```bash
curl -s -X POST http://localhost:8080/tasks \
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
curl -s http://localhost:8080/tasks/<task_id> | python3 -m json.tool
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
    "final_response": "Задача AL-38787: ...",
    "issue_key": "AL-38787"
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
  curl -s -X POST http://localhost:8080/tasks \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"Расскажи о задаче AL-$i\"}" &
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

### Other

| Переменная  | Описание            | По умолчанию |
| ----------- | ------------------- | ------------ |
| `LOG_LEVEL` | Уровень логирования | `INFO`       |

## Лицензия

MIT
