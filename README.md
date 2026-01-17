# HSE Prom Prog - Agile AI Assistant

Multi-agent система для анализа Jira-задач с использованием LangGraph, vLLM и
PostgreSQL.

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
- [Разработка](#разработка)
  - [Установка dev-зависимостей](#установка-dev-зависимостей)
  - [Code Quality](#code-quality)
  - [Тестирование](#тестирование)
- [Конфигурация](#конфигурация)
- [Архитектурные решения](#архитектурные-решения)
- [Лицензия](#лицензия)

## Архитектура

Приложение построено на основе LangGraph и использует три последовательных
агента для обработки пользовательских запросов о Jira-задачах:

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Supervisor    │────▶│  SQL Agent   │────▶│ Response Agent  │
│  (извлекает     │     │  (mock data) │     │  (форматирует   │
│   issue_key)    │     │              │     │    ответ)       │
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

- Форматирует данные из БД в читаемые таблицы
- Использует `tabulate` для создания красивых таблиц
- Группирует информацию по категориям (Основная, Спринт, Команда, Метрики)
- Обрабатывает ошибки и отсутствующие данные

### LLM Backend

- **Модель**: Qwen/Qwen2.5-3B-Instruct
- **Backend**: vLLM с OpenAI-compatible API
- **URL**: `http://localhost:8000/v1`

### База данных

- **СУБД**: PostgreSQL 16 (Alpine)
- **Таблица**: `jira_issues` с 29 полями
- **Индексы**: issue_key (PRIMARY), sprint_state, assignee_name
- **Тестовые данные**: 5 задач (ABC-123, AXYZ-789, PROJ-456, DEV-999, TECH-555)

## Структура проекта

```
hse-prom-prog/
├── hse_prom_prog/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py          # Supervisor agent
│   │   ├── sql_agent.py           # SQL agent с PostgreSQL
│   │   └── response_agent.py      # Response agent
│   ├── database/
│   │   ├── __init__.py
│   │   └── connection.py          # PostgreSQL connection manager
│   ├── graph/
│   │   ├── __init__.py
│   │   └── workflow.py            # LangGraph StateGraph
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py              # OpenAI client для vLLM
│   ├── config.py                  # Pydantic settings
│   └── main.py                    # Entry point
├── database/
│   └── init.sql                   # PostgreSQL schema & test data
├── tests/
│   ├── __init__.py
│   └── test_workflow.py           # Тесты workflow
├── docker-compose.yml             # Docker Compose configuration
├── Dockerfile
├── .env.example
├── pyproject.toml
├── poetry.lock
└── README.md
```

## Быстрый старт с Docker Compose

Самый простой способ запустить весь стек (PostgreSQL + vLLM + приложение):

```bash
# Клонируйте репозиторий
git clone <repository-url>
cd hse-prom-prog

# Скопируйте example файл окружения
cp .env.example .env

# Запустите все сервисы
docker-compose up -d

# Проверьте статус сервисов
docker-compose ps

# Запустите приложение (в отдельном терминале после запуска сервисов)
docker-compose run --rm app python -m hse_prom_prog.main "Выведи данные по задаче ABC-123"

# Остановите сервисы
docker-compose down
```

**Что запускается:**

- PostgreSQL на порту 5432 с тестовыми данными
- vLLM сервер на порту 8000
- Приложение готово к использованию

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
    vllm/vllm-openai:v0.6.0 \
    --model Qwen/Qwen2.5-3B-Instruct
```

### Шаг 4: Запуск приложения

```bash
poetry run python -m hse_prom_prog.main "Привет! Выведи данные по задаче ABC-123"
```

## Использование

### Примеры запросов

```bash
# Через Docker Compose
docker-compose run --rm app python -m hse_prom_prog.main "Выведи данные по задаче ABC-123"

# Локально
poetry run python -m hse_prom_prog.main "Выведи данные по задаче ABC-123"

# Другие примеры
poetry run python -m hse_prom_prog.main "Информация о задаче AXYZ-789"
poetry run python -m hse_prom_prog.main "Покажи мне что там с PROJ-456"
poetry run python -m hse_prom_prog.main "DEV-999"
```

### Пример вывода

```
🤖 Agile AI Assistant
📝 Query: Выведи данные по задаче ABC-123

============================================================

[Supervisor] Извлекаю ключ задачи...
[Supervisor] Найден ключ: ABC-123

[SQL Agent] Выполняю запрос к PostgreSQL...
[SQL Agent] Данные успешно получены

[Response Agent] Форматирую ответ...

============================================================

=== ФИНАЛЬНЫЙ ОТВЕТ ===

## Информация по задаче ABC-123

### Основная информация

+--------------------+----------------------+
| Поле               | Значение             |
+====================+======================+
| Ключ задачи        | ABC-123              |
| Проект             | ABC Project          |
| Тип                | Story                |
| Статус             | In Progress          |
| Создана            | 2026-01-07 10:00     |
+--------------------+----------------------+

### Спринт

+------------------------+--------------------+
| Поле                   | Значение           |
+========================+====================+
| ID спринта             | 101                |
| Название спринта       | Sprint 25          |
| Состояние спринта      | active             |
| Начало                 | 2026-01-06 00:00   |
| Конец                  | 2026-01-17 23:59   |
+------------------------+--------------------+

### Команда

+------------------+-------------------------+
| Поле             | Значение                |
+==================+=========================+
| Команда          | Team Alpha              |
| Репортер         | Ivan Petrov             |
| Исполнитель      | Olga Komkova            |
| Кластер          | Product Development     |
| Подразделение    | Backend Team            |
+------------------+-------------------------+

### Метрики

+------------------------------+-----------+
| Поле                         | Значение  |
+==============================+===========+
| Story Points (начало)        | 5.0       |
| Story Points (конец)         | 5.0       |
| Время в работе (ч)           | 12        |
| Время не исправлено (ч)      | 3         |
| Резолюция                    | —         |
+------------------------------+-----------+

### Метки

`backend`, `api`

---
*Данные получены из PostgreSQL базы данных*

---
*Запрос обработан успешно*

============================================================
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

### Other

| Переменная  | Описание            | По умолчанию |
| ----------- | ------------------- | ------------ |
| `LOG_LEVEL` | Уровень логирования | `INFO`       |

## Лицензия

MIT
