# HSE Prom Prog - Agile AI Assistant (Задание 1)

Базовый multi-agent прототип для анализа Jira-задач с использованием LangGraph и vLLM.

## Архитектура

Приложение построено на основе LangGraph и использует три последовательных агента для обработки пользовательских запросов о Jira-задачах:

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

**2. SQL Agent (заглушка)**
- Принимает issue_key от Supervisor
- Возвращает mock-данные вместо реального SQL-запроса
- В следующем задании будет подключен к PostgreSQL

**3. Response Agent**
- Форматирует финальный ответ в markdown
- Создает user-friendly вывод

### LLM Backend

- **Модель**: Qwen/Qwen2.5-3B-Instruct (4-bit квантизация)
- **Backend**: vLLM с OpenAI-compatible API
- **URL**: `http://localhost:8000/v1`

## Структура проекта

```
hse-prom-prog/
├── hse_prom_prog/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py          # Supervisor agent
│   │   ├── sql_agent.py           # SQL agent с заглушкой
│   │   └── response_agent.py      # Response agent
│   ├── graph/
│   │   ├── __init__.py
│   │   └── workflow.py            # LangGraph StateGraph
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py              # OpenAI client для vLLM
│   ├── config.py                  # Pydantic settings
│   └── main.py                    # Entry point
├── tests/
│   ├── __init__.py
│   └── test_workflow.py           # Тесты workflow
├── Dockerfile
├── .env.example
├── pyproject.toml
├── poetry.lock
└── README.md
```

## Установка

### Требования

- Python 3.12+
- Poetry
- vLLM (локально или через Docker)
- NVIDIA GPU (опционально, для vLLM с GPU)

### Шаг 1: Клонирование и установка зависимостей

```bash
# Клонируйте репозиторий
git clone <repository-url>
cd hse-prom-prog

# Установите зависимости через Poetry
poetry install
```

### Шаг 2: Настройка переменных окружения

```bash
# Скопируйте example файл
cp .env.example .env

# Отредактируйте .env при необходимости
nano .env
```

### Шаг 3: Запуск vLLM

#### Вариант A: Локальный запуск vLLM (рекомендуется для разработки)

```bash
# Установите vLLM
pip install vllm

# Запустите vLLM сервер
vllm serve Qwen/Qwen2.5-3B-Instruct \
    --quantization gptq \
    --dtype half \
    --max-model-len 2048 \
    --port 8000
```

#### Вариант B: Запуск vLLM в Docker

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-3B-Instruct \
    --quantization gptq \
    --dtype half \
    --max-model-len 2048
```

#### Вариант C: CPU-only запуск (медленнее, но без GPU)

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
    --dtype float16 \
    --max-model-len 2048 \
    --port 8000
```

### Шаг 4: Проверка vLLM

```bash
# Проверьте, что vLLM работает
curl http://localhost:8000/v1/models
```

## Использование

### Базовое использование

```bash
# Активируйте Poetry окружение
poetry shell

# Запустите приложение
python -m hse_prom_prog.main "Привет! Выведи данные по задаче ABC-123"
```

### Примеры запросов

```bash
# Пример 1: Простой запрос
python -m hse_prom_prog.main "Выведи данные по задаче XYZ-789"

# Пример 2: Запрос с дополнительным текстом
python -m hse_prom_prog.main "Привет! Мне нужна информация о задаче PROJ-456"

# Пример 3: Неформальный запрос
python -m hse_prom_prog.main "Покажи мне что там с DEV-999"
```

### Пример вывода

```
🤖 Agile AI Assistant
📝 Query: Привет! Выведи данные по задаче ABC-123

============================================================

[Supervisor] Извлекаю ключ задачи из запроса...
[Supervisor] Найден ключ: ABC-123

[SQL Agent] Обработка задачи ABC-123...
[SQL Agent] Привет! Во втором задании я научусь отправлять запрос в Postgres и пришлю данные по задаче ABC-123!

[Response Agent] Формирование ответа...

============================================================

=== ФИНАЛЬНЫЙ ОТВЕТ ===

## Результат обработки задачи ABC-123

Привет! Во втором задании я научусь отправлять запрос в Postgres и пришлю данные по задаче ABC-123!

---
*Запрос обработан успешно*

============================================================
```

## Запуск через Docker

### Сборка образа

```bash
# Сборка Docker образа
docker build -t hse-prom-prog .
```

### Запуск контейнера

```bash
# Запуск с дефолтным запросом
docker run --rm hse-prom-prog

# Запуск с кастомным запросом
docker run --rm hse-prom-prog python -m hse_prom_prog.main "Выведи данные по задаче XYZ-123"

# Запуск с подключением к vLLM на хосте
docker run --rm \
    -e VLLM_BASE_URL=http://host.docker.internal:8000/v1 \
    hse-prom-prog
```

## Тестирование

```bash
# Запустите тесты
poetry run pytest tests/

# С подробным выводом
poetry run pytest tests/ -v

# С покрытием кода
poetry run pytest tests/ --cov=hse_prom_prog
```

## Разработка

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

## Конфигурация

Все настройки управляются через переменные окружения (см. [.env.example](.env.example)):

| Переменная | Описание | По умолчанию |
|-----------|----------|--------------|
| `VLLM_BASE_URL` | URL vLLM API endpoint | `http://localhost:8000/v1` |
| `VLLM_MODEL` | Название модели | `Qwen/Qwen2.5-3B-Instruct` |
| `VLLM_API_KEY` | API ключ для vLLM | `EMPTY` |
| `VLLM_TEMPERATURE` | Temperature для LLM | `0.7` |
| `VLLM_MAX_TOKENS` | Максимум токенов | `512` |
| `LOG_LEVEL` | Уровень логирования | `INFO` |

## Архитектурные решения

### Почему LangGraph?

- Декларативное описание workflow через граф
- Встроенная поддержка state management
- Простая интеграция с LangChain
- Легко добавлять новых агентов

### Почему vLLM?

- Высокая производительность (continuous batching, PagedAttention)
- OpenAI-compatible API
- Поддержка квантизации (GPTQ, AWQ)
- Production-ready deployment

### Type Safety

- Все функции имеют type hints
- Pydantic для валидации настроек и состояния
- Строгая типизация через `TypedDict` для state

## Следующие шаги (Задание 2)

В следующем задании будет добавлено:

1. Подключение к PostgreSQL для хранения Jira-данных
2. Реальные SQL-запросы вместо mock-данных
3. Расширенная обработка данных из БД
4. Больше агентов для анализа задач

## Лицензия

MIT

## Авторы

HSE Prom Prog Team
