[← README](../README.md) · Раздел: Локальная разработка

# Локальная разработка

## Требования

- Python 3.12+
- Poetry
- PostgreSQL 16 (или Docker для PostgreSQL)
- Qdrant (или Docker для Qdrant)
- vLLM (или Docker для vLLM)
- NVIDIA GPU (рекомендуется для vLLM)

## Шаг 1: Установка зависимостей

```bash
# Клонируйте репозиторий
git clone <repository-url>
cd agile-assistant

# Установите зависимости через Poetry
poetry install
```

## Шаг 2: Настройка PostgreSQL

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

## Шаг 3: Скачивание модели и запуск vLLM

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
    --model /models/avibe-gptq-8bit --quantization gptq --dtype float16 --max-model-len 6144
```

## Шаг 4: Запуск приложения

```bash
poetry run python -m agile_assistant.main "Привет! Выведи данные по задаче AL-38787"
```

## Связанные разделы

- [Конфигурация](configuration.md) — env-переменные (`localhost` для всех
  сервисов)
