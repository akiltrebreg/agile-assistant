[← README](../README.md) · Раздел: Конфигурация

# Конфигурация

Все настройки управляются через переменные окружения (см.
[.env.example](../.env.example)).

## Содержание

- [vLLM Configuration](#vllm-configuration)
- [SQL vLLM Configuration](#sql-vllm-configuration)
- [PostgreSQL Configuration](#postgresql-configuration)
- [Qdrant Configuration](#qdrant-configuration)
- [S3 Configuration](#s3-configuration)
- [Embedding Configuration](#embedding-configuration)
- [Chunking Configuration](#chunking-configuration)
- [Search Configuration](#search-configuration)
- [Retriever Configuration](#retriever-configuration)
- [Reranker Configuration](#reranker-configuration)
- [Redis Configuration](#redis-configuration)
- [Celery Configuration](#celery-configuration)
- [FastAPI Configuration](#fastapi-configuration)
- [VSELLM Configuration (LLM-as-judge)](#vsellm-configuration-llm-as-judge)
- [Memory Layer Configuration](#memory-layer-configuration)
- [Monitoring](#monitoring)
- [Langfuse Tracing](#langfuse-tracing)
- [Other](#other)

## vLLM Configuration

| Переменная                | Описание                           | По умолчанию               |
| ------------------------- | ---------------------------------- | -------------------------- |
| `VLLM_BASE_URL`           | URL vLLM API endpoint              | `http://localhost:8000/v1` |
| `VLLM_MODEL`              | Название модели                    | `/models/avibe-gptq-8bit`  |
| `VLLM_API_KEY`            | API ключ для vLLM                  | `EMPTY`                    |
| `VLLM_TEMPERATURE`        | Temperature для LLM                | `0.0`                      |
| `VLLM_MAX_TOKENS`         | Максимум токенов                   | `600`                      |
| `VLLM_REPETITION_PENALTY` | Штраф за повторы (vLLM extra_body) | `1.1`                      |

> Для Docker Compose: `VLLM_BASE_URL=http://vllm:8000/v1`. Для Kubernetes —
> `http://vllm-server:8000/v1` (Service называется `vllm-server`, чтобы не
> конфликтовать с автогенерируемой `VLLM_PORT`).

## SQL vLLM Configuration

Отдельный vLLM-инстанс для SQL Agent (text-to-SQL через tool calling).

| Переменная          | Описание                        | По умолчанию                |
| ------------------- | ------------------------------- | --------------------------- |
| `SQL_VLLM_BASE_URL` | URL SQL vLLM API endpoint       | `http://localhost:8001/v1`  |
| `SQL_VLLM_MODEL`    | Название модели для text-to-SQL | `/models/qwen3-8b-awq-4bit` |

> Для Docker Compose: `SQL_VLLM_BASE_URL=http://vllm-sql:8000/v1`. Внутри
> контейнера vLLM слушает 8000, наружу маппится на 8001.

## PostgreSQL Configuration

| Переменная          | Описание        | По умолчанию   |
| ------------------- | --------------- | -------------- |
| `POSTGRES_HOST`     | Хост PostgreSQL | `localhost`    |
| `POSTGRES_PORT`     | Порт PostgreSQL | `5432`         |
| `POSTGRES_USER`     | Пользователь БД | `hse_user`     |
| `POSTGRES_PASSWORD` | Пароль БД       | `hse_password` |
| `POSTGRES_DB`       | Название БД     | `hse_jira_db`  |

## Qdrant Configuration

| Переменная               | Описание                                                                                         | По умолчанию            |
| ------------------------ | ------------------------------------------------------------------------------------------------ | ----------------------- |
| `QDRANT_URL`             | URL Qdrant сервера                                                                               | `http://localhost:6333` |
| `QDRANT_COLLECTION_NAME` | Название коллекции                                                                               | `business_docs`         |
| `EMBEDDING_MODEL`        | Имя папки в S3 / локальном кэше для embedding-модели (не Hub ID при заданном `S3_MODELS_BUCKET`) | `multilingual-e5-base`  |

## S3 Configuration

| Переменная                  | Описание                                      | По умолчанию                      |
| --------------------------- | --------------------------------------------- | --------------------------------- |
| `S3_ENDPOINT`               | S3 endpoint URL                               | `https://storage.yandexcloud.net` |
| `S3_BUCKET`                 | S3 bucket для основной vLLM-модели (avibe)    | `quant-models-agile`              |
| `S3_MODEL_PATH`             | Путь к avibe-модели внутри `S3_BUCKET`        | `models/avibe-gptq-8bit`          |
| `S3_KB_BUCKET`              | S3 bucket для базы знаний                     | `knowledge-base`                  |
| `S3_KB_PATH`                | Путь внутри bucket для базы знаний            | `knowledge_base`                  |
| `S3_DATA_BUCKET`            | S3 bucket для CSV данных                      | `database-agile`                  |
| `S3_DATA_PATH`              | Путь внутри bucket для CSV                    | `data`                            |
| `S3_MODELS_BUCKET`          | S3 bucket для ML-моделей (embedding/reranker) | `quant-models-agile`              |
| `S3_MODELS_PATH`            | Путь внутри bucket для моделей                | `models`                          |
| `EMBEDDING_MODEL_CACHE_DIR` | Локальный кэш моделей                         | `/app/models`                     |
| `AWS_ACCESS_KEY_ID`         | Ключ доступа Yandex Cloud                     | —                                 |
| `AWS_SECRET_ACCESS_KEY`     | Секретный ключ Yandex Cloud                   | —                                 |
| `AWS_DEFAULT_REGION`        | Регион Yandex Cloud                           | `ru-central1`                     |

> `S3_BUCKET` / `S3_MODEL_PATH` используются только compose-job'ом
> `download-model` для скачивания основной vLLM-модели. Для embedding и reranker
> отдельные `S3_MODELS_BUCKET` / `S3_MODELS_PATH` (могут указывать на тот же
> bucket, но через разные переменные).

## Embedding Configuration

| Переменная               | Описание                                        | По умолчанию |
| ------------------------ | ----------------------------------------------- | ------------ |
| `EMBEDDING_SPARSE_MODEL` | Sparse модель: `None` (BM25) или `BAAI/bge-m3`  | —            |
| `EMBEDDING_DIMENSION`    | Matryoshka-truncation (64/128/256/512/768/1024) | —            |

## Chunking Configuration

| Переменная          | Описание                          | По умолчанию |
| ------------------- | --------------------------------- | ------------ |
| `CHUNK_SIZE`        | Размер чанка для разбиения текста | `500`        |
| `CHUNK_OVERLAP`     | Перекрытие между чанками          | `200`        |
| `MAX_CONTEXT_CHARS` | Макс. символов контекста для LLM  | `4000`       |

## Search Configuration

| Переменная    | Описание                                    | По умолчанию |
| ------------- | ------------------------------------------- | ------------ |
| `SEARCH_TYPE` | Режим поиска: `dense`, `sparse`, `hybrid`   | `dense`      |
| `RRF_K`       | Параметр k для RRF fusion (только `hybrid`) | `60`         |

## Retriever Configuration

| Переменная            | Описание                               | По умолчанию |
| --------------------- | -------------------------------------- | ------------ |
| `RETRIEVER_TOP_K`     | Финальное число чанков (без реранкера) | `4`          |
| `RETRIEVER_INITIAL_K` | Число чанков для первичного извлечения | `20`         |

## Reranker Configuration

| Переменная           | Описание                                                                                               | По умолчанию         |
| -------------------- | ------------------------------------------------------------------------------------------------------ | -------------------- |
| `RERANKER_ENABLED`   | Включить cross-encoder reranking                                                                       | `true`               |
| `RERANKER_MODEL`     | Имя папки в S3 / локальном кэше для cross-encoder reranker (не Hub ID при заданном `S3_MODELS_BUCKET`) | `bge-reranker-v2-m3` |
| `RERANKER_THRESHOLD` | Минимальный скор для фильтрации чанков                                                                 | `0.01`               |
| `RERANKER_TOP_N`     | Число чанков после реранкинга                                                                          | `4`                  |

## Redis Configuration

| Переменная       | Описание         | По умолчанию |
| ---------------- | ---------------- | ------------ |
| `REDIS_HOST`     | Хост Redis       | `localhost`  |
| `REDIS_PORT`     | Порт Redis       | `6379`       |
| `REDIS_DB`       | Номер базы Redis | `0`          |
| `REDIS_PASSWORD` | Пароль Redis     | —            |

## Celery Configuration

| Переменная                    | Описание                    | По умолчанию |
| ----------------------------- | --------------------------- | ------------ |
| `CELERY_BROKER_URL`           | URL брокера (авто из Redis) | —            |
| `CELERY_TASK_TIME_LIMIT`      | Hard timeout задачи (сек)   | `600`        |
| `CELERY_TASK_SOFT_TIME_LIMIT` | Soft timeout задачи (сек)   | `300`        |

## FastAPI Configuration

| Переменная        | Описание                        | По умолчанию |
| ----------------- | ------------------------------- | ------------ |
| `FASTAPI_HOST`    | Хост FastAPI сервера            | `0.0.0.0`    |
| `FASTAPI_PORT`    | Порт FastAPI сервера            | `8080`       |
| `FASTAPI_WORKERS` | Кол-во uvicorn worker-процессов | `1`          |
| `CORS_ORIGINS`    | Разрешённые CORS origins        | `*`          |

## VSELLM Configuration (LLM-as-judge)

| Переменная        | Описание                                                                    | По умолчанию               |
| ----------------- | --------------------------------------------------------------------------- | -------------------------- |
| `VSELLM_API_KEY`  | API ключ для vsellm (RAGAS + judge-таска)                                   | —                          |
| `VSELLM_BASE_URL` | URL vsellm API endpoint                                                     | `https://api.vsellm.ru/v1` |
| `JUDGE_ENABLED`   | Master kill-switch для LLM-as-a-Judge. `false` → judge-таски не запускаются | `true`                     |

## Memory Layer Configuration

| Переменная                | По умолчанию | Что делает                                                             |
| ------------------------- | ------------ | ---------------------------------------------------------------------- |
| `HISTORY_TOKEN_BUDGET`    | `800`        | Бюджет токенов на блок `<conversation_history>` в промптах             |
| `SESSION_TIMEOUT_MINUTES` | `30`         | Через сколько минут idle диалог автоматически закрывается и ротируется |

Подробное описание поведения (sliding window, inactivity rotation, anaphora
carry-forward) — в [memory.md → Конфигурация](memory.md#конфигурация).

## Monitoring

| Переменная         | Описание                                             | По умолчанию |
| ------------------ | ---------------------------------------------------- | ------------ |
| `GRAFANA_PASSWORD` | Пароль admin-пользователя в Grafana (логин: `admin`) | `admin`      |

## Langfuse Tracing

**Client-side** (читаются проектом через Pydantic settings):

| Переменная            | Описание                                                               | По умолчанию           |
| --------------------- | ---------------------------------------------------------------------- | ---------------------- |
| `LANGFUSE_PUBLIC_KEY` | Public key проекта Langfuse (`pk-lf-...`). Пусто = не отправлять spans | —                      |
| `LANGFUSE_SECRET_KEY` | Secret key проекта Langfuse (`sk-lf-...`)                              | —                      |
| `LANGFUSE_HOST`       | URL Langfuse-сервера (внутри docker-network)                           | `http://langfuse:3000` |
| `LANGFUSE_ENABLED`    | Master kill-switch для SDK. `false` → `@observe` становится no-op      | `true`                 |

**Server-side** (читаются только Langfuse-контейнером, не проектом):

| Переменная                 | Описание                                                                                  | По умолчанию             |
| -------------------------- | ----------------------------------------------------------------------------------------- | ------------------------ |
| `LANGFUSE_NEXTAUTH_URL`    | URL, по которому пользователь открывает Langfuse в браузере (важно для NextAuth callback) | `http://localhost:3001`  |
| `LANGFUSE_NEXTAUTH_SECRET` | Секрет NextAuth для подписи сессий                                                        | `changeme-in-production` |
| `LANGFUSE_SALT`            | Salt для шифрования API-ключей в БД Langfuse                                              | `changeme-in-production` |

> ⚠ Для production-деплоя `LANGFUSE_NEXTAUTH_SECRET` и `LANGFUSE_SALT`
> **обязательно** заменить на случайные значения через `openssl rand -hex 32`.
> Дефолт `changeme-in-production` означает, что любой, кто знает это значение,
> может подделать админ-сессию Langfuse и расшифровать API-ключи.

## Other

| Переменная          | Описание                                                    | По умолчанию |
| ------------------- | ----------------------------------------------------------- | ------------ |
| `DEBUG`             | Debug-режим (hot-reload, verbose log)                       | `false`      |
| `LOG_LEVEL`         | Уровень логирования                                         | `INFO`       |
| `GUARDRAIL_ENABLED` | Включить L1 TopicGuard (regex prompt-injection + whitelist) | `true`       |

## Связанные разделы

- [Memory Layer → Конфигурация](memory.md#конфигурация) — env-переменные памяти
  и их поведение
- [Observability → Langfuse Tracing](observability.md#трейсинг-langfuse) —
  детали SDK и server-side env
