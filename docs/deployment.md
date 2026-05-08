[← README](../README.md) · Раздел: Nginx + Production

# Nginx + Production

Nginx выступает единой точкой входа (порт 80). FastAPI, Streamlit, Prometheus и
Grafana не имеют внешних портов и доступны только через reverse proxy.
Внутренние upstream-ы (`fastapi`, `streamlit`, `prometheus`, `grafana`,
`langfuse`) объявлены в [nginx/nginx.conf](../nginx/nginx.conf). Для тяжёлых JS
бандлов Grafana настроены увеличенные `proxy_buffers` и gzip — без них первая
загрузка дашбордов уходила в 30-60 секунд из-за дисковой буферизации.

## Архитектура контейнеров

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

| Контейнер          | Образ                                   | Роль                                                                  |
| ------------------ | --------------------------------------- | --------------------------------------------------------------------- |
| **nginx**          | `nginx:1.27-alpine`                     | Reverse proxy, единственный открытый порт (80)                        |
| **api**            | Dockerfile + gunicorn                   | FastAPI в production (gunicorn + UvicornWorker), экспонирует /metrics |
| **celery-worker**  | Dockerfile + celery threads             | Workflow-исполнитель, side-car `start_http_server(9100)` для метрик   |
| **static**         | `busybox` + volume                      | Хранит статические файлы, шарит volume с nginx                        |
| **qdrant**         | `qdrant:v1.13.2`                        | Векторное хранилище для RAG Agent                                     |
| **prometheus**     | `prom/prometheus:v2.53.0`               | Time-series для метрик; subpath `/prometheus/` за nginx               |
| **grafana**        | `grafana/grafana:11.1.0`                | Дашборды поверх Prometheus; subpath `/grafana/` за nginx              |
| **pg-exporter**    | `prometheuscommunity/postgres-exporter` | Side-car для PostgreSQL метрик                                        |
| **redis-exporter** | `oliver006/redis_exporter`              | Side-car для Redis метрик                                             |
| **langfuse**       | `langfuse/langfuse:2`                   | LLM-tracing UI + API; пишет в `langfuse_db` (тот же PostgreSQL)       |

## Маршруты nginx

| Путь              | Куда проксирует   | Особенности                                                                                   |
| ----------------- | ----------------- | --------------------------------------------------------------------------------------------- |
| `/api/*`          | `api:8080`        | Prefix `/api` удаляется (`/api/tasks` -> `/tasks`); `/api/metrics` отдаёт 404 (internal-only) |
| `/static/*`       | Диск / Streamlit  | Сначала volume, затем fallback на Streamlit                                                   |
| `/docs`, `/redoc` | `api:8080`        | Swagger UI                                                                                    |
| `/_stcore/stream` | `streamlit:8501`  | WebSocket (Upgrade + Connection headers)                                                      |
| `/prometheus/`    | `prometheus:9090` | Prometheus UI (gzip, large proxy_buffers для JS-бандлов)                                      |
| `/grafana/`       | `grafana:3000`    | Grafana UI (gzip, WebSocket для Live, large proxy_buffers)                                    |
| `/langfuse/`      | `langfuse:3000`   | Langfuse UI (опционально; основной доступ — `localhost:3001`)                                 |
| `/` (default)     | `streamlit:8501`  | Streamlit UI с WebSocket-поддержкой                                                           |

## Запуск production-стека

```bash
# 1. Запустить всё
docker compose up -d

# 2. Загрузить данные из S3 (при первом запуске)
docker compose run --rm load-data
docker compose run --rm app python -m agile_assistant.rag.ingest

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

## Проверка

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

## Связанные разделы

- [Конфигурация](configuration.md) — env-переменные для production-стека
