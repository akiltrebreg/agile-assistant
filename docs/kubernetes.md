[← README](../README.md) · Раздел: Kubernetes (minikube)

# Kubernetes (minikube)

Проект можно развернуть в Kubernetes с помощью minikube. Все манифесты находятся
в [k8s/](../k8s/).

## Требования

- [minikube](https://minikube.sigs.k8s.io/) v1.38+
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- Docker (driver для minikube)
- NVIDIA GPU + драйверы (для vLLM)
- [CloudNativePG](https://cloudnative-pg.io/) v1.25+ (устанавливается на шаге 1)

## Структура манифестов

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
│   ├── redis-exporter.yaml       # oliver006/redis_exporter (порт 9121, для ServiceMonitor)
│   ├── vllm.yaml                 # vLLM main (avibe-gptq-8bit, GPU, S3 init)
│   ├── vllm-sql.yaml             # vLLM SQL (Qwen3-8B-AWQ, GPU, S3 init)
│   ├── api.yaml                  # FastAPI (gunicorn, 2 replicas)
│   ├── celery-worker.yaml        # Celery main worker (threads, concurrency=1, side-car :9100)
│   ├── celery-judge.yaml         # Celery judge worker (queue=judge, side-car :9101)
│   ├── celery-beat.yaml          # Celery Beat scheduler (replicas:1 + Recreate)
│   └── streamlit.yaml            # Streamlit UI
├── services/
│   ├── qdrant-svc.yaml               # ClusterIP :6333, :6334
│   ├── redis-svc.yaml                # ClusterIP :6379
│   ├── redis-exporter-svc.yaml       # ClusterIP :9121 (port name: http-metrics)
│   ├── vllm-svc.yaml                 # ClusterIP :8000 (name: vllm-server)
│   ├── vllm-sql-svc.yaml             # ClusterIP :8000 (Service: vllm-sql)
│   ├── api-svc.yaml                  # ClusterIP :8080 (port name: http)
│   ├── celery-worker-svc.yaml        # Headless (clusterIP: None) — для ServiceMonitor discovery
│   ├── celery-judge-svc.yaml         # Headless — для ServiceMonitor discovery
│   └── streamlit-svc.yaml            # ClusterIP :8501
└── monitoring/                       # Prometheus Operator CRD (заменяют static_configs из docker-compose)
    ├── kube-prometheus-stack-values.yaml  # Helm values для оператора
    ├── api-servicemonitor.yaml            # → job=fastapi
    ├── celery-worker-servicemonitor.yaml  # → job=celery-worker
    ├── celery-judge-servicemonitor.yaml   # → job=celery-judge
    ├── vllm-main-servicemonitor.yaml      # → job=vllm-main
    ├── vllm-sql-servicemonitor.yaml       # → job=vllm-sql
    ├── qdrant-servicemonitor.yaml         # → job=qdrant
    ├── postgres-podmonitor.yaml           # PodMonitor по cnpg.io/cluster, job=postgresql
    ├── redis-exporter-servicemonitor.yaml # → job=redis
    ├── prometheus-rules.yaml              # PrometheusRule CRD — те же 18 алертов, что в alerts.yml
    └── README.md                          # Установка оператора + миграция с static_configs
```

## Архитектурные решения

- **PostgreSQL HA** — управляется оператором CloudNativePG. Кластер из 3
  инстансов (1 primary + 2 standby) со streaming replication и автоматическим
  failover. Оператор создаёт Service-ы `postgres-cluster-rw` (запись) и
  `postgres-cluster-ro` (чтение). Приложение подключается через
  `postgres-cluster-rw`.
- **Secrets** — pattern: 4 Secret-а (`app-secrets`, `postgres-credentials`,
  `registry-credentials`, `basic-auth`). Конкретные команды создания — ниже в
  шаге 2.
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

## Шаг 1: Запуск minikube с GPU

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

## Шаг 2: Сборка и публикация образа

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

## Шаг 3: Развёртывание

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
kubectl apply -f k8s/services/               # qdrant, redis, redis-exporter, vllm-server, vllm-sql, api, streamlit + headless celery-worker / celery-judge для ServiceMonitor
```

> Service-ы применяются до Deployment-ов, чтобы DNS-имена были доступны при
> старте контейнеров. PostgreSQL Service создаётся автоматически оператором
> CloudNativePG.

**3.3. Инфраструктура** — БД, векторное хранилище, кеш, LLM:

```bash
kubectl apply -f k8s/statefulsets/postgres-cluster.yaml   # CloudNativePG: 1 primary + 2 standby
kubectl apply -f k8s/deployments/qdrant.yaml              # Qdrant v1.13.2
kubectl apply -f k8s/deployments/redis.yaml               # Redis 7 (брокер Celery)
kubectl apply -f k8s/deployments/redis-exporter.yaml      # Side-car exporter для redis-метрик
kubectl apply -f k8s/deployments/vllm.yaml                # vLLM main: avibe-gptq-8bit (GPU, S3 download)
kubectl apply -f k8s/deployments/vllm-sql.yaml            # vLLM SQL: Qwen3-8B-AWQ (GPU, S3 download)
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
kubectl apply -f k8s/deployments/celery-worker.yaml   # Celery main worker (embedding-модель, 4Gi RAM)
kubectl apply -f k8s/deployments/celery-judge.yaml    # Celery judge worker (queue=judge, vsellm)
kubectl apply -f k8s/deployments/celery-beat.yaml     # Celery Beat (replicas:1 + Recreate, периодический sync)
kubectl apply -f k8s/deployments/streamlit.yaml        # Streamlit UI
```

**3.7. Ingress** — маршрутизация внешнего трафика:

```bash
kubectl apply -f k8s/ingress.yaml    # 4 Ingress-ресурса: API, static-svg, static-css, UI
```

**3.8. Monitoring (опционально, Prometheus Operator)** — заменяет
`static_configs` из `monitoring/prometheus/prometheus.yml` на ServiceMonitor /
PodMonitor / PrometheusRule CRD. Подробности в
[k8s/monitoring/README.md](../k8s/monitoring/README.md):

```bash
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  -f k8s/monitoring/kube-prometheus-stack-values.yaml
kubectl apply -f k8s/monitoring/    # 7 ServiceMonitor + 1 PodMonitor + 1 PrometheusRule
```

## Шаг 4: Проверка

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

## Шаг 5: Использование

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

## Маршруты Ingress

| Путь              | Сервис           | Особенности                                         |
| ----------------- | ---------------- | --------------------------------------------------- |
| `/api/*`          | `api:8080`       | Prefix `/api` удаляется (rewrite-target)            |
| `/static/*.svg`   | `streamlit:8501` | Content-Type: image/svg+xml (configuration-snippet) |
| `/static/*.css`   | `streamlit:8501` | Content-Type: text/css (configuration-snippet)      |
| `/docs`, `/redoc` | `api:8080`       | Swagger UI                                          |
| `/_stcore`        | `streamlit:8501` | WebSocket-эндпоинт Streamlit                        |
| `/` (default)     | `streamlit:8501` | Streamlit UI                                        |

## Полезные команды

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

## Связанные разделы

- [Конфигурация](configuration.md) — env-переменные (особенно про коллизию
  `VLLM_PORT`)
- [Observability](observability.md) — Prometheus Operator + ServiceMonitor /
  PrometheusRule CRD
