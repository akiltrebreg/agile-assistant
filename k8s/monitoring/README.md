# Kubernetes monitoring (ServiceMonitor / PodMonitor / PrometheusRule)

This folder is the k8s-native equivalent of
`monitoring/prometheus/prometheus.yml` and `monitoring/prometheus/alerts.yml`.
In docker-compose Prometheus loads those files directly from a bind mount; in
Kubernetes the
[Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator)
owns the Prometheus config and discovers scrape targets through CRDs.

## Why this exists

`monitoring/prometheus/prometheus.yml` uses `static_configs:` — a list of
hard-coded `host:port` pairs. That works in docker-compose where service names
are stable DNS, but it is an anti-pattern in Kubernetes:

- pods get fresh IPs on every reschedule;
- scaling a Deployment from 1 → 3 replicas does not add new scrape targets;
- adding a new component requires editing Prometheus config and reloading.

`ServiceMonitor` / `PodMonitor` solve this by pointing Prometheus at a **label
selector** instead of an IP. The operator watches Service / EndpointSlice / Pod
objects and rewrites Prometheus's runtime config on every change. Scaling
becomes free; pod restarts are invisible.

## Layout

```
k8s/monitoring/
├── kube-prometheus-stack-values.yaml   # Helm values for the operator chart
├── api-servicemonitor.yaml             # FastAPI               → job=fastapi
├── celery-worker-servicemonitor.yaml   # Celery main queue     → job=celery-worker
├── celery-judge-servicemonitor.yaml    # Celery judge queue    → job=celery-judge
├── vllm-main-servicemonitor.yaml       # vLLM avibe-gptq-8bit  → job=vllm-main
├── vllm-sql-servicemonitor.yaml        # vLLM Qwen3-8B-AWQ     → job=vllm-sql
├── qdrant-servicemonitor.yaml          # Qdrant                → job=qdrant
├── postgres-podmonitor.yaml            # CNPG pods             → job=postgresql
├── redis-exporter-servicemonitor.yaml  # redis_exporter        → job=redis
└── prometheus-rules.yaml               # 18 alerts (PrometheusRule)
```

Each CRD carries `release: kube-prometheus-stack` so the chart's default
selectors pick them up.

The `relabelings:` block on every endpoint forces a stable `job=` label that
matches the historical names from `prometheus.yml`. This is the only thing
keeping every PromQL query in `monitoring/grafana/dashboards/` and every alert
in `prometheus-rules.yaml` working without rewrites.

## Postgres: PodMonitor, not pg-exporter

In docker-compose we run a separate `prometheuscommunity/postgres-exporter`
container. In k8s the project uses [CloudNativePG](https://cloudnative-pg.io/),
which already exposes `/metrics` on every primary/replica pod via a named
`metrics` port — there is nothing to install, just to discover. `PodMonitor`
(vs. `ServiceMonitor`) targets pods directly so every instance is scraped on
failover, not only whatever is currently behind the `-rw` / `-ro` Service.

The PodMonitor selects on the CNPG-managed label
`cnpg.io/cluster: postgres-cluster` and surfaces the instance role (`primary` /
`replica`) as a `cnpg_role` label. Note that the `PostgreSQLReplicationLag`
alert in `prometheus-rules.yaml` queries `cnpg_pg_replication_lag` — the metric
CNPG exposes — instead of the `pg_replication_lag_seconds` from
postgres-exporter, so the alert actually fires under k8s.

## Install

```bash
# 1. Install the operator (one-time, cluster-wide)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  -f k8s/monitoring/kube-prometheus-stack-values.yaml

# 2. Apply our CRDs (re-runnable, declarative)
kubectl apply -f k8s/monitoring/

# 3. Verify scrape targets are UP
kubectl -n monitoring port-forward svc/kube-prometheus-stack-prometheus 9090:9090
# → open http://localhost:9090/targets and check every job is "UP"
```

## After install: what to delete

Once every job in Prometheus's `/targets` page reads `UP`:

- `monitoring/prometheus/prometheus.yml` — replaced by ServiceMonitors. The
  `prometheus.yml` is still used by docker-compose, so keep the file but stop
  trusting it as the k8s source of truth.
- `monitoring/prometheus/alerts.yml` — superseded by
  `k8s/monitoring/prometheus-rules.yaml` in k8s. Same caveat: keep it for
  docker-compose, but treat the CRD as authoritative for k8s.

The two configs MUST be edited together until docker-compose is retired. There
is no automated drift check.

## Adding a new component

1. Make sure the workload has a Service (`ClusterIP` or headless) with a
   **named** port — `port: http-metrics` is the convention here.
   ServiceMonitor's `endpoints[].port` matches by service-port name, not by
   number, so unnamed ports are invisible to the operator.
2. Drop a new `<name>-servicemonitor.yaml` next to the others, copy the closest
   existing template, and adjust:
   - `metadata.name` and labels;
   - `spec.selector.matchLabels` to match your Service labels;
   - `endpoints[].port` to the named metrics port;
   - the `relabelings:` block to set a stable `job=` label.
3. `kubectl apply -f k8s/monitoring/<name>-servicemonitor.yaml`.
4. Confirm in `/targets` that the new job appears as UP.
