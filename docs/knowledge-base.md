[← README](../README.md) · Раздел: База знаний и RAG

# База знаний и RAG

RAG Agent использует базу знаний для ответов на вопросы о теории, практиках и
регламентах. Документы хранятся в S3 (бакет `S3_KB_BUCKET`, по умолчанию
`knowledge-base`) и загружаются в Qdrant через ingestion pipeline. При
отсутствии S3-конфигурации используется локальная папка `knowledge_base/`.

## Структура `knowledge_base/`

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

## Загрузка документов в Qdrant

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

## Связанные разделы

- [Архитектура → Векторное хранилище (Qdrant)](architecture.md#векторное-хранилище-qdrant)
  — конфигурация коллекции, retriever, режимы поиска
- [Оценка → RAG-пайплайн (RAGAS)](evaluation.md#rag-пайплайн-ragas) — golden
  dataset и метрики качества RAG
