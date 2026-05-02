[← README](../README.md) · Раздел: База данных

# База данных — схема и инспекция

Полная схема PostgreSQL после применения миграций + рабочие psql-сниппеты для
проверки профиля, сессий и окна памяти.

## Структура БД

После применения миграций (`docker compose run --rm migrate`) в схеме `public`
появляются шесть таблиц.

| Таблица                  | Миграция  | Назначение                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------ | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `alembic_version`        | служебная | Текущая ревизия Alembic. Гарантирует, что миграции не накатятся повторно.                                                                                                                                                                                                                                                                                                                                                              |
| `tasks`                  | 001 + 005 | Аудит-журнал асинхронных запросов: `task_id` (UUID), `query`, `status` (`PENDING` / `PROCESSING` / `COMPLETED` / `FAILED`), `result` (JSONB с финальным ответом), `error`, `celery_task_id`, тайминги (`created_at` / `started_at` / `completed_at`), `workflow_state` для отладки и `conversation_id` (FK на `conversations`, `ON DELETE SET NULL`) для связи с диалогом. На неё опирается API `POST /tasks` + `GET /tasks/{id}`.     |
| `conversations`          | 003       | **Сессия чата** (short-term memory). Хранит `id`, `user_id` (FK → `user_profiles`, добавляется в 004), `title`, `summary` + `summary_turn_index` (rolling-summary для истории длиннее окна), `is_active`, `created_at` / `updated_at`. Триггер `set_updated_at()` обновляет `updated_at` на любом `UPDATE` — на этом построена inactivity rotation (диалог старше 30 мин закрывается автоматически).                                   |
| `messages`               | 003       | **Окно диалога**: реплики пользователя и ассистента. `conversation_id` (FK CASCADE), `turn_index` (UNIQUE на пару conversation+turn), `role` (`user` / `assistant`), `content` (полный, до 50 000 символов) + `content_truncated` (~150 токенов для быстрой пересборки промпта), `metadata` JSONB (Supervisor сохраняет туда `entities` — отсюда тянется анафора между ходами).                                                        |
| `user_profiles`          | 004       | **Долговременная память** пользователя. `external_id` (стабильный UUID из cookie / `?uid=`), `display_name`, `preferences` JSONB (например, `default_team` — подставляется Supervisor'ом, когда в запросе нет команды), `context_summary` (текстовая выжимка по всем сессиям), счётчики `total_conversations` / `total_messages`. Гейт: профиль наполняется не раньше, чем пользователь отправил минимум 6 сообщений с одной командой. |
| `conversation_summaries` | 004       | Архив завершённых диалогов: при `POST /conversations/{id}/close` Celery-задача `summarize_session` пишет сюда `summary`, `topics[]`, `turn_count`. Используется для `context_summary` в `user_profiles` и для восстановления long-term-контекста при возврате пользователя. FK на `conversations` и `user_profiles` — оба `ON DELETE CASCADE`.                                                                                         |

Связи между таблицами:

```
user_profiles.id ──┐
                   ├─◄ conversations.user_id      (ON DELETE SET NULL)
                   └─◄ conversation_summaries.user_id (CASCADE)

conversations.id ──┐
                   ├─◄ messages.conversation_id      (CASCADE)
                   ├─◄ conversation_summaries.conversation_id (CASCADE)
                   └─◄ tasks.conversation_id        (SET NULL)
```

`tasks` намеренно живёт сбоку: задача — это аудит-запись, она переживает
удаление диалога (FK `SET NULL`), чтобы остаться в Grafana-метриках и отладочных
дампах.

Доменные таблицы (`report_agile_dashboard` и `report_agile_dashboard_metrics`) —
данные Jira, которые загружаются через `load-data` job из CSV S3. Их схема
описана в [architecture.md → База данных](architecture.md#база-данных).

## Инспекция своих данных через psql

После общения в Streamlit можно посмотреть, какой у вас профиль, как ходило окно
памяти и что писал workflow. Все команды — на сервере, через
`docker compose exec postgres psql ...`. Везде ниже подставляйте свой `<UID>`
(это значение `?uid=...` из URL Streamlit) и `<CID>` (id диалога).

### Найти свой `external_id`

Если на сервере общались только вы — это самый свежий профиль:

```bash
docker compose exec postgres psql -U hse_user -d hse_jira_db -c "
SELECT external_id, total_conversations, total_messages, updated_at
FROM user_profiles
ORDER BY updated_at DESC
LIMIT 5;"
```

### Профиль пользователя

```bash
docker compose exec postgres psql -U hse_user -d hse_jira_db -c "
SELECT external_id, preferences, context_summary,
       total_conversations, total_messages,
       created_at, updated_at
FROM user_profiles
WHERE external_id = '<UID>';"
```

В `preferences` появляется, например, `{\"default_team\": \"cthulhu\"}` после
того как пользователь набрал ≥6 сообщений с одной командой (в т.ч. через
анафорический carry-forward).

### Список своих диалогов

```bash
docker compose exec postgres psql -U hse_user -d hse_jira_db -c "
SELECT c.id AS conversation_id, c.is_active, c.summary_turn_index,
       c.created_at, c.updated_at,
       COUNT(m.id) AS msg_count
FROM conversations c
LEFT JOIN messages m ON m.conversation_id = c.id
WHERE c.user_id = (SELECT id FROM user_profiles WHERE external_id = '<UID>')
GROUP BY c.id
ORDER BY c.updated_at DESC;"
```

### Окно памяти — все ходы с метаданными

Самое полезное представление: показывает, что Supervisor извлёк/перенёс из
контекста на каждом ходу.

```bash
docker compose exec postgres psql -U hse_user -d hse_jira_db -c "
SELECT turn_index, role,
       LEFT(content, 80) AS preview,
       metadata->'entities'    AS entities,
       metadata->>'query_type' AS query_type,
       metadata->>'intent'     AS intent
FROM messages
WHERE conversation_id = '<CID>'
ORDER BY turn_index;"
```

Anaphora carry-forward виден напрямую: на ходу «А какой у них scope drop?» поле
`entities.team_name` должно совпадать с командой из предыдущего хода — это и
есть «окно памяти сработало».

### Полный текст одного хода (полный + truncated)

```bash
docker compose exec postgres psql -U hse_user -d hse_jira_db -c "
SELECT turn_index, role, content, content_truncated, metadata
FROM messages
WHERE conversation_id = '<CID>' AND turn_index = 4;"
```

`content_truncated` (~150 токенов от полного ответа) — это то, что попадёт
обратно в промпт Supervisor'а на следующем ходу, чтобы длинный ответ не съедал
бюджет окна.

### Audit-журнал задач (`tasks`)

Каждый запрос порождает строку в `tasks` с дампом `workflow_state` — удобно для
дебага маршрутизации.

```bash
docker compose exec postgres psql -U hse_user -d hse_jira_db -c "
SELECT task_id, status,
       LEFT(query, 60)              AS query_preview,
       workflow_state->'query_type' AS qt,
       workflow_state->'intent'     AS intent,
       workflow_state->'entities'   AS entities,
       workflow_state->'rag_sources' AS rag_sources,
       completed_at - started_at    AS duration
FROM tasks
WHERE conversation_id = '<CID>'
ORDER BY created_at;"
```

Полный workflow_state одного хода (полезно при странных ответах):

```bash
docker compose exec postgres psql -U hse_user -d hse_jira_db -c "
SELECT jsonb_pretty(workflow_state)
FROM tasks WHERE task_id = '<TASK_ID>';"
```

### Саммари закрытых сессий

Пишется при `POST /conversations/{id}/close` (кнопка «Новый диалог» в UI) или
при inactivity rotation. Если ещё в том же диалоге — таблица пустая:

```bash
docker compose exec postgres psql -U hse_user -d hse_jira_db -c "
SELECT conversation_id, summary, topics, turn_count, created_at
FROM conversation_summaries
WHERE user_id = (SELECT id FROM user_profiles WHERE external_id = '<UID>')
ORDER BY created_at DESC;"
```

### Langfuse trace

Если `LANGFUSE_ENABLED=true` — `http://195.209.218.21:3001/` → Traces, поиск по
`user_id` / `session_id`. Покажет дерево спанов (Supervisor → SQL/RAG/Hybrid →
Validator → Response) с input/output каждого LLM-вызова и таймингами.

### Шорткат — «всё про мой последний диалог»

```bash
docker compose exec postgres psql -U hse_user -d hse_jira_db <<'SQL'
WITH me AS (
  SELECT id, external_id, preferences
  FROM user_profiles
  ORDER BY updated_at DESC
  LIMIT 1
), last_conv AS (
  SELECT c.id, c.is_active, c.created_at
  FROM conversations c JOIN me ON c.user_id = me.id
  ORDER BY c.updated_at DESC LIMIT 1
)
SELECT m.turn_index, m.role,
       LEFT(m.content, 70)  AS preview,
       m.metadata->'entities' AS entities
FROM messages m
JOIN last_conv lc ON m.conversation_id = lc.id
ORDER BY m.turn_index;
SQL
```

Найдёт свежайший профиль и распечатает все ходы последнего диалога с
извлечёнными entities — самый компактный способ увидеть, как окно реально
таскало контекст между запросами.

## Связанные разделы

- [Memory Layer](memory.md) — поведение sliding window, профиля, ротации
- [API](api.md) — где `tasks` создаётся и обновляется (POST /tasks → Celery →
  workflow_state)
- [Архитектура → База данных](architecture.md#база-данных) — доменные таблицы
  Jira
