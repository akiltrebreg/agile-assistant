[← README](../README.md) · Раздел: Memory Layer

# Memory Layer

Двухуровневая память: короткоживущий контекст диалога (sliding window + rolling
summary) + долгоживущий профиль пользователя (default_team, frequent_metrics).
Вся память живёт в PostgreSQL, дополнительных контейнеров не требует.

Конфигурируется тремя env-переменными:

| Переменная                | Значение по умолчанию | Что задаёт                                                 |
| ------------------------- | --------------------- | ---------------------------------------------------------- |
| `HISTORY_TOKEN_BUDGET`    | 800                   | Бюджет токенов на блок `<conversation_history>` в промптах |
| `SESSION_TIMEOUT_MINUTES` | 30                    | Через сколько минут idle диалог закрывается и ротируется   |
| `MAX_CONVERSATION_TURNS`  | 50                    | Верхняя граница хранимых ходов на один диалог              |

## Контекст диалога

Short-term память: последние N ходов + rolling summary более старых.

- **Sliding window по токенам**. `ContextBuilder` загружает все сообщения
  диалога, идёт от свежих к старым и копит ходы, пока суммарная стоимость не
  превышает `HISTORY_TOKEN_BUDGET`. Всегда оставляет хотя бы 1 ход.
- **Двойной контент**. Сообщения хранятся в двух полях: `content` (полный ответ,
  для пользователя) и `content_truncated` (~150 токенов, для replay в промпт).
  Длинный ответ ассистента не съест бюджет на следующем ходе.
- **Rolling summary**. Что не влезло в окно — покрывается `summary` диалога.
  Пересчитывается асинхронно через Celery (`summarize_session`), никогда не
  блокирует хот-путь.
- **Inactivity rotation**. Если после предыдущего хода прошло больше
  `SESSION_TIMEOUT_MINUTES`, workflow автоматически закрывает старый диалог,
  запускает суммаризацию (для long-term rolling profile) и создаёт новый.
- **Явная ротация**. Любой UI-выход из текущего чата — кнопка «Новый диалог» или
  клик по другому диалогу в истории — на бэкенде вызывает
  `POST /conversations/{id}/close`: текущая сессия закрывается, шедулится
  суммаризация и обновление long-term профиля. Отдельной кнопки «Завершить
  диалог» нет — её ментальная модель совпадает с «Новый диалог».
- **Анафорический carry-forward**. При анафорических маркерах («эта команда»,
  «тот спринт», «у них», «в том же», «покажи ещё») entities (`team_name`,
  `sprint_name`, `cluster`, `assignee`) из предыдущего хода восстанавливаются,
  если LLM не извлёк их из текущего запроса. Реализовано как layer 6 в Entity
  Sanitizer (см.
  [architecture.md → Supervisor Agent](architecture.md#1-supervisor-agent-классификатор-запросов)
  для разбора всех 7 слоёв). Маркеры детектируются через лемматизацию
  (`pymorphy3`): запрос токенизируется, каждое слово приводится к нормальной
  форме и пересекается с компактным лемма-сетом (`этот`, `тот`, `такой`, `они`,
  `он`, `она`, `свой`, …) — все падежи/рода/числа покрываются без ручного
  перечисления.
- **Fallback enum extractor** — слой 7 в Entity Sanitizer: если LLM не заполнил
  `issue_type` / `status` / `metric_name`, тот же `_SYNONYM_MAPS` (что
  нормализует выход LLM в layer 1) переиспользуется как извлекатель из сырого
  запроса. Один словарь — два направления, нет дублирования источника правды.

**Multi-turn eval**
([eval/run_multiturn_eval.py](../eval/run_multiturn_eval.py), 14 кейсов, 5
подкатегорий): carry-forward accuracy, false carry-forward rate, routing
accuracy. Deploy gate: ≥ 85% carry accuracy, 0% false carry. Dataset расширяет
[eval/supervisor_golden_dataset.json](../eval/supervisor_golden_dataset.json)
кейсами с массивом `turns`.

## Профиль пользователя

Long-term память, накапливается по сообщениям пользователя, дропается в промпты
Supervisor и Response Agent.

- **Идентификация**. В Streamlit стабильный UUID пинится в
  `st.query_params["uid"]` (модуль
  [streamlit_app/auth.py](../streamlit_app/auth.py)) — один и тот же таб
  браузера попадает в одну и ту же строку `user_profiles`. Под SSO меняется
  только этот модуль.
- **Что сохраняется** (детерминированно, без LLM — `ProfileExtractor`):
  - `default_team` — команда, которая упоминалась в ≥ 60% сообщений
    пользователя; Supervisor подставляет её как team_name, когда запрос без
    явной команды (защищено rule «query wins over profile»).
  - `frequent_metrics` — top-3 метрики, о которых пользователь обычно
    спрашивает.
  - `recent_sprints`, `dominant_query_types` — для будущих подсказок.
- **Гейт**. Профиль наполняется не раньше, чем пользователь отправил минимум 6
  сообщений с одной командой (включая carry-forward через анафору) — иначе
  default_team и frequent_metrics остаются пустыми.
- **Context summary** — rolling roll-up 10 последних session summary'ев; меньше
  3 — склеиваем, иначе LLM генерирует мета-саммари (max 200 токенов). Пишется в
  `user_profiles.context_summary` и отображается в промпте как
  `<user_profile>…</user_profile>`.
- **Асинхронное обновление**. После каждого завершённого хода Celery таска
  `update_profile_async` пересчитывает preferences + context_summary, чтобы не
  удлинять путь ответа пользователю.

## Persistence

Все данные памяти живут в PostgreSQL, в таблицах `conversations`, `messages`,
`user_profiles`, `conversation_summaries` (миграции Alembic 003–005). Полная
схема + ER-диаграмма + psql-инспекция своих данных — в
[database.md](database.md).

## Конфигурация

Те же три переменные, что и в начале раздела, но с пояснениями:

| Переменная                | По умолчанию | Поведение                                                                                 |
| ------------------------- | ------------ | ----------------------------------------------------------------------------------------- |
| `HISTORY_TOKEN_BUDGET`    | 800          | Сколько токенов из истории попадает в промпт. Меньше = быстрее, больше = глубже контекст. |
| `SESSION_TIMEOUT_MINUTES` | 30           | После такого простоя диалог закрывается, в фоне запускается summarize_session.            |
| `MAX_CONVERSATION_TURNS`  | 50           | Жёсткий потолок длины диалога — выше него старые ходы перестают читаться (и обобщаются).  |

## Связанные разделы

- [База данных](database.md) — схема `conversations` / `messages` /
  `user_profiles` / `conversation_summaries` + psql-инспекция
- [Архитектура → Supervisor Agent](architecture.md#1-supervisor-agent-классификатор-запросов)
  — все 7 слоёв Entity Sanitizer (включая carry-forward)
- [API → close](api.md) — `POST /conversations/{id}/close` запускает
  summarize_session
- [Конфигурация](configuration.md) — общая таблица env-переменных
