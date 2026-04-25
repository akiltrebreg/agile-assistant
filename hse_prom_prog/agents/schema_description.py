"""Database schema descriptions for LLM prompts.

Contains human-readable descriptions of database tables, columns,
and example values used in Supervisor and SQL Agent prompts.
"""

SCHEMA_DESCRIPTION = """
В БД есть две таблицы:
- report_agile_dashboard — задачи Jira (1 строка = 1 задача в 1 спринте).
  Запросы о конкретной задаче, фильтрах по командам/статусам/типам → эта таблица.
- report_agile_dashboard_metrics — агрегированные метрики команд по спринтам.
  Запросы о done_total / scope_drop / velocity / sprint_goal / cancel_rate → эта.

## Enum-значения (нормализуй к этому формату)

issue_type ∈ {Bug, Story, Improvement, Epic, Task, Sub-task}
  Синонимы: 'баг/багов/баги' → Bug, 'сторис/story' → Story,
            'улучшение/improvement' → Improvement, 'эпик/epic' → Epic,
            'задача/task' → Task, 'саб-таска/подзадача' → Sub-task

status ∈ {Open, In Progress, Done, Closed, Cancelled}
  Синонимы: 'открыта/открытые' → Open, 'в работе/в прогрессе' → In Progress,
            'сделана/готово/done' → Done, 'закрыта/закрытые/closed' → Closed,
            'отменена/отменённые/cancelled' → Cancelled

metric_name ∈ {done_total, scope_drop, velocity, sprint_goal, cancel_rate,
               initial_commitment_sp, added_work_sp, final_commitment_sp, complete_sp}
  Синонимы: 'скорость/velocity' → velocity,
            'процент выполнения/done total' → done_total,
            'сброс скоупа/scope drop' → scope_drop,
            'цель спринта/sprint goal' → sprint_goal,
            'доля отменённого/cancel rate' → cancel_rate,
            'завершённые SP' → complete_sp,
            'начальный объём/initial commitment' → initial_commitment_sp,
            'добавленная работа/added work' → added_work_sp,
            'итоговый объём/final commitment' → final_commitment_sp

## Открытые поля (извлекай из запроса как есть, не нормализуй)

team_name — после слов 'команда/команды/у команды/team'
  или перед словом 'команда' ('marketplace команда').
sprint_name — после слов 'спринт/спринте/в спринте/sprint'
  (может содержать кавычки, #, номер квартала, любую строку).
cluster — после слов 'кластер/кластере/в кластере/cluster'.
assignee — после слов 'исполнитель/от/assignee'.
issue_key — паттерн [A-Z]+-число (например, AL-38787, DATA-1234).
""".strip()

SUPERVISOR_FEW_SHOT_EXAMPLES = """
## Примеры (по одному на каждую комбинацию query_type × intent)

Q: "AL-38787"
A: {"query_type": "sql", "intent": "task", "entities": {"issue_key": "AL-38787"}}

Q: "Задачи команды lpop в спринте #1 Q1'26"
A: {"query_type": "sql", "intent": "tasks_filter", \
"entities": {"team_name": "lpop", "sprint_name": "#1 Q1'26"}}

Q: "Какой scope drop у команды cthulhu?"
A: {"query_type": "sql", "intent": "metric", \
"entities": {"team_name": "cthulhu", "metric_name": "scope_drop"}}

Q: "Scope drop cthulhu"
A: {"query_type": "sql", "intent": "metric", \
"entities": {"team_name": "cthulhu", "metric_name": "scope_drop"}}

Q: "Метрики за спринт #1 Q1'26"
A: {"query_type": "sql", "intent": "metric", \
"entities": {"sprint_name": "#1 Q1'26"}}

Q: "Как рассчитывается Done Total?"
A: {"query_type": "rag", "intent": "general", "entities": {}}

Q: "Done total команды lpop и что можно улучшить"
A: {"query_type": "hybrid", "intent": "metric", \
"entities": {"team_name": "lpop", "metric_name": "done_total"}}

Q: "Привет"
A: {"query_type": "simple", "intent": "general", "entities": {}}

Q: "Расскажи анекдот"
A: {"query_type": "off_topic", "intent": "general", "entities": {}}

Q: "Какая погода в Москве?"
A: {"query_type": "off_topic", "intent": "general", "entities": {}}
""".strip()

ALLOWED_TABLES = frozenset(
    {
        "report_agile_dashboard",
        "report_agile_dashboard_metrics",
    }
)

SQL_MAX_ROWS = 100
