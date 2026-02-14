"""Database schema descriptions for LLM prompts.

Contains human-readable descriptions of database tables, columns,
and example values used in Supervisor and SQL Agent prompts.
"""

SCHEMA_DESCRIPTION = """
Доступные таблицы в базе данных PostgreSQL:

== Таблица 1: report_agile_dashboard ==
Назначение: витрина с задачами Jira. Каждая строка — одна задача в одном спринте.
Если задача участвовала в нескольких спринтах, она будет в нескольких строках.

Колонки:
- issue_key (VARCHAR) — ключ задачи, например 'AL-38787', 'AL-39041'
- jirasprint_id (BIGINT) — ID спринта
- sprint_name (VARCHAR) — название спринта, например "#1 Q1'26", "25Q4.6 - Мандариновый рывок"
- start_date (TIMESTAMP) — дата начала спринта
- end_date (TIMESTAMP) — дата окончания спринта
- complete_date (TIMESTAMP) — фактическая дата завершения спринта
- activation_date (TIMESTAMP) — дата активации спринта
- sprint_state (VARCHAR) — состояние спринта: 'closed', 'active'
- issue_department (VARCHAR) — департамент, например 'AL'
- issue_project (VARCHAR) — проект, например 'DeepMind Logistics'
- unit (VARCHAR) — юнит/подразделение, например 'Logistics Platform', 'ML Platform'
- cluster (VARCHAR) — кластер, например 'Logistics', 'Architecture', 'Marketplace'
- issue_type (VARCHAR) — тип задачи: 'Improvement', 'Bug', 'Story', 'Task', 'Sub-task'
- feature_teams (VARCHAR) — команда, например 'cthulhu', 'lpop', 'linehaul', 'honey badger'
- storypoints_act (FLOAT) — актуальные story points
- reporter (VARCHAR) — автор задачи (имя)
- create_time (TIMESTAMP) — дата создания задачи
- resolution_time (TIMESTAMP) — дата резолюции
- summary (VARCHAR) — краткое описание задачи
- resolution (VARCHAR) — резолюция: 'Done', 'Cancelled', NULL
- issue_status_act (VARCHAR) — текущий статус: 'In Progress', 'Open', 'Closed', 'Done'
- labels (VARCHAR) — метки задачи
- assignee_name (VARCHAR) — исполнитель задачи (имя)
- issue_status_end_of_sprint (VARCHAR) — статус на конец спринта
- storypoints_end_of_sprint (FLOAT) — story points на конец спринта
- storypoints_start_of_sprint (FLOAT) — story points на начало спринта
- storypoints_next_sprint (FLOAT) — story points в следующем спринте
- time_h_not_fixed (INT) — часы без исправления
- time_h_in_progress (INT) — часы в работе
- merged_pr_count (INT) — количество влитых PR
- feature_teams_start_of_sprint (VARCHAR) — команда на начало спринта
- feature_teams_end_of_sprint (VARCHAR) — команда на конец спринта
- dev_approach (VARCHAR) — подход разработки: 'Scrum', 'Kanban'
- is_report (BOOLEAN) — отчетная задача
- is_tech_debt (BOOLEAN) — технический долг
- epic_issue_key (VARCHAR) — ключ эпика

== Таблица 2: report_agile_dashboard_metrics ==
Назначение: агрегированные метрики команд по спринтам. Каждая строка — одна команда в одном спринте.

Колонки:
- cluster_name (VARCHAR) — кластер, например 'Logistics', 'AI Lab', 'FinTech'
- unit_name (VARCHAR) — юнит, например 'Logistics Platform', 'Payments'
- feature_teams (VARCHAR) — команда, например 'cthulhu', 'lpop', 'linehaul'
- jirasprint_id (BIGINT) — ID спринта
- sprint_name (VARCHAR) — название спринта, например "#1 Q1'26", "25Q4.6 - Мандариновый рывок"
- activation_date (TIMESTAMP) — дата активации спринта
- complete_date (TIMESTAMP) — дата завершения спринта
- initial_commitment_sp (FLOAT) — начальный объем в story points
- added_work_sp (FLOAT) — добавленная работа в story points
- final_commitment_sp (FLOAT) — итоговый объем в story points
- undone_sp (FLOAT) — незавершённые story points
- complete_sp (FLOAT) — завершённые story points
- dev_potential_sp (FLOAT) — потенциал разработки в story points
- scope_drop (FLOAT) — процент сброса скоупа (%)
- done_total (FLOAT) — процент выполнения (%)
- sprint_goal (FLOAT) — процент достижения цели спринта (%)
- complete_initial_sp (FLOAT) — завершённые из начального объема
- complete_count_sg (INT) — количество выполненных целей спринта
- count_sg (INT) — общее количество целей спринта
- sprint_state (VARCHAR) — состояние спринта: 'closed', 'active'
- dev_approach (VARCHAR) — подход разработки: 'Scrum', 'Kanban'
- initial_commitment_issues (FLOAT) — начальное количество задач
- added_work_issues (FLOAT) — добавленные задачи
- final_commitment_issues (FLOAT) — итоговое количество задач
- undone_issues (FLOAT) — незавершённые задачи
- cancel_issues (FLOAT) — отменённые задачи
- complete_issues (FLOAT) — завершённые задачи
- scope_drop_issues (FLOAT) — процент сброса скоупа по задачам (%)
- done_total_issues (FLOAT) — процент выполнения по задачам (%)
- done_issues (FLOAT) — количество выполненных задач
- complete_initial_issues (FLOAT) — завершённые из начальных задач
- cancel_rate (FLOAT) — процент отмены (%)
""".strip()

SUPERVISOR_FEW_SHOT_EXAMPLES = """
Примеры классификации:

Запрос: "Выведи данные по задаче AL-38787"
{"intent": "task", "entities": {"issue_key": "AL-38787"}}

Запрос: "AL-39041"
{"intent": "task", "entities": {"issue_key": "AL-39041"}}

Запрос: "Информация о задаче AL-38799"
{"intent": "task", "entities": {"issue_key": "AL-38799"}}

Запрос: "Все задачи команды cthulhu"
{"intent": "tasks_filter", "entities": {"team_name": "cthulhu"}}

Запрос: "Задачи команды lpop в спринте #1 Q1'26"
{"intent": "tasks_filter", "entities": {"team_name": "lpop", "sprint_name": "#1 Q1'26"}}

Запрос: "Покажи баги команды honey badger"
{"intent": "tasks_filter", "entities": {"team_name": "honey badger", "issue_type": "Bug"}}

Запрос: "Задачи со статусом In Progress в кластере Logistics"
{"intent": "tasks_filter", "entities": {"status": "In Progress", "cluster": "Logistics"}}

Запрос: "Done Total из спринта Мандариновый рывок"
{"intent": "metric", "entities": {"sprint_name": "Мандариновый рывок", "metric_name": "done_total"}}

Запрос: "Какой scope drop у команды cthulhu"
{"intent": "metric", "entities": {"team_name": "cthulhu", "metric_name": "scope_drop"}}

Запрос: "Метрики команды lpop за спринт #2 Q1'26"
{"intent": "metric", "entities": {"team_name": "lpop", "sprint_name": "#2 Q1'26"}}

Запрос: "Velocity команды linehaul"
{"intent": "metric", "entities": {"team_name": "linehaul", "metric_name": "complete_sp"}}

Запрос: "Что такое спринт в Agile?"
{"intent": "general", "entities": {}}

Запрос: "Привет"
{"intent": "general", "entities": {}}

Запрос: "Как рассчитываются story points?"
{"intent": "general", "entities": {}}
""".strip()

ALLOWED_TABLES = frozenset(
    {
        "report_agile_dashboard",
        "report_agile_dashboard_metrics",
    }
)

SQL_MAX_ROWS = 100
