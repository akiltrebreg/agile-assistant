-- Initialize Agile Dashboard database
-- This script creates the schema and loads data from CSV files

-- Create report_agile_dashboard table
CREATE TABLE IF NOT EXISTS report_agile_dashboard (
    launch_id INT NOT NULL,
    issue_key VARCHAR(32),
    jirasprint_id BIGINT,
    sprint_name VARCHAR(256),
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    complete_date TIMESTAMP,
    activation_date TIMESTAMP,
    sprint_state VARCHAR(128),
    issue_department VARCHAR(32),
    issue_project VARCHAR(256),
    unit VARCHAR(128),
    cluster VARCHAR(128),
    issue_type VARCHAR(256),
    feature_teams VARCHAR(256),
    storypoints_act FLOAT,
    reporter VARCHAR(256),
    create_time TIMESTAMP,
    resolution_time TIMESTAMP,
    summary VARCHAR(1024),
    resolution VARCHAR(256),
    issue_status_act VARCHAR(256),
    labels VARCHAR(1024),
    issue_unit VARCHAR(1024),
    added_until_sprint_start BOOLEAN,
    sprint_change_date_until_start TIMESTAMP,
    sprint_change_date_until_end TIMESTAMP,
    num_of_current_sprint_for_task INT,
    issue_status_end_of_sprint VARCHAR(256),
    storypoints_end_of_sprint FLOAT,
    storypoints_start_of_sprint FLOAT,
    storypoints_next_sprint FLOAT,
    original_estimate_end_of_sprint INT,
    original_estimate_start_of_sprint INT,
    original_estimate_act INT,
    assignee_name VARCHAR(256),
    issue_priority_for_bug VARCHAR(256),
    time_h_not_fixed INT,
    time_h_in_progress INT,
    merged_pr_count INT,
    last_comment_body VARCHAR(4000),
    issue_touch_time INT,
    feature_teams_start_of_sprint VARCHAR(256),
    feature_teams_end_of_sprint VARCHAR(256),
    issue_potential_removal_date TIMESTAMP,
    removal_status VARCHAR(256),
    version_id INT NOT NULL DEFAULT 0,
    is_report BOOLEAN,
    is_tech_debt BOOLEAN,
    estimate_sprint_end FLOAT,
    dev_approach VARCHAR(32),
    feature_teams_resolution VARCHAR(256),
    dev_approach_resolution VARCHAR(32),
    drop_team VARCHAR(256),
    is_pr BOOLEAN,
    epic_issue_key VARCHAR(32),
    actual_resolution VARCHAR(256),
    last_cancelled_date TIMESTAMP
);

-- Create report_agile_dashboard_metrics table
CREATE TABLE IF NOT EXISTS report_agile_dashboard_metrics (
    launch_id INT NOT NULL,
    cluster_name VARCHAR(128),
    unit_name VARCHAR(128),
    feature_teams VARCHAR(256),
    jirasprint_id BIGINT,
    sprint_name VARCHAR(256),
    activation_date TIMESTAMP,
    complete_date TIMESTAMP,
    initial_commitment_sp FLOAT,
    added_work_sp FLOAT,
    final_commitment_sp FLOAT,
    undone_sp FLOAT,
    complete_sp FLOAT,
    dev_potential_sp FLOAT,
    scope_drop FLOAT,
    done_total FLOAT,
    sprint_goal FLOAT,
    complete_initial_sp FLOAT,
    complete_count_sg INT,
    count_sg INT,
    sprint_state VARCHAR(32),
    version_id INT NOT NULL,
    count_retro_ai INT,
    dev_approach VARCHAR(32),
    initial_commitment_issues FLOAT,
    added_work_issues FLOAT,
    final_commitment_issues FLOAT,
    undone_issues FLOAT,
    cancel_issues FLOAT,
    complete_issues FLOAT,
    scope_drop_issues FLOAT,
    done_total_issues FLOAT,
    done_issues FLOAT,
    complete_initial_issues FLOAT,
    cancel_rate FLOAT
);

-- Create indexes for better query performance on report_agile_dashboard
CREATE INDEX IF NOT EXISTS idx_dashboard_issue_key ON report_agile_dashboard(issue_key);
CREATE INDEX IF NOT EXISTS idx_dashboard_sprint_id ON report_agile_dashboard(jirasprint_id);
CREATE INDEX IF NOT EXISTS idx_dashboard_sprint_state ON report_agile_dashboard(sprint_state);
CREATE INDEX IF NOT EXISTS idx_dashboard_assignee ON report_agile_dashboard(assignee_name);
CREATE INDEX IF NOT EXISTS idx_dashboard_launch_id ON report_agile_dashboard(launch_id);
CREATE INDEX IF NOT EXISTS idx_dashboard_feature_teams ON report_agile_dashboard(feature_teams);

-- Create indexes for better query performance on report_agile_dashboard_metrics
CREATE INDEX IF NOT EXISTS idx_metrics_sprint_id ON report_agile_dashboard_metrics(jirasprint_id);
CREATE INDEX IF NOT EXISTS idx_metrics_feature_teams ON report_agile_dashboard_metrics(feature_teams);
CREATE INDEX IF NOT EXISTS idx_metrics_launch_id ON report_agile_dashboard_metrics(launch_id);
CREATE INDEX IF NOT EXISTS idx_metrics_sprint_state ON report_agile_dashboard_metrics(sprint_state);

-- CSV data is loaded from S3 by hse_prom_prog/database/load_csv.py
-- (run via: docker compose run --rm load-data)

-- ═══════════════════════════════════════════════════════════════
-- COMMENT ON — описания таблиц и колонок для text2sql
-- Оптимизированы по результатам eval (v5: 20/46)
-- ═══════════════════════════════════════════════════════════════

-- ── Таблица задач ──────────────────────────────────────────────

COMMENT ON TABLE report_agile_dashboard IS
'Задачи Jira: одна строка = одна задача в одном спринте. '
'Используйте для: поиска задач, подсчёта задач (COUNT), суммы story points (SUM storypoints_act), '
'фильтрации по команде/статусу/типу/кластеру/исполнителю. '
'НЕ используйте для метрик velocity, done_total, scope_drop, cancel_rate — они в report_agile_dashboard_metrics.';

COMMENT ON COLUMN report_agile_dashboard.issue_key IS 'уникальный ключ задачи (пример: AL-38787). Для поиска задачи: WHERE issue_key = ''AL-38787''';
COMMENT ON COLUMN report_agile_dashboard.feature_teams IS 'название команды (пример: cthulhu, lpop, linehaul, shopping cart). Фильтр: WHERE feature_teams ILIKE ''%имя%''';
COMMENT ON COLUMN report_agile_dashboard.sprint_name IS 'название спринта (пример: #1 Q1''26, #2 Q1''26, #3 Q1''26). Фильтр: WHERE sprint_name ILIKE ''%#1 Q1''''26%''';
COMMENT ON COLUMN report_agile_dashboard.cluster IS 'кластер (пример: Logistics, Marketplace, AI Lab). Фильтр: WHERE cluster ILIKE ''%Logistics%''';
COMMENT ON COLUMN report_agile_dashboard.issue_type IS 'тип задачи: Bug, Story, Task, Improvement, Epic, Sprint Goal. Для поиска багов: WHERE issue_type = ''Bug''. Для Story: WHERE issue_type = ''Story''';
COMMENT ON COLUMN report_agile_dashboard.issue_status_act IS 'текущий статус: Done, In Progress, Open, Groomed, Cancelled. Для закрытых: WHERE issue_status_act = ''Done''';
COMMENT ON COLUMN report_agile_dashboard.assignee_name IS 'логин исполнителя. Фильтр: WHERE assignee_name ILIKE ''%логин%''';
COMMENT ON COLUMN report_agile_dashboard.storypoints_act IS 'текущие story points задачи. Для суммы SP команды: SUM(storypoints_act). Для среднего размера Story: AVG(storypoints_act) WHERE issue_type = ''Story''';
COMMENT ON COLUMN report_agile_dashboard.issue_priority_for_bug IS 'приоритет бага: P1, P2, P3, P4 (только для Bug)';
COMMENT ON COLUMN report_agile_dashboard.summary IS 'текстовое описание задачи';
COMMENT ON COLUMN report_agile_dashboard.reporter IS 'автор задачи';
COMMENT ON COLUMN report_agile_dashboard.create_time IS 'дата создания задачи';
COMMENT ON COLUMN report_agile_dashboard.resolution_time IS 'дата закрытия задачи';
COMMENT ON COLUMN report_agile_dashboard.sprint_state IS 'статус спринта: active или closed';
COMMENT ON COLUMN report_agile_dashboard.issue_project IS 'проект Jira';
COMMENT ON COLUMN report_agile_dashboard.unit IS 'подразделение';
COMMENT ON COLUMN report_agile_dashboard.storypoints_end_of_sprint IS 'SP задачи на конец спринта (историческое значение)';
COMMENT ON COLUMN report_agile_dashboard.storypoints_start_of_sprint IS 'SP задачи на начало спринта (историческое значение)';
COMMENT ON COLUMN report_agile_dashboard.time_h_in_progress IS 'часы в статусе In Progress';
COMMENT ON COLUMN report_agile_dashboard.merged_pr_count IS 'количество merged pull requests';
COMMENT ON COLUMN report_agile_dashboard.issue_status_end_of_sprint IS 'статус задачи на конец спринта (историческое значение)';
COMMENT ON COLUMN report_agile_dashboard.start_date IS 'дата начала спринта';
COMMENT ON COLUMN report_agile_dashboard.end_date IS 'дата окончания спринта';
COMMENT ON COLUMN report_agile_dashboard.complete_date IS 'дата завершения спринта';
COMMENT ON COLUMN report_agile_dashboard.added_until_sprint_start IS 'задача добавлена до начала спринта (true/false)';
COMMENT ON COLUMN report_agile_dashboard.num_of_current_sprint_for_task IS 'номер текущего спринта для задачи (сколько спринтов задача в работе)';

-- ── Таблица метрик ─────────────────────────────────────────────

COMMENT ON TABLE report_agile_dashboard_metrics IS
'Агрегированные метрики команд по спринтам: одна строка = одна команда в одном спринте. '
'Используйте для: velocity (complete_sp), done_total, scope_drop, cancel_rate, sprint_goal, added_work_sp. '
'Для сравнения команд используйте AVG() GROUP BY feature_teams. '
'НЕ используйте для подсчёта количества задач или суммы SP задач — они в report_agile_dashboard.';

COMMENT ON COLUMN report_agile_dashboard_metrics.feature_teams IS 'название команды (то же что в report_agile_dashboard). ВСЕГДА включайте в SELECT при запросе метрик';
COMMENT ON COLUMN report_agile_dashboard_metrics.sprint_name IS 'название спринта. ВСЕГДА включайте в SELECT при запросе метрик конкретной команды';
COMMENT ON COLUMN report_agile_dashboard_metrics.cluster_name IS 'кластер (внимание: здесь cluster_name, а не cluster как в dashboard)';
COMMENT ON COLUMN report_agile_dashboard_metrics.unit_name IS 'подразделение (внимание: здесь unit_name, а не unit как в dashboard)';
COMMENT ON COLUMN report_agile_dashboard_metrics.sprint_state IS 'статус спринта: active или closed';

-- Ключевые метрики
COMMENT ON COLUMN report_agile_dashboard_metrics.complete_sp IS 'velocity = выполненные SP за спринт. Для средней velocity: AVG(complete_sp). Для суммарной: SUM(complete_sp)';
COMMENT ON COLUMN report_agile_dashboard_metrics.done_total IS 'done total %% — процент выполнения спринта. Для среднего: AVG(done_total) WHERE feature_teams ILIKE ...';
COMMENT ON COLUMN report_agile_dashboard_metrics.scope_drop IS 'scope drop %% — процент снятых задач. Для среднего: AVG(scope_drop) WHERE feature_teams ILIKE ...';
COMMENT ON COLUMN report_agile_dashboard_metrics.cancel_rate IS 'cancel rate %% — процент отменённых задач';
COMMENT ON COLUMN report_agile_dashboard_metrics.sprint_goal IS 'sprint goal %% — процент достижения цели спринта';

-- SP метрики спринта (НЕ для подсчёта SP задач)
COMMENT ON COLUMN report_agile_dashboard_metrics.initial_commitment_sp IS 'начальный объём SP спринта (коммитмент команды, НЕ сумма SP задач)';
COMMENT ON COLUMN report_agile_dashboard_metrics.added_work_sp IS 'SP добавленные в спринт после старта. Для вопросов "сколько SP добавлено" используйте SUM(added_work_sp)';
COMMENT ON COLUMN report_agile_dashboard_metrics.final_commitment_sp IS 'итоговый объём SP спринта (коммитмент, НЕ сумма SP задач)';
COMMENT ON COLUMN report_agile_dashboard_metrics.undone_sp IS 'невыполненные SP за спринт';
COMMENT ON COLUMN report_agile_dashboard_metrics.dev_potential_sp IS 'потенциал разработки в SP';
COMMENT ON COLUMN report_agile_dashboard_metrics.complete_initial_sp IS 'выполненные SP из начального коммитмента';

-- Метрики по задачам (issues)
COMMENT ON COLUMN report_agile_dashboard_metrics.complete_issues IS 'количество завершённых задач в спринте';
COMMENT ON COLUMN report_agile_dashboard_metrics.initial_commitment_issues IS 'начальное количество задач в спринте';
COMMENT ON COLUMN report_agile_dashboard_metrics.added_work_issues IS 'задачи добавленные после старта спринта';
COMMENT ON COLUMN report_agile_dashboard_metrics.final_commitment_issues IS 'итоговое количество задач в спринте';
COMMENT ON COLUMN report_agile_dashboard_metrics.undone_issues IS 'невыполненные задачи';
COMMENT ON COLUMN report_agile_dashboard_metrics.cancel_issues IS 'отменённые задачи';
COMMENT ON COLUMN report_agile_dashboard_metrics.scope_drop_issues IS 'scope drop по задачам %%';
COMMENT ON COLUMN report_agile_dashboard_metrics.done_total_issues IS 'done total по задачам %%';
COMMENT ON COLUMN report_agile_dashboard_metrics.done_issues IS 'завершённые задачи (Done)';
COMMENT ON COLUMN report_agile_dashboard_metrics.complete_initial_issues IS 'выполненные задачи из начального коммитмента';
COMMENT ON COLUMN report_agile_dashboard_metrics.count_sg IS 'количество sprint goals';
COMMENT ON COLUMN report_agile_dashboard_metrics.complete_count_sg IS 'выполненные sprint goals';
COMMENT ON COLUMN report_agile_dashboard_metrics.count_retro_ai IS 'количество ретроспектив AI';

\echo 'Schema initialization complete!'
\echo 'Tables created:'
\echo '  - report_agile_dashboard'
\echo '  - report_agile_dashboard_metrics'
