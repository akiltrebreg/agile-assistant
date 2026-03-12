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

-- Load data from CSV files using COPY command
-- Note: Empty strings are treated as NULL values, encoding is UTF-8

\echo 'Loading data into report_agile_dashboard...'
COPY report_agile_dashboard (
    launch_id, issue_key, jirasprint_id, sprint_name, start_date, end_date,
    complete_date, activation_date, sprint_state, issue_department, issue_project,
    unit, cluster, issue_type, feature_teams, storypoints_act, reporter,
    create_time, resolution_time, summary, resolution, issue_status_act, labels,
    issue_unit, added_until_sprint_start, sprint_change_date_until_start,
    sprint_change_date_until_end, num_of_current_sprint_for_task,
    issue_status_end_of_sprint, storypoints_end_of_sprint,
    storypoints_start_of_sprint, storypoints_next_sprint,
    original_estimate_end_of_sprint, original_estimate_start_of_sprint,
    original_estimate_act, assignee_name, issue_priority_for_bug,
    time_h_not_fixed, time_h_in_progress, merged_pr_count, last_comment_body,
    issue_touch_time, feature_teams_start_of_sprint, feature_teams_end_of_sprint,
    issue_potential_removal_date, removal_status, version_id, is_report,
    is_tech_debt, estimate_sprint_end, dev_approach, feature_teams_resolution,
    dev_approach_resolution, drop_team, is_pr, epic_issue_key, actual_resolution,
    last_cancelled_date
)
FROM '/docker-entrypoint-initdb.d/data/report_agile_dashboard.csv'
WITH (
    FORMAT csv,
    HEADER true,
    DELIMITER ',',
    ENCODING 'UTF8',
    NULL ''
);

\echo 'Loading data into report_agile_dashboard_metrics...'
COPY report_agile_dashboard_metrics (
    launch_id, cluster_name, unit_name, feature_teams, jirasprint_id,
    sprint_name, activation_date, complete_date, initial_commitment_sp,
    added_work_sp, final_commitment_sp, undone_sp, complete_sp,
    dev_potential_sp, scope_drop, done_total, sprint_goal,
    complete_initial_sp, complete_count_sg, count_sg, sprint_state,
    version_id, count_retro_ai, dev_approach, initial_commitment_issues,
    added_work_issues, final_commitment_issues, undone_issues,
    cancel_issues, complete_issues, scope_drop_issues,
    done_total_issues, done_issues, complete_initial_issues, cancel_rate
)
FROM '/docker-entrypoint-initdb.d/data/report_agile_dashboard_metrics.csv'
WITH (
    FORMAT csv,
    HEADER true,
    DELIMITER ',',
    ENCODING 'UTF8',
    NULL ''
);

\echo 'Database initialization complete!'
\echo 'Tables created:'
\echo '  - report_agile_dashboard'
\echo '  - report_agile_dashboard_metrics'

-- Display row counts
SELECT 'report_agile_dashboard' AS table_name, COUNT(*) AS row_count
FROM report_agile_dashboard
UNION ALL
SELECT 'report_agile_dashboard_metrics' AS table_name, COUNT(*) AS row_count
FROM report_agile_dashboard_metrics;
