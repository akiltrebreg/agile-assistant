-- Initialize Jira Issues database
-- This script creates the schema and populates test data

-- Create jira_issues table
CREATE TABLE IF NOT EXISTS jira_issues (
    issue_key VARCHAR(32) PRIMARY KEY,
    jirasprint_id INT,
    sprint_name VARCHAR(256),
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    activation_date TIMESTAMP,
    complete_date TIMESTAMP,
    sprint_state VARCHAR(128) CHECK (sprint_state IN ('active', 'closed', 'cancelled')),
    issue_project VARCHAR(256),
    cluster VARCHAR(128),
    unit VARCHAR(128),
    issue_type VARCHAR(256),
    feature_teams VARCHAR(256),
    reporter VARCHAR(256),
    create_time TIMESTAMP,
    resolution_time TIMESTAMP,
    resolution VARCHAR(256),
    labels TEXT[],
    issue_status_end_of_sprint VARCHAR(256),
    storypoints_end_of_sprint FLOAT,
    storypoints_start_of_sprint FLOAT,
    storypoints_next_sprint FLOAT,
    assignee_name VARCHAR(256),
    time_h_not_fixed INT,
    time_h_in_progress INT,
    feature_teams_start_of_sprint VARCHAR(256),
    feature_teams_end_of_sprint VARCHAR(256),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_issue_key ON jira_issues(issue_key);
CREATE INDEX IF NOT EXISTS idx_sprint_state ON jira_issues(sprint_state);
CREATE INDEX IF NOT EXISTS idx_assignee ON jira_issues(assignee_name);
CREATE INDEX IF NOT EXISTS idx_sprint_id ON jira_issues(jirasprint_id);

-- Insert test data
INSERT INTO jira_issues (
    issue_key, jirasprint_id, sprint_name, start_date, end_date,
    sprint_state, issue_project, issue_type, feature_teams,
    reporter, assignee_name, issue_status_end_of_sprint,
    storypoints_start_of_sprint, storypoints_end_of_sprint,
    time_h_in_progress, time_h_not_fixed, create_time, labels,
    cluster, unit, resolution
) VALUES
    -- ABC-123: Active sprint, in progress
    ('ABC-123', 101, 'Sprint 25', '2026-01-06 00:00:00', '2026-01-17 23:59:59',
     'active', 'ABC Project', 'Story', 'Team Alpha',
     'Ivan Petrov', 'Olga Komkova', 'In Progress',
     5.0, 5.0, 12, 3, '2026-01-07 10:00:00', ARRAY['backend', 'api'],
     'Product Development', 'Backend Team', NULL),

    -- AXYZ-789: Closed sprint, completed bug fix
    ('AXYZ-789', 99, 'Sprint 24', '2025-12-23 00:00:00', '2026-01-03 23:59:59',
     'closed', 'AXYZ Project', 'Bug', 'Team Beta',
     'Maria Ivanova', 'Dmitry Kozlov', 'Done',
     3.0, 0.0, 0, 0, '2025-12-24 14:30:00', ARRAY['bugfix', 'critical'],
     'Quality Assurance', 'QA Team', 'Fixed'),

    -- PROJ-456: Active sprint, not started yet
    ('PROJ-456', 102, 'Sprint 26', '2026-01-20 00:00:00', '2026-01-31 23:59:59',
     'active', 'PROJ Core', 'Task', 'Team Gamma',
     'Alexey Volkov', 'Elena Nikolaeva', 'To Do',
     8.0, 8.0, 0, 8, '2026-01-10 09:15:00', ARRAY['infrastructure', 'deployment'],
     'Platform', 'DevOps', NULL),

    -- DEV-999: Cancelled sprint, epic not completed
    ('DEV-999', 98, 'Sprint 23', '2025-12-09 00:00:00', '2025-12-20 23:59:59',
     'cancelled', 'DevOps', 'Epic', 'Platform Team',
     'Sergey Morozov', NULL, 'Cancelled',
     13.0, 13.0, 2, 11, '2025-12-10 11:45:00', ARRAY['epic', 'cancelled'],
     'Infrastructure', 'Platform Team', 'Cancelled'),

    -- TECH-555: Active sprint, in code review
    ('TECH-555', 101, 'Sprint 25', '2026-01-06 00:00:00', '2026-01-17 23:59:59',
     'active', 'Tech Debt', 'Improvement', 'Team Alpha',
     'Olga Petrova', 'Ivan Petrov', 'Code Review',
     2.0, 2.0, 6, 1, '2026-01-08 16:20:00', ARRAY['tech-debt', 'refactoring'],
     'Product Development', 'Backend Team', NULL)
ON CONFLICT (issue_key) DO NOTHING;

-- Create a function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_jira_issues_updated_at
    BEFORE UPDATE ON jira_issues
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (if needed)
-- GRANT ALL PRIVILEGES ON TABLE jira_issues TO hse_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO hse_user;
