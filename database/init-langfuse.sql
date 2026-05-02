-- Create the Langfuse database alongside the application database.
-- Runs on first PostgreSQL container init only; \gexec makes the CREATE
-- conditional, so re-running an existing volume is a no-op.
SELECT 'CREATE DATABASE langfuse_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse_db')
\gexec
