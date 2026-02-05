-- DB is created with the commando "docker compose up -d"


-- Creates the extension for UUID generation
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Creates the model table (if not already created)
CREATE TABLE IF NOT EXISTS model (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    accuracy DOUBLE PRECISION,
    status TEXT NOT NULL DEFAULT 'staging',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (name, version)
    );

-- Future tables here