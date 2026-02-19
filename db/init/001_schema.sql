-- DB is created with the commando "docker compose up -d"


-- Creates the extension for UUID generation
CREATE EXTENSION IF NOT EXISTS pgcrypto;


CREATE TABLE IF NOT EXISTS training_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    data BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (name)
);

-- Creates the model table (if not already created)
CREATE TABLE IF NOT EXISTS model (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    accuracy DOUBLE PRECISION,
    precision DOUBLE PRECISION,
    recall DOUBLE PRECISION,
    model_data BYTEA NOT NULL,
    training_data_id UUID,
    status TEXT NOT NULL DEFAULT 'staging',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (name, version),
    FOREIGN KEY (training_data_id) REFERENCES training_data(id) ON DELETE SET NULL
    );

-- Future tables here


CREATE TABLE IF NOT EXISTS Model_parameters(
    model_id UUID NOT NULL,
    name TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (model_id, name, value),
    FOREIGN KEY (model_id) REFERENCES model(id) ON DELETE CASCADE
    
);
