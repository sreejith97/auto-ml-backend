import asyncio
import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")
    
if DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    source_type VARCHAR(50),
    r2_raw_path text,
    row_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS columns_spec (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    column_name VARCHAR(255) NOT NULL,
    role VARCHAR(50),
    semantic_type VARCHAR(50),
    missing_meaning TEXT,
    notes TEXT,
    UNIQUE(dataset_id, column_name)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    project_name VARCHAR(255),
    description TEXT,
    model_type VARCHAR(50),
    task_type VARCHAR(50),
    target_column VARCHAR(255),
    time_column VARCHAR(255),
    forecast_horizon INTEGER,
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stage_outputs (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    stage VARCHAR(50),
    r2_snapshot_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rules (
    id SERIAL PRIMARY KEY,
    column_signature VARCHAR(255) NOT NULL,
    question TEXT,
    answer TEXT,
    transformation TEXT,
    dataset_scope VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    use_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS questions_pending (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    column_name VARCHAR(255),
    issue_type VARCHAR(50),
    confidence REAL,
    evidence_jsonb JSONB,
    status VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS model_runs (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    estimator VARCHAR(100),
    metrics_jsonb JSONB,
    r2_model_path TEXT,
    feature_importance_jsonb JSONB,
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    role TEXT CHECK (role IN ('system', 'user', 'assistant')),
    content TEXT NOT NULL,
    stage TEXT NOT NULL,
    related_question_id INTEGER NULL REFERENCES questions_pending(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS feature_proposals (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    column_name VARCHAR(255) NOT NULL,
    proposed_transform VARCHAR(100),
    params_jsonb JSONB,
    rationale TEXT,
    final_transform VARCHAR(100),
    final_params_jsonb JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

async def init_db():
    print(f"Connecting to DB...")
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        print("Executing schema setup...")
        await conn.execute(SCHEMA_SQL)
        print("Schema setup complete!")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(init_db())
