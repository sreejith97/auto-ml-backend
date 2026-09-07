import asyncpg
import json
from typing import Dict, Any, Optional

async def insert_model_run(
    conn: asyncpg.Connection,
    run_id: int,
    estimator: str,
    metrics: dict,
    r2_model_path: str,
    feature_importance: dict,
    version: int
) -> int:
    # First, deactivate all other versions for this run
    await conn.execute("UPDATE model_runs SET is_active = false WHERE run_id = $1", run_id)
    
    query = """
        INSERT INTO model_runs (run_id, estimator, metrics_jsonb, r2_model_path, feature_importance_jsonb, version, is_active)
        VALUES ($1, $2, $3, $4, $5, $6, true)
        RETURNING id
    """
    return await conn.fetchval(
        query, 
        run_id, 
        estimator, 
        json.dumps(metrics), 
        r2_model_path, 
        json.dumps(feature_importance),
        version
    )

async def get_model_run(conn: asyncpg.Connection, run_id: int, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
    if version:
        query = """
            SELECT id, run_id, estimator, metrics_jsonb, r2_model_path, feature_importance_jsonb, version, is_active, created_at
            FROM model_runs
            WHERE run_id = $1 AND version = $2
            LIMIT 1
        """
        record = await conn.fetchrow(query, run_id, version)
    else:
        # Get active version, fallback to newest if none active
        query = """
            SELECT id, run_id, estimator, metrics_jsonb, r2_model_path, feature_importance_jsonb, version, is_active, created_at
            FROM model_runs
            WHERE run_id = $1
            ORDER BY is_active DESC, version DESC
            LIMIT 1
        """
        record = await conn.fetchrow(query, run_id)
        
    if not record:
        return None
        
    d = dict(record)
    d['metrics'] = json.loads(d['metrics_jsonb']) if d['metrics_jsonb'] else {}
    d['feature_importance'] = json.loads(d['feature_importance_jsonb']) if d['feature_importance_jsonb'] else {}
    del d['metrics_jsonb']
    del d['feature_importance_jsonb']
    return d

async def get_versions(conn: asyncpg.Connection, run_id: int):
    query = """
        SELECT id, version, is_active, estimator, metrics_jsonb, created_at
        FROM model_runs
        WHERE run_id = $1
        ORDER BY version DESC
    """
    records = await conn.fetch(query, run_id)
    result = []
    for r in records:
        d = dict(r)
        d['metrics'] = json.loads(d['metrics_jsonb']) if d['metrics_jsonb'] else {}
        del d['metrics_jsonb']
        result.append(d)
    return result
    
async def activate_version(conn: asyncpg.Connection, run_id: int, version: int):
    await conn.execute("UPDATE model_runs SET is_active = false WHERE run_id = $1", run_id)
    await conn.execute("UPDATE model_runs SET is_active = true WHERE run_id = $1 AND version = $2", run_id, version)
