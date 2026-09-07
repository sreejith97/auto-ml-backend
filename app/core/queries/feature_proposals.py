import asyncpg
import json
from typing import List, Dict, Any, Optional

async def insert_proposal(
    conn: asyncpg.Connection, 
    run_id: int, 
    column_name: str, 
    proposed_transform: str, 
    params: dict, 
    rationale: str, 
    status: str = 'pending'
) -> int:
    query = """
        INSERT INTO feature_proposals (run_id, column_name, proposed_transform, params_jsonb, rationale, status)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
    """
    params_json = json.dumps(params) if params else None
    return await conn.fetchval(query, run_id, column_name, proposed_transform, params_json, rationale, status)

async def get_proposals(conn: asyncpg.Connection, run_id: int) -> List[Dict[str, Any]]:
    query = """
        SELECT id, run_id, column_name, proposed_transform, params_jsonb, rationale, final_transform, final_params_jsonb, status, created_at
        FROM feature_proposals
        WHERE run_id = $1
        ORDER BY id ASC
    """
    records = await conn.fetch(query, run_id)
    results = []
    for r in records:
        d = dict(r)
        d['params'] = json.loads(d['params_jsonb']) if d['params_jsonb'] else {}
        d['final_params'] = json.loads(d['final_params_jsonb']) if d['final_params_jsonb'] else {}
        # Clean up raw json strings
        del d['params_jsonb']
        del d['final_params_jsonb']
        results.append(d)
    return results

async def update_proposal(
    conn: asyncpg.Connection, 
    proposal_id: int, 
    final_transform: str, 
    params: dict, 
    status: str
):
    query = """
        UPDATE feature_proposals 
        SET final_transform = $1, final_params_jsonb = $2, status = $3
        WHERE id = $4
    """
    params_json = json.dumps(params) if params else None
    await conn.execute(query, final_transform, params_json, status, proposal_id)

async def count_pending_proposals(conn: asyncpg.Connection, run_id: int) -> int:
    query = "SELECT COUNT(*) FROM feature_proposals WHERE run_id = $1 AND status = 'pending'"
    return await conn.fetchval(query, run_id)
