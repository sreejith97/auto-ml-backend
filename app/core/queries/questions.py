import asyncpg
from typing import List, Optional
import json

async def insert_pending(
    conn: asyncpg.Connection,
    run_id: int,
    column_name: str,
    issue_type: str,
    confidence: float,
    evidence_jsonb: dict
) -> int:
    query = """
        INSERT INTO questions_pending (run_id, column_name, issue_type, confidence, evidence_jsonb, status)
        VALUES ($1, $2, $3, $4, $5, 'pending')
        RETURNING id
    """
    return await conn.fetchval(query, run_id, column_name, issue_type, confidence, json.dumps(evidence_jsonb))

async def get_pending(conn: asyncpg.Connection, run_id: int) -> List[asyncpg.Record]:
    query = "SELECT * FROM questions_pending WHERE run_id = $1 AND status = 'pending' ORDER BY id ASC"
    return await conn.fetch(query, run_id)

async def resolve_question(conn: asyncpg.Connection, question_id: int, answer: str):
    query = "UPDATE questions_pending SET status = 'resolved', evidence_jsonb = jsonb_set(COALESCE(evidence_jsonb, '{}'::jsonb), '{resolved_answer}', $1::jsonb) WHERE id = $2"
    await conn.execute(query, json.dumps(answer), question_id)

async def get_resolved(conn: asyncpg.Connection, run_id: int) -> List[asyncpg.Record]:
    query = "SELECT * FROM questions_pending WHERE run_id = $1 AND status = 'resolved' ORDER BY id ASC"
    return await conn.fetch(query, run_id)

async def insert_resolved_system(
    conn: asyncpg.Connection,
    run_id: int,
    column_name: str,
    issue_type: str,
    confidence: float,
    evidence_jsonb: dict,
    resolved_answer: str
) -> int:
    evidence_jsonb["resolved_answer"] = resolved_answer
    query = """
        INSERT INTO questions_pending (run_id, column_name, issue_type, confidence, evidence_jsonb, status)
        VALUES ($1, $2, $3, $4, $5, 'resolved')
        RETURNING id
    """
    return await conn.fetchval(query, run_id, column_name, issue_type, confidence, json.dumps(evidence_jsonb))
