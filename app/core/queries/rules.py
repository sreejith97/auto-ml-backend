import asyncpg
from typing import Optional

async def find_matching_rule(conn: asyncpg.Connection, column_signature: str) -> Optional[asyncpg.Record]:
    query = """
        SELECT * FROM rules 
        WHERE column_signature = $1 
        ORDER BY created_at DESC 
        LIMIT 1
    """
    return await conn.fetchrow(query, column_signature)

async def upsert_rule(
    conn: asyncpg.Connection, 
    column_signature: str, 
    question: str, 
    answer: str, 
    transformation: str, 
    dataset_scope: str = 'global'
) -> int:
    query = """
        INSERT INTO rules (column_signature, question, answer, transformation, dataset_scope, use_count, last_used_at)
        VALUES ($1, $2, $3, $4, $5, 1, CURRENT_TIMESTAMP)
        ON CONFLICT (column_signature) DO UPDATE SET
            answer = EXCLUDED.answer,
            transformation = EXCLUDED.transformation,
            use_count = rules.use_count + 1,
            last_used_at = CURRENT_TIMESTAMP
        RETURNING id
    """
    return await conn.fetchval(query, column_signature, question, answer, transformation, dataset_scope)

async def increment_rule_usage(conn: asyncpg.Connection, rule_id: int):
    query = """
        UPDATE rules 
        SET use_count = use_count + 1, last_used_at = CURRENT_TIMESTAMP 
        WHERE id = $1
    """
    await conn.execute(query, rule_id)
