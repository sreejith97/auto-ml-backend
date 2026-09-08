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
    existing_id = await conn.fetchval("SELECT id FROM rules WHERE column_signature = $1 LIMIT 1", column_signature)
    if existing_id:
        await conn.execute(
            """
            UPDATE rules 
            SET answer = $1, transformation = $2, use_count = use_count + 1, last_used_at = CURRENT_TIMESTAMP 
            WHERE id = $3
            """,
            answer, transformation, existing_id
        )
        return existing_id
    else:
        return await conn.fetchval(
            """
            INSERT INTO rules (column_signature, question, answer, transformation, dataset_scope, use_count, last_used_at)
            VALUES ($1, $2, $3, $4, $5, 1, CURRENT_TIMESTAMP)
            RETURNING id
            """,
            column_signature, question, answer, transformation, dataset_scope
        )

async def increment_rule_usage(conn: asyncpg.Connection, rule_id: int):
    query = """
        UPDATE rules 
        SET use_count = use_count + 1, last_used_at = CURRENT_TIMESTAMP 
        WHERE id = $1
    """
    await conn.execute(query, rule_id)
