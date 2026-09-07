import asyncpg
from typing import List, Optional

async def insert_message(
    conn: asyncpg.Connection,
    run_id: int,
    role: str,
    content: str,
    stage: str,
    related_question_id: Optional[int] = None
) -> int:
    query = """
        INSERT INTO conversation_messages (run_id, role, content, stage, related_question_id)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
    """
    return await conn.fetchval(query, run_id, role, content, stage, related_question_id)

async def get_messages(conn: asyncpg.Connection, run_id: int) -> List[asyncpg.Record]:
    query = """
        SELECT * FROM conversation_messages 
        WHERE run_id = $1 
        ORDER BY created_at ASC
    """
    return await conn.fetch(query, run_id)
