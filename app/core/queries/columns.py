import asyncpg
from typing import List, Optional

async def upsert_column_spec(
    conn: asyncpg.Connection, 
    dataset_id: int, 
    column_name: str, 
    role: str, 
    semantic_type: str, 
    missing_meaning: Optional[str], 
    notes: Optional[str]
) -> int:
    query = """
        INSERT INTO columns_spec (dataset_id, column_name, role, semantic_type, missing_meaning, notes)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (dataset_id, column_name) 
        DO UPDATE SET 
            role = EXCLUDED.role,
            semantic_type = EXCLUDED.semantic_type,
            missing_meaning = EXCLUDED.missing_meaning,
            notes = EXCLUDED.notes
        RETURNING id
    """
    return await conn.fetchval(query, dataset_id, column_name, role, semantic_type, missing_meaning, notes)

async def get_columns_for_dataset(conn: asyncpg.Connection, dataset_id: int) -> List[asyncpg.Record]:
    query = "SELECT * FROM columns_spec WHERE dataset_id = $1 ORDER BY id ASC"
    return await conn.fetch(query, dataset_id)

async def count_target_columns(conn: asyncpg.Connection, dataset_id: int) -> int:
    query = "SELECT COUNT(*) FROM columns_spec WHERE dataset_id = $1 AND role = 'target'"
    return await conn.fetchval(query, dataset_id)
