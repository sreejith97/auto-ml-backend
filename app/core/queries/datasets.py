import asyncpg

async def create_dataset(conn: asyncpg.Connection, name: str, source_type: str, r2_raw_path: str, row_count: int) -> int:
    query = """
        INSERT INTO datasets (name, source_type, r2_raw_path, row_count)
        VALUES ($1, $2, $3, $4)
        RETURNING id
    """
    return await conn.fetchval(query, name, source_type, r2_raw_path, row_count)

async def get_dataset(conn: asyncpg.Connection, dataset_id: int) -> asyncpg.Record:
    query = "SELECT * FROM datasets WHERE id = $1"
    return await conn.fetchrow(query, dataset_id)
