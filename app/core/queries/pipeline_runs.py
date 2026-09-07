import asyncpg
from typing import Optional

async def insert_run(
    conn: asyncpg.Connection,
    user_id: int,
    dataset_id: int,
    task_type: str,
    target_column: str,
    time_column: Optional[str] = None,
    forecast_horizon: Optional[int] = None
) -> int:
    query = """
        INSERT INTO pipeline_runs (user_id, dataset_id, task_type, target_column, time_column, forecast_horizon, status)
        VALUES ($1, $2, $3, $4, $5, $6, 'task_defined')
        RETURNING id
    """
    return await conn.fetchval(query, user_id, dataset_id, task_type, target_column, time_column, forecast_horizon)

async def update_run_task(
    conn: asyncpg.Connection,
    run_id: int,
    task_type: str,
    target_column: str,
    time_column: Optional[str] = None,
    forecast_horizon: Optional[int] = None
) -> None:
    query = """
        UPDATE pipeline_runs 
        SET task_type = $1, target_column = $2, time_column = $3, forecast_horizon = $4, status = 'task_defined'
        WHERE id = $5
    """
    await conn.execute(query, task_type, target_column, time_column, forecast_horizon, run_id)
