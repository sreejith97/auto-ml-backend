"""
app/agents/narration.py — Narration Agent (Agno-backed shim)
=============================================================
All narrate_*() functions now delegate to the Agno NarrationAgent in
`app/workflows/steps/narration_step.py`.

The function signatures are unchanged so all existing route callers
continue to work without modification during Phase 4.
"""

import asyncpg
import logging

logger = logging.getLogger(__name__)


async def narrate_ingestion(conn: asyncpg.Connection, run_id: int, dataset_name: str, row_count: int, schema_list: list = None):
    from app.workflows.steps.narration_step import narrate_ingestion as _narrate
    return await _narrate(conn, run_id, dataset_name, row_count, schema_list)


async def narrate_columns(conn: asyncpg.Connection, run_id: int, target_col: str):
    from app.workflows.steps.narration_step import narrate_columns as _narrate
    return await _narrate(conn, run_id, target_col)


async def narrate_task(conn: asyncpg.Connection, run_id: int, task_type: str, target_col: str):
    from app.workflows.steps.narration_step import narrate_task as _narrate
    return await _narrate(conn, run_id, task_type, target_col)


async def narrate_cleaning(conn: asyncpg.Connection, run_id: int, auto_applied: int, pending: int):
    from app.workflows.steps.narration_step import narrate_cleaning as _narrate
    return await _narrate(conn, run_id, auto_applied, pending)


async def narrate_eda(conn: asyncpg.Connection, run_id: int, findings: list) -> str:
    from app.workflows.steps.narration_step import narrate_eda as _narrate
    return await _narrate(conn, run_id, findings)


async def narrate_features(conn: asyncpg.Connection, run_id: int, proposals_list: list):
    from app.workflows.steps.narration_step import narrate_features as _narrate
    return await _narrate(conn, run_id, proposals_list)


async def narrate_results(conn: asyncpg.Connection, run_id: int, top_model: str, accuracy: float):
    from app.workflows.steps.narration_step import narrate_results as _narrate
    return await _narrate(conn, run_id, top_model, accuracy)
