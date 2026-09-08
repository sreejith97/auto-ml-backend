from fastapi import APIRouter, Depends, HTTPException
import asyncpg
from app.core.db import get_db
from app.core.queries import columns as col_queries
from app.models.schemas import EDAResponse
from app.api.deps import get_current_user
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/{run_id}", response_model=EDAResponse)
async def get_eda(
    run_id: int,
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    run_info = await db.fetchrow(
        "SELECT dataset_id, task_type, target_column, user_id FROM pipeline_runs WHERE id = $1",
        run_id,
    )
    if not run_info:
        raise HTTPException(status_code=404, detail="Run not found")
    if run_info["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    from app.core.storage import load_dataframe
    try:
        df = await load_dataframe(db, run_id, stage="cleaning")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    cols_spec = await col_queries.get_columns_for_dataset(db, run_info["dataset_id"])
    cols_list = [dict(c) for c in cols_spec]

    # Delegate to Agno EDA step (handles stats + correlation + class_balance + synthesis)
    from app.workflows.steps.eda_step import run_eda
    result = await run_eda(
        run_id=run_id,
        conn=db,
        df=df,
        columns_spec=cols_list,
        target_col=run_info["target_column"],
        task_type=run_info["task_type"],
    )

    return EDAResponse(
        summary_stats=result["stats"],
        correlation_matrix=result["correlation"],
        class_balance=result["class_balance"],
        findings_summary=result["findings"],
        narration=result["narration"],
    )
