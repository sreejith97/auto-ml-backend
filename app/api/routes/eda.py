from fastapi import APIRouter, Depends, HTTPException
import asyncpg
import pandas as pd
from app.core.db import get_db
from app.core.queries import columns as col_queries
from app.models.schemas import EDAResponse
from app.agents import eda, narration
from app.api.deps import get_current_user
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/{run_id}", response_model=EDAResponse)
async def get_eda(
    run_id: int, 
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    # 1. Fetch run info
    run_info = await db.fetchrow("SELECT dataset_id, task_type, target_column, user_id FROM pipeline_runs WHERE id = $1", run_id)
    if not run_info:
        raise HTTPException(status_code=404, detail="Run not found")
    if run_info['user_id'] != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    dataset_id = run_info['dataset_id']
    task_type = run_info['task_type']
    target_column = run_info['target_column']
    
    from app.core.storage import load_dataframe
    try:
        df = await load_dataframe(db, run_id, stage='cleaning')
        logger.info("Successfully loaded dataframe for EDA.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")
        
    cols_spec = await col_queries.get_columns_for_dataset(db, dataset_id)
    cols_list = [dict(c) for c in cols_spec]
    
    # 2. Run deterministic EDA logic
    stats = eda.compute_summary_stats(df, cols_list)
    correlation = eda.compute_correlation_matrix(df, cols_list)
    class_balance = eda.compute_class_balance(df, target_column, task_type)
    
    # 3. Generate structured findings
    findings = eda.generate_findings_summary(stats, correlation, class_balance)
    
    # 4. Narrate the findings (uses LLM only for phrasing)
    narration_text = await narration.narrate_eda(db, run_id, findings)
    
    return EDAResponse(
        summary_stats=stats,
        correlation_matrix=correlation,
        class_balance=class_balance,
        findings_summary=findings,
        narration=narration_text
    )
