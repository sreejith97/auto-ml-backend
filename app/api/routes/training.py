from fastapi import APIRouter, Depends, HTTPException
import asyncpg
from app.core.db import get_db
from app.api.deps import get_current_user
from app.agents import training as tr
from app.agents import narration
from app.core.queries import model_runs as mr_queries
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/{run_id}/run")
async def run_training(
    run_id: int, 
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if not owner:
        raise HTTPException(status_code=404, detail="Run not found")
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    # Check if already trained
    existing = await mr_queries.get_model_run(db, run_id)
    if existing:
        return {"status": "success", "message": "Already trained"}
        
    try:
        # Run agent
        result = await tr.train_model(db, run_id)
        
        # Save results
        await mr_queries.insert_model_run(
            conn=db,
            run_id=run_id,
            estimator=result['estimator'],
            metrics=result['metrics'],
            r2_model_path=result['path'],
            feature_importance=result['feature_importance'],
            version=result['version']
        )
        
        # Update pipeline state
        await db.execute("UPDATE pipeline_runs SET status = 'trained' WHERE id = $1", run_id)
        
        # Narrate results
        primary_metric = result['metrics'].get('cv_accuracy') or result['metrics'].get('cv_r2', 0)
        await narration.narrate_results(db, run_id, result['estimator'], primary_metric)
        
        return {"status": "success", "x_columns": result["x_columns"]}
    except Exception as e:
        import traceback
        logger.error(f"Training failed for run {run_id}:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{run_id}/results")
async def get_results(
    run_id: int,
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if not owner:
        raise HTTPException(status_code=404, detail="Run not found")
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    run_data = await mr_queries.get_model_run(db, run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Model run not found")
        
    return run_data
