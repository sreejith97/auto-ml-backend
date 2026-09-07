from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
import asyncpg
import pandas as pd
import io
from app.core.db import get_db
from app.api.deps import get_current_user
from app.agents.inference import load_active_model, validate_input_row, predict_single, predict_batch
from app.core.queries import columns as col_queries
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class PredictRequest(BaseModel):
    row: dict

@router.get("/{run_id}/schema")
async def get_schema(
    run_id: int, 
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if not owner:
        raise HTTPException(status_code=404, detail="Run not found")
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    dataset_id = await db.fetchval("SELECT dataset_id FROM pipeline_runs WHERE id = $1", run_id)
    cols_spec = await col_queries.get_columns_for_dataset(db, dataset_id)
    
    schema = []
    for c in cols_spec:
        if c['role'] == 'feature':
            schema.append({
                "column_name": c['column_name'],
                "semantic_type": c['semantic_type']
            })
            
    return {"schema": schema}


@router.post("/{run_id}")
async def predict_single_row(
    run_id: int,
    request: PredictRequest,
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if not owner:
        raise HTTPException(status_code=404, detail="Run not found")
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    # Load model
    try:
        bundle = await load_active_model(db, run_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
        
    # Validate columns
    dataset_id = await db.fetchval("SELECT dataset_id FROM pipeline_runs WHERE id = $1", run_id)
    cols_spec = await col_queries.get_columns_for_dataset(db, dataset_id)
    
    missing = validate_input_row(request.row, [dict(c) for c in cols_spec])
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required feature columns: {missing}")
        
    # Get task type
    task_type = await db.fetchval("SELECT task_type FROM pipeline_runs WHERE id = $1", run_id)
    
    # Predict
    try:
        pred, prob, warnings = predict_single(bundle, request.row, task_type)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
        
    response = {
        "prediction": pred,
        "warnings": warnings
    }
    if prob is not None:
        response["probability"] = prob
        
    return response


@router.post("/{run_id}/batch")
async def predict_batch_csv(
    run_id: int,
    file: UploadFile = File(...),
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if not owner:
        raise HTTPException(status_code=404, detail="Run not found")
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    # Load model
    try:
        bundle = await load_active_model(db, run_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
        
    # Read CSV
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read CSV")
        
    # Validate columns
    dataset_id = await db.fetchval("SELECT dataset_id FROM pipeline_runs WHERE id = $1", run_id)
    cols_spec = await col_queries.get_columns_for_dataset(db, dataset_id)
    
    required_features = [c['column_name'] for c in cols_spec if c['role'] == 'feature']
    missing = []
    for f in required_features:
        if f not in df.columns:
            missing.append(f)
            
    if missing:
        raise HTTPException(status_code=400, detail=f"CSV missing required feature columns: {missing}")
        
    # Get task type
    task_type = await db.fetchval("SELECT task_type FROM pipeline_runs WHERE id = $1", run_id)
    
    # Predict
    try:
        results = predict_batch(bundle, df, task_type)
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")
        
    return {"results": results}
