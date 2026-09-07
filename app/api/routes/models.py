from fastapi import APIRouter, Depends, HTTPException
import asyncpg
import pandas as pd
import numpy as np
from app.core.db import get_db
from app.api.deps import get_current_user
from app.core.queries import model_runs as mr_queries
from app.core.storage import load_model, load_dataframe
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/{run_id}/versions")
async def get_versions(
    run_id: int, 
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if not owner:
        raise HTTPException(status_code=404, detail="Run not found")
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    return await mr_queries.get_versions(db, run_id)

@router.post("/{run_id}/activate")
async def activate_version(
    run_id: int,
    version: int,
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if not owner:
        raise HTTPException(status_code=404, detail="Run not found")
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    await mr_queries.activate_version(db, run_id, version)
    return {"status": "success", "message": f"Activated version {version}"}

@router.post("/{run_id}/validate")
async def validate_model(
    run_id: int,
    version: int = None,
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if not owner:
        raise HTTPException(status_code=404, detail="Run not found")
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    model_run = await mr_queries.get_model_run(db, run_id, version=version)
    if not model_run:
        raise HTTPException(status_code=404, detail="Model run not found")
        
    # Load artifact
    artifact = await load_model(db, run_id, model_run['r2_model_path'])
    preprocessor = artifact['preprocessor']
    model = artifact['model']
    test_idx = artifact.get('test_indices', [])
    
    # Load original cleaned data
    df_clean = await load_dataframe(db, run_id, stage='cleaning')
    
    # Extract test split
    df_test_raw = df_clean.loc[test_idx].copy()
    
    if len(df_test_raw) == 0:
        return {"status": "error", "message": "No test indices saved for this artifact"}
        
    # Preprocess the raw test set
    df_test_transformed = preprocessor.transform(df_test_raw)
    
    # Separate target
    pipeline = await db.fetchrow("SELECT target_column, task_type FROM pipeline_runs WHERE id = $1", run_id)
    target_column = pipeline['target_column']
    task_type = pipeline['task_type']
    
    # y_test must be extracted from the raw df because the preprocessor dropped it!
    y_test = df_test_raw[target_column]
    
    # X_test_transformed is exactly the output of the preprocessor since we used remainder='drop'
    # but in case it still has id, we drop it
    X_test_transformed = df_test_transformed
    if 'id' in X_test_transformed.columns:
        X_test_transformed = X_test_transformed.drop(columns=['id'])
    if target_column in X_test_transformed.columns:
        X_test_transformed = X_test_transformed.drop(columns=[target_column])
        
    # Predict
    y_pred = model.predict(X_test_transformed)
    
    # Calculate metrics dynamically
    metrics = {}
    if task_type.lower() == 'classification':
        from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, f1_score
        # we need training labels to know if binary? Actually we can check y_test or just rely on stored metric
        is_binary = len(np.unique(y_test.dropna())) == 2
        metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
        metrics['f1'] = float(f1_score(y_test, y_pred, average='binary' if is_binary else 'weighted'))
        metrics['precision'] = float(precision_score(y_test, y_pred, average='binary' if is_binary else 'weighted', zero_division=0))
        metrics['recall'] = float(recall_score(y_test, y_pred, average='binary' if is_binary else 'weighted', zero_division=0))
        if is_binary and hasattr(model, "predict_proba"):
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_test, model.predict_proba(X_test_transformed)[:, 1]))
            except:
                pass
    else:
        from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
        metrics['r2'] = float(r2_score(y_test, y_pred))
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics['mae'] = float(mean_absolute_error(y_test, y_pred))
        
    stored_metrics = model_run['metrics']
    
    # Compare
    mismatches = []
    for k in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'r2', 'rmse', 'mae']:
        if k in stored_metrics and k in metrics:
            if not np.isclose(stored_metrics[k], metrics[k], rtol=1e-5):
                mismatches.append(f"{k}: stored {stored_metrics[k]} != reproduced {metrics[k]}")
                
    if mismatches:
        return {
            "status": "failed",
            "message": "Validation failed: Reloaded metrics do not match training metrics.",
            "mismatches": mismatches,
            "reproduced_metrics": metrics,
            "stored_metrics": stored_metrics
        }
        
    return {
        "status": "success",
        "message": "Validation passed! Reproducibility confirmed.",
        "reproduced_metrics": metrics
    }
