from fastapi import APIRouter, Depends, HTTPException
import asyncpg
import pandas as pd
from typing import List
from app.core.db import get_db
from app.core.storage import load_dataframe, save_snapshot
from app.core.queries import columns as col_queries
from app.core.queries import feature_proposals as fp_queries
from app.api.deps import get_current_user
from app.agents import feature_engineering as fe
from app.agents import narration
from pydantic import BaseModel
import logging
import numpy as np

def log1p_transform(X):
    return np.log1p(np.clip(X, a_min=0, a_max=None))

def datetime_features(X):
    col = X.columns[0]
    dt = pd.to_datetime(X[col], errors='coerce')
    out = pd.DataFrame({
        f"{col}_year": dt.dt.year,
        f"{col}_month": dt.dt.month,
        f"{col}_day": dt.dt.day,
        f"{col}_dayofweek": dt.dt.dayofweek
    }, index=X.index)
    return out

def get_dt_names(estimator, input_features):
    col = input_features[0]
    return [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek"]

logger = logging.getLogger(__name__)
router = APIRouter()

class ApproveRequest(BaseModel):
    proposal_id: int
    final_transform: str
    final_params: dict

@router.get("/{run_id}")
async def get_features(
    run_id: int, 
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    proposals = await fp_queries.get_proposals(db, run_id)
    return proposals

@router.post("/{run_id}/propose")
async def propose_features(
    run_id: int, 
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if not owner:
        raise HTTPException(status_code=404, detail="Run not found")
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    # Check if proposals already exist (idempotency)
    existing_proposals = await fp_queries.get_proposals(db, run_id)
    if existing_proposals:
        return existing_proposals
        
    dataset_id = await db.fetchval("SELECT dataset_id FROM pipeline_runs WHERE id = $1", run_id)
    
    # Load cleaned data
    try:
        df = await load_dataframe(db, run_id, stage='cleaning')
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")
        
    cols_spec = await col_queries.get_columns_for_dataset(db, dataset_id)
    
    new_proposals = []
    
    for col_record in cols_spec:
        col_name = col_record['column_name']
        if col_name not in df.columns:
            continue
            
        proposal = fe.propose_transform(col_name, dict(col_record), df[col_name])
        if proposal:
            proposal_id = await fp_queries.insert_proposal(
                conn=db,
                run_id=run_id,
                column_name=proposal['column_name'],
                proposed_transform=proposal['proposed_transform'],
                params=proposal['params'],
                rationale=proposal['rationale']
            )
            proposal['id'] = proposal_id
            proposal['status'] = 'pending'
            new_proposals.append(proposal)
            
    await narration.narrate_features(db, run_id, new_proposals)
    
    # Fetch final format from DB to match GET output
    return await fp_queries.get_proposals(db, run_id)


@router.post("/{run_id}/approve")
async def approve_features(
    run_id: int,
    approvals: List[ApproveRequest],
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    # Process approvals
    for app in approvals:
        await fp_queries.update_proposal(
            conn=db,
            proposal_id=app.proposal_id,
            final_transform=app.final_transform,
            params=app.final_params,
            status='approved'
        )
        
    # Check if any pending remain
    pending_count = await fp_queries.count_pending_proposals(db, run_id)
    if pending_count > 0:
        return {"status": "partial", "message": f"{pending_count} proposals still pending"}
        
    # All approved. Apply transforms
    try:
        df = await load_dataframe(db, run_id, stage='cleaning')
    except Exception as e:
        logger.error(f"Failed to load cleaning data for run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load cleaning data: {e}")
    
    proposals = await fp_queries.get_proposals(db, run_id)
    
    dataset_id = await db.fetchval("SELECT dataset_id FROM pipeline_runs WHERE id = $1", run_id)
    target_col = await db.fetchval("SELECT target_column FROM pipeline_runs WHERE id = $1", run_id)
    cols_spec = await col_queries.get_columns_for_dataset(db, dataset_id)
    
    excluded_cols = [c['column_name'] for c in cols_spec if c['role'] in ('id', 'ignore')]
    if target_col and target_col not in excluded_cols:
        excluded_cols.append(target_col)
    
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.model_selection import train_test_split
    from app.core.storage import save_model
    import json
    import os
    import boto3
    
    # We will split df into train and test
    try:
        train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    except ValueError:
        # Fallback for very small datasets
        train_idx = df.index
        test_idx = pd.Index([])
        
    transformers = []
    drop_cols = []
    passthrough_cols = []
    seen_cols = set()
    for p in proposals:
        col = p['column_name']
        if col not in df.columns or col in seen_cols:
            continue
            
        seen_cols.add(col)
        trans = p['final_transform']
        
        if trans == "one_hot_encode":
            transformers.append((f"ohe_{col}", OneHotEncoder(handle_unknown='ignore', sparse_output=False), [col]))
        elif trans == "ordinal_encode":
            transformers.append((f"ord_{col}", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), [col]))
        elif trans == "standard_scale":
            transformers.append((f"std_{col}", StandardScaler(), [col]))
        elif trans == "log1p_standard_scale":
            pipe = Pipeline([
                ('log1p', FunctionTransformer(log1p_transform, feature_names_out='one-to-one')),
                ('scaler', StandardScaler())
            ])
            transformers.append((f"logstd_{col}", pipe, [col]))
        elif trans == "datetime_decompose":
            # Just manual function transformer
            transformers.append((f"dt_{col}", FunctionTransformer(datetime_features, feature_names_out=get_dt_names), [col]))
        elif trans == "drop":
            drop_cols.append(col)
        elif trans in ["passthrough", "manual_handling"]:
            passthrough_cols.append(col)
            
    if drop_cols:
        transformers.append(("drop_trans", "drop", drop_cols))
    if passthrough_cols:
        transformers.append(("pass_trans", "passthrough", passthrough_cols))
    
    if not transformers:
        raise HTTPException(status_code=400, detail="No valid feature transforms to apply. Check your proposals.")
        
    try:
        preprocessor = ColumnTransformer(
            transformers, 
            remainder='drop', 
            verbose_feature_names_out=False
        )
        preprocessor.set_output(transform="pandas")
        
        df_train = df.loc[train_idx].copy()
        df_test = df.loc[test_idx].copy()
        
        df_train_excluded = df_train[[c for c in excluded_cols if c in df_train.columns]]
        df_test_excluded = df_test[[c for c in excluded_cols if c in df_test.columns]]
        
        df_train_features = df_train.drop(columns=[c for c in excluded_cols if c in df_train.columns])
        df_test_features = df_test.drop(columns=[c for c in excluded_cols if c in df_test.columns])
        
        # Fit on train features only
        df_train_transformed = preprocessor.fit_transform(df_train_features)
        
        # Transform test features
        if len(df_test_features) > 0:
            df_test_transformed = preprocessor.transform(df_test_features)
        else:
            df_test_transformed = pd.DataFrame(columns=df_train_transformed.columns)
            
        # Reattach excluded columns (id, ignore, target)
        for c in df_train_excluded.columns:
            df_train_transformed[c] = df_train_excluded[c].values
            if len(df_test_features) > 0:
                df_test_transformed[c] = df_test_excluded[c].values
    except Exception as e:
        logger.error(f"Feature transformation failed for run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Feature transformation failed: {e}")
        
    # Save transformed features
    try:
        train_snapshot = await save_snapshot(db, df_train_transformed, run_id, f"dataset_{run_id}_features_train.csv")
        test_snapshot = await save_snapshot(db, df_test_transformed, run_id, f"dataset_{run_id}_features_test.csv")
        
        # Save preprocessor to R2
        preprocessor_filename = f"preprocessor_{run_id}.joblib"
        preprocessor_path = await save_model(db, preprocessor, run_id, preprocessor_filename)
        
        # Save test indices to R2
        indices_filename = f"split_indices_{run_id}.json"
        indices_json = json.dumps({"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()})
        
        from app.core.storage import get_s3_client
        r2 = get_s3_client()
        r2_bucket = os.getenv("R2_BUCKET_NAME")
        if r2 and r2_bucket:
            r2.put_object(Bucket=r2_bucket, Key=indices_filename, Body=indices_json.encode('utf-8'))
            indices_path = indices_filename
        else:
            local_dir = f"local_storage/{run_id}"
            os.makedirs(local_dir, exist_ok=True)
            indices_path = os.path.join(local_dir, indices_filename)
            with open(indices_path, "w") as f:
                f.write(indices_json)
    except Exception as e:
        logger.error(f"Failed to save artifacts for run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save artifacts: {e}")
                
    # Update stage outputs — use DELETE+INSERT to handle re-runs idempotently
    for stage_name in ['features_train', 'features_test', 'preprocessor', 'split_indices']:
        await db.execute("DELETE FROM stage_outputs WHERE run_id = $1 AND stage = $2", run_id, stage_name)
    
    await db.execute("UPDATE pipeline_runs SET status = 'features_engineered' WHERE id = $1", run_id)
    await db.execute("INSERT INTO stage_outputs (run_id, stage, r2_snapshot_path) VALUES ($1, 'features_train', $2)", run_id, train_snapshot)
    await db.execute("INSERT INTO stage_outputs (run_id, stage, r2_snapshot_path) VALUES ($1, 'features_test', $2)", run_id, test_snapshot)
    await db.execute("INSERT INTO stage_outputs (run_id, stage, r2_snapshot_path) VALUES ($1, 'preprocessor', $2)", run_id, preprocessor_path)
    await db.execute("INSERT INTO stage_outputs (run_id, stage, r2_snapshot_path) VALUES ($1, 'split_indices', $2)", run_id, indices_path)
    
    return {"status": "success", "message": "Features engineered and split successfully"}

