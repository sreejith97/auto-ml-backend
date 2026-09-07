import asyncpg
import numpy as np
import pandas as pd
from app.core.storage import load_dataframe, save_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import logging

logger = logging.getLogger(__name__)

async def train_model(db: asyncpg.Connection, run_id: int):
    import json
    import os
    import boto3
    from app.core.storage import load_model
    
    # 1. Load data
    df_train = await load_dataframe(db, run_id, stage='features_train')
    df_test = await load_dataframe(db, run_id, stage='features_test')
    
    # 2. Get pipeline info
    pipeline = await db.fetchrow("SELECT target_column, task_type, dataset_id FROM pipeline_runs WHERE id = $1", run_id)
    target_column = pipeline['target_column']
    task_type = pipeline['task_type']
    dataset_id = pipeline['dataset_id']
    
    # 3. Load test indices
    indices_path = await db.fetchval("SELECT r2_snapshot_path FROM stage_outputs WHERE run_id = $1 AND stage = 'split_indices' ORDER BY created_at DESC LIMIT 1", run_id)
    from app.core.storage import get_s3_client
    s3 = get_s3_client()
    bucket = os.getenv("R2_BUCKET_NAME")
    try:
        if not s3 or not bucket: raise Exception("No R2 client")
        obj = s3.get_object(Bucket=bucket, Key=indices_path)
        indices_json = json.loads(obj['Body'].read().decode('utf-8'))
        test_idx = indices_json['test_idx']
        train_idx = indices_json['train_idx']
    except:
        # Fallback to local
        local_dir = f"local_storage/{run_id}"
        local_path = os.path.join(local_dir, indices_path)
        with open(local_path, "r") as f:
            indices_json = json.load(f)
            test_idx = indices_json['test_idx']
            train_idx = indices_json['train_idx']
            
    # Load preprocessor
    preprocessor_path = await db.fetchval("SELECT r2_snapshot_path FROM stage_outputs WHERE run_id = $1 AND stage = 'preprocessor' ORDER BY created_at DESC LIMIT 1", run_id)
    preprocessor = await load_model(db, run_id, preprocessor_path)
    
    # 4. Separate X and y
    if target_column not in df_train.columns:
        raise ValueError(f"Target column '{target_column}' missing from feature matrix.")
        
    y_train = df_train[target_column]
    y_test = df_test[target_column]
    
    cols_spec = await db.fetch("SELECT column_name, role FROM columns_spec WHERE dataset_id = $1", dataset_id)
    excluded_cols = [c['column_name'] for c in cols_spec if c['role'] in ('id', 'ignore')]
    
    X_train = df_train.drop(columns=[target_column])
    X_train = X_train.drop(columns=[c for c in excluded_cols if c in X_train.columns])
    
    X_test = df_test.drop(columns=[target_column])
    X_test = X_test.drop(columns=[c for c in excluded_cols if c in X_test.columns])
    
    # Fill any remaining NaNs safely
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Ensure all columns are numeric
    non_numeric = []
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or str(X_train[col].dtype) == 'category':
            non_numeric.append(col)
            
    if non_numeric:
        raise ValueError(f"Feature matrix contains non-numeric columns that must be transformed or ignored: {non_numeric}")
            
    # 5. Model selection
    if task_type.lower() == 'classification':
        models = {
            'RandomForestClassifier': RandomForestClassifier(n_estimators=50, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
        }
        is_binary = len(np.unique(y_train.dropna())) == 2
        scoring = 'roc_auc' if is_binary else 'f1_weighted'
    else:
        models = {
            'RandomForestRegressor': RandomForestRegressor(n_estimators=50, random_state=42),
            'LinearRegression': LinearRegression()
        }
        scoring = 'r2'
        
    best_model_name = None
    best_model = None
    best_score = -float('inf')
    
    for name, model in models.items():
        try:
            # Quick CV on TRAIN ONLY
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring=scoring)
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = name
                best_model = model
        except Exception as e:
            logger.error(f"Failed to CV {name}: {e}")
            
    if best_model is None:
        best_model_name = list(models.keys())[0]
        best_model = list(models.values())[0]
        
    # 6. Fit best model on entire train dataset
    best_model.fit(X_train, y_train)
    
    # 7. Evaluate on held-out test split
    if len(X_test) > 0:
        y_pred = best_model.predict(X_test)
        
        metrics = {}
        if task_type.lower() == 'classification':
            from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, f1_score
            is_binary = len(np.unique(y_train.dropna())) == 2
            metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
            metrics['f1'] = float(f1_score(y_test, y_pred, average='binary' if is_binary else 'weighted'))
            metrics['precision'] = float(precision_score(y_test, y_pred, average='binary' if is_binary else 'weighted', zero_division=0))
            metrics['recall'] = float(recall_score(y_test, y_pred, average='binary' if is_binary else 'weighted', zero_division=0))
            if is_binary and hasattr(best_model, "predict_proba"):
                try:
                    metrics['roc_auc'] = float(roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))
                except:
                    pass
            metrics['cv_score'] = float(best_score)
            metrics['scoring_metric'] = scoring
        else:
            from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
            metrics['r2'] = float(r2_score(y_test, y_pred))
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics['mae'] = float(mean_absolute_error(y_test, y_pred))
            metrics['cv_r2'] = float(best_score)
            metrics['scoring_metric'] = scoring
    else:
        metrics = {"error": "Test set too small or empty"}
        
    # 8. Extract feature importance
    importances = {}
    if hasattr(best_model, 'feature_importances_'):
        fi = best_model.feature_importances_
        importances = {k: float(v) for k, v in zip(X_train.columns, fi)}
    elif hasattr(best_model, 'coef_'):
        coefs = best_model.coef_[0] if best_model.coef_.ndim > 1 else best_model.coef_
        importances = {k: float(abs(v)) for k, v in zip(X_train.columns, coefs)}
        
    # Normalize importances
    total_imp = sum(importances.values())
    if total_imp > 0:
        importances = {k: v/total_imp for k, v in importances.items()}
        
    # 9. Get feature names from preprocessor
    # ColumnTransformer feature names
    try:
        preprocessor_features = preprocessor.get_feature_names_out().tolist()
    except:
        preprocessor_features = list(X_train.columns)
        
    # 9.5 Extract raw min/max bounds and known categories
    df_clean = await load_dataframe(db, run_id, stage='cleaning')
    df_train_raw = df_clean.loc[train_idx].copy()
    
    cols_spec = await db.fetch("SELECT column_name, role FROM columns_spec WHERE dataset_id = $1", dataset_id)
    raw_feature_cols = [c['column_name'] for c in cols_spec if c['role'] == 'feature']
    
    numeric_bounds = {}
    known_categories = {}
    for col in df_train_raw.columns:
        if col not in raw_feature_cols: continue
        if pd.api.types.is_numeric_dtype(df_train_raw[col]):
            numeric_bounds[col] = {
                "min": float(df_train_raw[col].min()),
                "max": float(df_train_raw[col].max())
            }
        else:
            known_categories[col] = df_train_raw[col].dropna().astype(str).unique().tolist()
            
    # 10. Package artifact
    artifact = {
        "preprocessor": preprocessor,
        "model": best_model,
        "feature_columns": list(X_train.columns),
        "preprocessor_features": preprocessor_features,
        "test_indices": test_idx,
        "numeric_bounds": numeric_bounds,
        "known_categories": known_categories
    }
    
    # Calculate next version
    max_ver = await db.fetchval("SELECT MAX(version) FROM model_runs WHERE run_id = $1", run_id)
    next_ver = (max_ver or 0) + 1
    
    filename = f"model_{run_id}_v{next_ver}_{best_model_name.lower()}.joblib"
    saved_path = await save_model(db, artifact, run_id, filename)
    
    # Insert model run
    # Wait, we should do insertion here instead of just returning since we need to set is_active=True?
    # No, returning it is fine, the router (routes/training.py) inserts it.
    
    return {
        "estimator": best_model_name,
        "metrics": metrics,
        "feature_importance": importances,
        "path": saved_path,
        "x_columns": list(X_train.columns),
        "version": next_ver
    }
