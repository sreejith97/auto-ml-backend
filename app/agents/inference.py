import pandas as pd
import numpy as np
import asyncpg
from typing import List, Dict, Any, Tuple
from app.core.storage import load_model
from app.core.queries.model_runs import get_model_run

async def load_active_model(db: asyncpg.Connection, run_id: int) -> dict:
    model_run = await get_model_run(db, run_id) # get_model_run returns active or newest by default
    if not model_run or not model_run.get('is_active'):
        raise ValueError(f"No active model found for run {run_id}")
    
    artifact = await load_model(db, run_id, model_run['r2_model_path'])
    return artifact

def validate_input_row(row: dict, columns_spec: List[dict]) -> List[str]:
    required_features = [c['column_name'] for c in columns_spec if c['role'] == 'feature']
    missing = []
    for f in required_features:
        if f not in row:
            missing.append(f)
    return missing

def process_warnings(bundle: dict, df: pd.DataFrame) -> List[List[str]]:
    numeric_bounds = bundle.get("numeric_bounds", {})
    known_categories = bundle.get("known_categories", {})
    
    all_warnings = []
    for idx, row in df.iterrows():
        row_warnings = []
        for col, val in row.items():
            if pd.isna(val):
                continue
                
            # Check OOD
            if col in numeric_bounds:
                try:
                    num_val = float(val)
                    c_min = numeric_bounds[col]['min']
                    c_max = numeric_bounds[col]['max']
                    # Configurable margin, e.g. 20% of range
                    range_span = c_max - c_min
                    margin = range_span * 0.2 if range_span > 0 else 0
                    
                    if num_val < (c_min - margin) or num_val > (c_max + margin):
                        row_warnings.append(f"Input value {num_val} for column '{col}' is outside the training range [{c_min}, {c_max}] (with 20% margin).")
                except:
                    pass
                    
            # Check unseen categories
            if col in known_categories:
                val_str = str(val)
                if val_str not in known_categories[col]:
                    row_warnings.append(f"Unseen category '{val_str}' in column '{col}'.")
                    
        all_warnings.append(row_warnings)
    return all_warnings

def predict_single(bundle: dict, row: dict, task_type: str) -> Tuple[Any, float, List[str]]:
    # Remove id or target if they exist in row (safety)
    # The preprocessor handles whatever columns are given, but passthrough could pass target
    # So we strictly use what was given minus any non-features if we had a spec,
    # but practically we just let the preprocessor handle it.
    
    df = pd.DataFrame([row])
    
    # Check warnings
    warnings = process_warnings(bundle, df)[0]
    
    # Preprocess
    preprocessor = bundle['preprocessor']
    model = bundle['model']
    
    # Ensure all columns required by preprocessor are present
    X_transformed = preprocessor.transform(df)
    
    # Depending on passthrough, target might have been passed through if it was in the row!
    # So we must ensure X_transformed matches feature_columns
    feature_cols = bundle['feature_columns']
    # If the user passed target, it might be in X_transformed. 
    # But preprocessor.transform outputs a dataframe with same columns as df, unless they were selected.
    # To be safe, we select exactly the columns the model was trained on
    if isinstance(X_transformed, pd.DataFrame):
        try:
            X_final = X_transformed[feature_cols]
        except KeyError:
            # Fallback
            X_final = X_transformed
    else:
        X_final = X_transformed
        
    pred = model.predict(X_final)[0]
    prob = None
    
    if task_type.lower() == 'classification' and hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_final)[0]
        # binary prob
        if len(probs) == 2:
            prob = float(probs[1])
        else:
            # multiclass, take max prob
            prob = float(max(probs))
            
    return pred, prob, warnings

def predict_batch(bundle: dict, df: pd.DataFrame, task_type: str) -> List[dict]:
    warnings_list = process_warnings(bundle, df)
    
    preprocessor = bundle['preprocessor']
    model = bundle['model']
    
    X_transformed = preprocessor.transform(df)
    
    feature_cols = bundle['feature_columns']
    if isinstance(X_transformed, pd.DataFrame):
        try:
            X_final = X_transformed[feature_cols]
        except KeyError:
            X_final = X_transformed
    else:
        X_final = X_transformed
        
    preds = model.predict(X_final)
    probs = [None] * len(preds)
    
    if task_type.lower() == 'classification' and hasattr(model, 'predict_proba'):
        all_probs = model.predict_proba(X_final)
        for i, p in enumerate(all_probs):
            if len(p) == 2:
                probs[i] = float(p[1])
            else:
                probs[i] = float(max(p))
                
    results = []
    for i in range(len(preds)):
        res = {
            "row_index": i,
            "prediction": float(preds[i]) if isinstance(preds[i], (np.floating, float)) else preds[i],
            "warnings": warnings_list[i]
        }
        if probs[i] is not None:
            res["probability"] = probs[i]
        results.append(res)
        
    return results
