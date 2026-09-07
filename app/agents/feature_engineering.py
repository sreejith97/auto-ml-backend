import pandas as pd
import numpy as np

def propose_transform(col_name: str, col_meta: dict, series: pd.Series) -> dict:
    role = col_meta.get("role", "feature")
    
    # Skip id, ignore, target columns entirely
    if role in ['id', 'ignore', 'target']:
        return None
        
    semantic_type = col_meta.get("semantic_type", "unknown")
    
    proposal = {
        "column_name": col_name,
        "semantic_type": semantic_type,
        "proposed_transform": "unknown",
        "params": {},
        "rationale": ""
    }
    
    # Text / Free-text
    if semantic_type in ['text', 'free_text', 'free-text']:
        proposal["proposed_transform"] = "manual_handling"
        proposal["rationale"] = "Text columns require manual NLP handling (e.g., TF-IDF or Embeddings) which is not auto-proposed in this phase."
        return proposal
        
    # Boolean
    if semantic_type == 'boolean' or series.dtype == 'bool' or set(series.dropna().unique()) <= {True, False, 0, 1}:
        proposal["proposed_transform"] = "passthrough"
        proposal["rationale"] = "Boolean columns are already model-ready and do not require transformation."
        proposal["semantic_type"] = "boolean" # Ensure it's labeled correctly
        return proposal
        
    # Datetime
    if semantic_type == 'datetime' or pd.api.types.is_datetime64_any_dtype(series):
        proposal["proposed_transform"] = "datetime_decompose"
        proposal["rationale"] = "Datetime column will be decomposed into year, month, day, and day_of_week to allow the model to learn temporal patterns."
        return proposal
        
    # Categorical
    if semantic_type == 'categorical' or series.dtype == 'object':
        nunique = series.nunique()
        if nunique <= 10:
            proposal["proposed_transform"] = "one_hot_encode"
            proposal["rationale"] = f"Column has low cardinality ({nunique} <= 10), making OneHotEncoding the optimal choice."
        else:
            proposal["proposed_transform"] = "ordinal_encode"
            proposal["rationale"] = f"Column has high cardinality ({nunique} > 10). Ordinal encoding is proposed to avoid creating too many sparse columns."
        return proposal
        
    # Numeric
    if semantic_type == 'numeric' or pd.api.types.is_numeric_dtype(series):
        # Calculate skewness safely
        skew = 0.0
        try:
            skew = series.skew()
        except:
            pass
            
        if pd.notna(skew) and abs(skew) > 1.0:
            if (series <= 0).any():
                proposal["proposed_transform"] = "standard_scale"
                proposal["rationale"] = f"Column exhibits high skew (skew = {skew:.2f}), but contains non-positive values where log1p is undefined. Falling back to standard scaling only."
            else:
                proposal["proposed_transform"] = "log1p_standard_scale"
                proposal["rationale"] = f"Column exhibits high skew (skew = {skew:.2f}). Proposing a log1p transform followed by standard scaling."
        else:
            proposal["proposed_transform"] = "standard_scale"
            proposal["rationale"] = "Numeric column with normal variance. Standard scaling is proposed to center the data."
        return proposal
        
    # Fallback
    proposal["proposed_transform"] = "passthrough"
    proposal["rationale"] = "Could not confidently determine a transformation. Passing through."
    return proposal
