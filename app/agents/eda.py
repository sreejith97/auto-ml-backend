import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

def compute_summary_stats(df: pd.DataFrame, columns_spec: List[Dict]) -> Dict[str, Any]:
    """Computes deterministic summary statistics per column."""
    stats = {}
    
    for col_meta in columns_spec:
        col = col_meta['column_name']
        role = col_meta['role']
        
        if role in ('id', 'ignore') or col not in df.columns:
            continue
            
        s = df[col]
        col_stats = {"role": role, "missing": int(s.isnull().sum())}
        
        if pd.api.types.is_numeric_dtype(s):
            col_stats.update({
                "type": "numeric",
                "mean": float(s.mean()) if not pd.isna(s.mean()) else None,
                "median": float(s.median()) if not pd.isna(s.median()) else None,
                "std": float(s.std()) if not pd.isna(s.std()) else None,
                "min": float(s.min()) if not pd.isna(s.min()) else None,
                "max": float(s.max()) if not pd.isna(s.max()) else None
            })
        elif pd.api.types.is_datetime64_any_dtype(s):
            col_stats.update({
                "type": "datetime",
                "min": str(s.min()),
                "max": str(s.max())
            })
        else:
            vc = s.value_counts().head(5)
            col_stats.update({
                "type": "categorical",
                "unique_count": int(s.nunique()),
                "top_categories": vc.to_dict()
            })
            
        stats[col] = col_stats
        
    return stats

def compute_correlation_matrix(df: pd.DataFrame, columns_spec: List[Dict]) -> Dict[str, Any]:
    """Computes Pearson correlation matrix for numeric feature/target columns."""
    allowed_cols = [c['column_name'] for c in columns_spec if c['role'] in ('feature', 'target')]
    num_df = df[[c for c in allowed_cols if c in df.columns]].select_dtypes(include=[np.number])
    
    if num_df.empty:
        return {"columns": [], "matrix": []}
        
    corr = num_df.corr().round(3)
    
    # Replace NaNs with 0 for JSON serialization
    corr = corr.fillna(0)
    
    return {
        "columns": corr.columns.tolist(),
        "matrix": corr.values.tolist()
    }

def compute_class_balance(df: pd.DataFrame, target_column: str, task_type: str) -> Optional[Dict[str, Any]]:
    """Computes class balance only for classification tasks."""
    if task_type != 'classification' or target_column not in df.columns:
        return None
        
    s = df[target_column].dropna()
    vc = s.value_counts(normalize=True)
    raw_vc = s.value_counts()
    
    is_imbalanced = False
    if len(vc) > 1:
        majority = vc.max()
        minority = vc.min()
        if minority < 0.10: # Minority class is less than 10%
            is_imbalanced = True
            
    classes = [{"class": str(k), "count": int(raw_vc[k]), "pct": float(vc[k])} for k in vc.index]
    
    return {
        "classes": classes,
        "is_imbalanced": is_imbalanced
    }

def generate_findings_summary(stats: Dict[str, Any], correlation: Dict[str, Any], class_balance: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deterministic rule engine to surface 'notable' findings from the EDA data.
    These are passed to the LLM for natural language phrasing.
    """
    findings = []
    
    # 1. Class Imbalance
    if class_balance and class_balance.get("is_imbalanced"):
        minority = min(class_balance["classes"], key=lambda x: x["pct"])
        findings.append({
            "type": "class_imbalance",
            "detail": f"The dataset is highly imbalanced. The minority class '{minority['class']}' only represents {minority['pct']*100:.1f}% of the data."
        })
        
    # 2. High Correlations
    if correlation and correlation.get("columns"):
        cols = correlation["columns"]
        matrix = correlation["matrix"]
        high_corrs = []
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                val = matrix[i][j]
                if abs(val) > 0.75:
                    high_corrs.append((cols[i], cols[j], val))
                    
        if high_corrs:
            # Just take the highest one for brevity
            highest = max(high_corrs, key=lambda x: abs(x[2]))
            findings.append({
                "type": "high_correlation",
                "detail": f"Features '{highest[0]}' and '{highest[1]}' are highly correlated (r={highest[2]:.2f})."
            })
            
    # 3. High Skew / Outliers (approximated via mean/median delta)
    for col, c_stats in stats.items():
        if c_stats.get("type") == "numeric" and c_stats.get("mean") is not None and c_stats.get("std") is not None and c_stats["std"] > 0:
            skew_proxy = abs(c_stats["mean"] - c_stats["median"]) / c_stats["std"]
            if skew_proxy > 1.5:
                findings.append({
                    "type": "high_skew",
                    "detail": f"Column '{col}' exhibits significant skew or extreme values (mean={c_stats['mean']:.1f}, median={c_stats['median']:.1f})."
                })
                
    return findings
