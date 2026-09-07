import pandas as pd
import numpy as np
from typing import Dict, Any

def detect_missing(df: pd.DataFrame, col: str, col_meta: dict) -> Dict[str, Any]:
    """
    Detect missing values and calculate confidence for automated imputation/dropping.
    
    Dissertation Methods Reasoning:
    The confidence of automating missing value resolution hinges primarily on prior semantic knowledge.
    If the user explicitly declared the meaning of nulls (e.g. 'zero_is_valid' or 'skipped') during 
    the Column Specification phase, the system has 100% (1.0) confidence to apply that transformation.
    If missingness is extremely low (<1%), it is statistically safe to drop rows (confidence 0.9).
    Otherwise, if no semantic meaning was provided, missing value imputation is inherently ambiguous 
    and requires domain judgment, resulting in low confidence (0.3) to trigger a human question.
    """
    null_count = df[col].isnull().sum()
    if null_count == 0:
        return None
        
    null_pct = null_count / len(df)
    missing_meaning = col_meta.get("missing_meaning")
    
    evidence = {"null_count": int(null_count), "null_pct": float(null_pct)}
    
    # 1. High confidence via semantic metadata
    if missing_meaning == 'zero_is_valid':
        return {
            "issue": "missing_values",
            "confidence": 1.0,
            "evidence": evidence,
            "proposed_fix": "fill_zero"
        }
    elif missing_meaning in ('skipped', 'not_applicable'):
        return {
            "issue": "missing_values",
            "confidence": 1.0,
            "evidence": evidence,
            "proposed_fix": "drop_rows"
        }
        
    # 2. High confidence via statistical insignificance
    if null_pct < 0.01:
        return {
            "issue": "missing_values",
            "confidence": 0.9,
            "evidence": evidence,
            "proposed_fix": "drop_rows"
        }
        
    # 3. Low confidence (Ambiguous)
    return {
        "issue": "missing_values",
        "confidence": 0.3,
        "evidence": evidence,
        "proposed_fix": "impute_median" if pd.api.types.is_numeric_dtype(df[col]) else "impute_mode"
    }

def detect_outliers(df: pd.DataFrame, col: str, col_meta: dict) -> Dict[str, Any]:
    """
    Detect outliers using ensemble consensus (Z-Score and IQR).
    
    Dissertation Methods Reasoning:
    Automated outlier removal is dangerous because genuine extreme values can be mistaken for errors.
    This engine uses two disparate statistical methods: Z-score (parametric, assumes normality) 
    and IQR (non-parametric, robust to skew). 
    - If both methods identify the EXACT SAME points as outliers, the confidence is high (0.85) 
      that these are true anomalies.
    - If the methods disagree (one flags points the other misses), it signals distributional ambiguity,
      dropping the confidence significantly (0.4) to mandate human review.
    """
    if not pd.api.types.is_numeric_dtype(df[col]):
        return None
        
    s = df[col].dropna()
    if len(s) < 10:
        return None
        
    # Z-Score
    mean = s.mean()
    std = s.std()
    if std == 0:
        return None
    z_scores = (s - mean) / std
    z_outliers = set(s[abs(z_scores) > 3].index)
    
    # IQR
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = set(s[(s < (Q1 - 1.5 * IQR)) | (s > (Q3 + 1.5 * IQR))].index)
    
    if not z_outliers and not iqr_outliers:
        return None
        
    # Confidence calculation based on consensus
    union_count = len(z_outliers.union(iqr_outliers))
    intersection_count = len(z_outliers.intersection(iqr_outliers))
    
    if union_count == 0:
        return None
        
    agreement_ratio = intersection_count / union_count
    
    # If perfect agreement and small % of data, very high confidence
    pct_affected = union_count / len(df)
    if agreement_ratio > 0.9 and pct_affected < 0.05:
        confidence = 0.88
    else:
        confidence = 0.4 + (agreement_ratio * 0.3) # Max 0.7 if they agree but large % affected
        
    evidence = {
        "z_outliers": len(z_outliers), 
        "iqr_outliers": len(iqr_outliers),
        "intersection": intersection_count
    }
    
    return {
        "issue": "outliers",
        "confidence": confidence,
        "evidence": evidence,
        "proposed_fix": "cap_iqr" if confidence > 0.5 else "drop"
    }

def detect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect exact row duplicates.
    
    Dissertation Methods Reasoning:
    Exact row duplicates usually represent data entry errors or join artifacts.
    However, in highly categorical or low-dimensionality datasets, duplicates might be mathematically valid.
    The confidence is inversely proportional to the percentage of the dataset affected.
    If duplicates constitute <5% of the data, we are highly confident (0.95) they are errors to drop.
    If they constitute >20%, this is likely a structural feature of the dataset, lowering confidence (0.5) 
    and requiring human confirmation before aggressively dropping data.
    """
    dup_count = df.duplicated().sum()
    if dup_count == 0:
        return None
        
    dup_pct = dup_count / len(df)
    confidence = 0.95 if dup_pct < 0.05 else (0.8 if dup_pct < 0.1 else 0.5)
    
    return {
        "issue": "duplicate_rows",
        "confidence": confidence,
        "evidence": {"dup_count": int(dup_count), "dup_pct": float(dup_pct)},
        "proposed_fix": "drop_duplicates"
    }

def detect_format_inconsistency(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """
    Detect format inconsistencies (e.g., mixed casing in categorical columns).
    
    Dissertation Methods Reasoning:
    Categorical columns often suffer from slight data entry inconsistencies (e.g., 'New York' vs 'new york').
    If lowercasing the entire column collapses the number of unique categorical values, an inconsistency exists.
    If the majority format represents >90% of the entries, we have high confidence (0.9) to coerce the minority 
    to the majority casing. If the split is more ambiguous, human judgment is required.
    """
    if not pd.api.types.is_object_dtype(df[col]):
        return None
        
    s = df[col].dropna().astype(str)
    if len(s) == 0:
        return None
        
    unique_raw = s.nunique()
    unique_lower = s.str.lower().nunique()
    
    if unique_raw > unique_lower:
        # We found casing inconsistencies
        value_counts = s.value_counts()
        majority_pct = value_counts.iloc[0] / len(s)
        
        confidence = 0.9 if majority_pct > 0.8 else 0.6
        
        return {
            "issue": "mixed_casing",
            "confidence": confidence,
            "evidence": {"raw_unique": int(unique_raw), "lower_unique": int(unique_lower)},
            "proposed_fix": "lowercase_all"
        }
        
    return None
