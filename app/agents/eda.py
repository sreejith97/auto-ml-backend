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
                "top_categories": {str(k): int(v) for k, v in vc.to_dict().items()}
            })
            
        stats[col] = col_stats
        
    return stats

def compute_dataset_health(df: pd.DataFrame) -> Dict[str, Any]:
    """EDA Step 1: Overall dataset dimension, memory, duplicates, and missingness health."""
    total_rows = int(len(df))
    total_cols = int(len(df.columns))
    memory_mb = round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 2)
    duplicate_rows = int(df.duplicated().sum())
    total_cells = total_rows * total_cols if total_rows > 0 else 1
    total_nulls = int(df.isnull().sum().sum())
    missing_pct = round(float(total_nulls / total_cells) * 100, 2)

    metrics = {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "memory_mb": memory_mb,
        "duplicate_rows": duplicate_rows,
        "total_nulls": total_nulls,
        "missing_pct": missing_pct,
    }

    recs = []
    if duplicate_rows > 0:
        recs.append(f"Remove {duplicate_rows} duplicate rows using data deduplication.")
    if missing_pct > 15.0:
        recs.append(f"Overall missingness is high ({missing_pct}%). Consider column-level dropping or advanced imputation.")
    elif missing_pct > 0:
        recs.append("Impute remaining missing values before model training.")
    else:
        recs.append("Data completeness is 100%. No missing value remediation required.")

    findings = (
        f"Dataset contains {total_rows:,} rows across {total_cols} columns ({memory_mb} MB memory). "
        f"{duplicate_rows} duplicate rows detected, with {missing_pct}% overall missing values."
    )

    return {
        "metrics": metrics,
        "findings": findings,
        "recommendations": recs
    }

def compute_target_analysis(df: pd.DataFrame, target_column: str, task_type: str) -> Dict[str, Any]:
    """EDA Step 2: Target column distribution & task-specific profiling."""
    if not target_column or target_column not in df.columns:
        return {
            "metrics": {},
            "findings": "Target column not specified or missing from dataset.",
            "recommendations": []
        }

    s = df[target_column].dropna()
    recs = []

    if task_type == 'classification':
        vc = s.value_counts(normalize=True)
        raw_vc = s.value_counts()
        classes = [{"class": str(k), "count": int(raw_vc[k]), "pct": round(float(vc[k]) * 100, 2)} for k in vc.index]
        is_imbalanced = False
        if len(vc) > 1 and vc.min() < 0.10:
            is_imbalanced = True

        metrics = {
            "task_type": "classification",
            "target_column": target_column,
            "num_classes": len(classes),
            "classes": classes,
            "is_imbalanced": is_imbalanced
        }

        if is_imbalanced:
            minority = min(classes, key=lambda x: x["pct"])
            findings = f"Target '{target_column}' is imbalanced. Minority class '{minority['class']}' represents only {minority['pct']}% of records."
            recs.append("Apply class weighting (e.g. `class_weight='balanced'`) or SMOTE oversampling during training.")
            recs.append("Evaluate models using PR-AUC or F1-score instead of standard Accuracy.")
        else:
            findings = f"Target '{target_column}' is balanced across {len(classes)} classes."
            recs.append("Standard classification loss functions (Cross-Entropy/LogLoss) are suitable.")

    else: # regression
        num_s = pd.to_numeric(s, errors='coerce').dropna()
        mean_val = float(num_s.mean()) if not num_s.empty else 0.0
        median_val = float(num_s.median()) if not num_s.empty else 0.0
        std_val = float(num_s.std()) if len(num_s) > 1 else 0.0
        skew_val = float(num_s.skew()) if len(num_s) > 2 else 0.0
        min_val = float(num_s.min()) if not num_s.empty else 0.0
        max_val = float(num_s.max()) if not num_s.empty else 0.0

        metrics = {
            "task_type": "regression",
            "target_column": target_column,
            "mean": round(mean_val, 3),
            "median": round(median_val, 3),
            "std": round(std_val, 3),
            "skewness": round(skew_val, 3),
            "min": round(min_val, 3),
            "max": round(max_val, 3)
        }

        if abs(skew_val) > 1.0:
            findings = f"Target '{target_column}' exhibits significant skewness ({skew_val:.2f}) with mean={mean_val:.2f} vs median={median_val:.2f}."
            recs.append("Apply `log1p` or Box-Cox transformation to normalize the target distribution.")
            recs.append("Use MAE or MedAE as evaluation metrics to mitigate extreme value impact.")
        else:
            findings = f"Target '{target_column}' has a fairly symmetric numeric distribution (skewness = {skew_val:.2f})."
            recs.append("Standard RMSE and MSE regression metrics are appropriate.")

    return {
        "metrics": metrics,
        "findings": findings,
        "recommendations": recs
    }

def compute_outlier_skew_analysis(df: pd.DataFrame, columns_spec: List[Dict]) -> Dict[str, Any]:
    """EDA Step 3: Outliers (IQR / Z-score) and skewness across numeric features."""
    feature_cols = [c['column_name'] for c in columns_spec if c['role'] in ('feature', 'target')]
    numeric_cols = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    skewed_cols = []
    outlier_summary = []
    total_outliers = 0
    recs = []

    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 3:
            continue
        mean_val = s.mean()
        std_val = s.std()
        skew_val = s.skew()

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        iqr_outliers = int(((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum())
        z_outliers = int((abs((s - mean_val) / std_val) > 3.0).sum()) if std_val > 0 else 0

        if abs(skew_val) > 1.2 or z_outliers > 0:
            outlier_summary.append({
                "column": col,
                "skewness": round(float(skew_val), 2),
                "iqr_outliers": iqr_outliers,
                "z_outliers": z_outliers,
            })
            total_outliers += z_outliers
            if abs(skew_val) > 1.2:
                skewed_cols.append(col)

    metrics = {
        "outlier_columns_count": len(outlier_summary),
        "total_z_outliers": total_outliers,
        "skewed_features": skewed_cols,
        "details": outlier_summary[:10]
    }

    if skewed_cols:
        recs.append(f"Apply log or quantile scaling on heavily skewed features: {', '.join(skewed_cols[:4])}.")
    if total_outliers > 0:
        recs.append(f"Cap or clip {total_outliers} extreme outliers beyond 3 standard deviations using RobustScaler or Winsorization.")
    if not recs:
        recs.append("Numeric feature distributions are well-behaved. No extreme outlier clipping required.")

    findings = (
        f"Identified {len(outlier_summary)} numeric feature(s) with noticeable skewness or extreme values, "
        f"totaling {total_outliers} z-score extreme outliers."
    )

    return {
        "metrics": metrics,
        "findings": findings,
        "recommendations": recs
    }

def compute_feature_target_relationships(df: pd.DataFrame, target_column: str, task_type: str, columns_spec: List[Dict]) -> Dict[str, Any]:
    """EDA Step 4: Association ranking between features and the target variable."""
    if not target_column or target_column not in df.columns:
        return {
            "metrics": {"rankings": []},
            "findings": "Target column unavailable for feature-target correlation.",
            "recommendations": []
        }

    feature_cols = [c['column_name'] for c in columns_spec if c['role'] == 'feature' and c['column_name'] in df.columns]
    rankings = []
    recs = []

    target_s = df[target_column]
    is_target_num = pd.api.types.is_numeric_dtype(target_s)

    for col in feature_cols:
        s = df[col]
        if s.dropna().empty:
            continue

        if pd.api.types.is_numeric_dtype(s) and is_target_num:
            corr_val = df[[col, target_column]].dropna().corr().iloc[0, 1]
            if not pd.isna(corr_val):
                rankings.append({
                    "feature": col,
                    "type": "numeric-numeric",
                    "score": round(float(abs(corr_val)), 3),
                    "raw_score": round(float(corr_val), 3)
                })
        elif pd.api.types.is_numeric_dtype(s) and task_type == 'classification':
            # Group difference score
            try:
                grouped = df.groupby(target_column)[col].mean()
                if len(grouped) > 1:
                    overall_std = s.std()
                    diff = (grouped.max() - grouped.min()) / overall_std if overall_std > 0 else 0
                    rankings.append({
                        "feature": col,
                        "type": "numeric-categorical",
                        "score": round(float(diff), 3),
                        "raw_score": round(float(diff), 3)
                    })
            except Exception:
                pass

    rankings.sort(key=lambda x: x["score"], reverse=True)
    top_features = [r["feature"] for r in rankings[:5]]
    potential_leakage = [r["feature"] for r in rankings if r["score"] > 0.92]

    metrics = {
        "rankings": rankings[:10],
        "top_features": top_features,
        "potential_leakage": potential_leakage
    }

    if potential_leakage:
        findings = f"Caution: Features {potential_leakage} show extremely high correlation (>0.92) with target '{target_column}', indicating potential data leakage."
        recs.append(f"Inspect and drop leakage candidate columns: {', '.join(potential_leakage)}.")
    elif top_features:
        findings = f"Top predictive feature drivers for target '{target_column}' are: {', '.join(top_features)}."
        recs.append("Prioritize top features in downstream feature engineering and model explainability.")
    else:
        findings = "No strong linear or group-mean relationships detected with target."
        recs.append("Consider non-linear models (RandomForest, XGBoost) or creating feature interaction terms.")

    return {
        "metrics": metrics,
        "findings": findings,
        "recommendations": recs
    }

def compute_categorical_profiling(df: pd.DataFrame, columns_spec: List[Dict]) -> Dict[str, Any]:
    """EDA Step 5: High-cardinality and rare category profiling."""
    cat_cols = [c['column_name'] for c in columns_spec if c['role'] in ('feature', 'target') and c['column_name'] in df.columns and not pd.api.types.is_numeric_dtype(df[c['column_name']])]
    
    high_cardinality = []
    rare_categories = []
    recs = []

    for col in cat_cols:
        s = df[col].dropna()
        n_unique = s.nunique()
        total_len = len(s)
        if total_len == 0:
            continue

        if n_unique > 30 or (n_unique / total_len > 0.20 and n_unique > 10):
            high_cardinality.append({"column": col, "unique_count": int(n_unique)})

        vc_pct = s.value_counts(normalize=True)
        rare_count = int((vc_pct < 0.01).sum())
        if rare_count > 0:
            rare_categories.append({"column": col, "rare_levels_count": rare_count})

    metrics = {
        "categorical_columns_count": len(cat_cols),
        "high_cardinality_features": high_cardinality,
        "rare_categories": rare_categories
    }

    if high_cardinality:
        cols_str = ", ".join([h["column"] for h in high_cardinality])
        recs.append(f"High-cardinality features detected ({cols_str}). Use Target Encoding or Frequency Encoding instead of One-Hot Encoding.")
    if rare_categories:
        recs.append("Group rare categories (<1% frequency) into a single `'Other'` category to prevent model overfitting.")
    if not recs:
        recs.append("Categorical feature cardinalities are manageable for standard encoding (One-Hot / Ordinal).")

    findings = (
        f"Analyzed {len(cat_cols)} categorical feature(s). "
        f"Found {len(high_cardinality)} high-cardinality feature(s) and {len(rare_categories)} column(s) with rare category levels."
    )

    return {
        "metrics": metrics,
        "findings": findings,
        "recommendations": recs
    }

def compute_correlation_matrix(df: pd.DataFrame, columns_spec: List[Dict]) -> Dict[str, Any]:
    """EDA Step 6: Pairwise feature-feature Pearson correlation matrix and multicollinearity."""
    allowed_cols = [c['column_name'] for c in columns_spec if c['role'] in ('feature', 'target')]
    num_df = df[[c for c in allowed_cols if c in df.columns]].select_dtypes(include=[np.number])
    
    if num_df.empty:
        return {
            "columns": [],
            "matrix": [],
            "findings": "No numerical features available for correlation analysis.",
            "recommendations": []
        }
        
    corr = num_df.corr().round(3).fillna(0)
    cols = corr.columns.tolist()
    matrix = corr.values.tolist()

    high_corrs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = matrix[i][j]
            if abs(val) > 0.75:
                high_corrs.append((cols[i], cols[j], val))

    recs = []
    if high_corrs:
        highest = max(high_corrs, key=lambda x: abs(x[2]))
        findings = f"Multicollinearity warning: {len(high_corrs)} feature pair(s) show strong correlation (|r| > 0.75), highest being '{highest[0]}' & '{highest[1]}' (r={highest[2]:.2f})."
        recs.append(f"Prune one of the highly correlated feature pairs or apply PCA to mitigate multicollinearity.")
    else:
        findings = "Low feature-to-feature correlation detected across all numerical columns."
        recs.append("No feature pruning required for correlation. All numerical features retain unique signals.")

    return {
        "columns": cols,
        "matrix": matrix,
        "findings": findings,
        "recommendations": recs
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

def generate_findings_summary(
    stats: Dict[str, Any],
    correlation: Dict[str, Any],
    class_balance: Optional[Dict[str, Any]],
    dataset_health: Optional[Dict[str, Any]] = None,
    target_analysis: Optional[Dict[str, Any]] = None,
    outlier_analysis: Optional[Dict[str, Any]] = None,
    feature_target_relations: Optional[Dict[str, Any]] = None,
    categorical_profiling: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Aggregates all findings across all 6 EDA steps into a structured list."""
    findings = []

    if dataset_health and dataset_health.get("findings"):
        findings.append({
            "type": "dataset_health",
            "detail": dataset_health["findings"],
            "recommendations": dataset_health.get("recommendations", [])
        })

    if target_analysis and target_analysis.get("findings"):
        findings.append({
            "type": "target_analysis",
            "detail": target_analysis["findings"],
            "recommendations": target_analysis.get("recommendations", [])
        })

    if outlier_analysis and outlier_analysis.get("findings"):
        findings.append({
            "type": "outlier_analysis",
            "detail": outlier_analysis["findings"],
            "recommendations": outlier_analysis.get("recommendations", [])
        })

    if feature_target_relations and feature_target_relations.get("findings"):
        findings.append({
            "type": "feature_target_relations",
            "detail": feature_target_relations["findings"],
            "recommendations": feature_target_relations.get("recommendations", [])
        })

    if correlation and correlation.get("findings"):
        findings.append({
            "type": "multicollinearity",
            "detail": correlation["findings"],
            "recommendations": correlation.get("recommendations", [])
        })

    if categorical_profiling and categorical_profiling.get("findings"):
        findings.append({
            "type": "categorical_profiling",
            "detail": categorical_profiling["findings"],
            "recommendations": categorical_profiling.get("recommendations", [])
        })

    return findings
