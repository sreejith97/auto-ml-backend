"""
app/mcp_servers/eda_server.py — FastMCP Server for EDA Tools
============================================================
Exposes MCP tools for statistical profiling, correlation, and drift analysis.
"""

import json
import logging
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Initialize FastMCP Server for EDA
mcp = FastMCP("AutoML-EDA-Server")


@mcp.tool()

def compute_descriptive_stats(column_name: str, values: list) -> str:
    """
    Computes comprehensive descriptive statistics for a dataset column.
    
    Args:
        column_name: Name of the column.
        values: List of data values for the column.
        
    Returns:
        JSON formatted string containing summary stats (mean, std, median, min, max, skewness, null_count).
    """
    try:
        s = pd.Series(values)
        if pd.api.types.is_numeric_dtype(s):
            stats = {
                "column_name": column_name,
                "type": "numeric",
                "count": int(s.count()),
                "null_count": int(s.isnull().sum()),
                "mean": float(s.mean()) if not s.empty else 0.0,
                "std": float(s.std()) if len(s) > 1 else 0.0,
                "median": float(s.median()) if not s.empty else 0.0,
                "min": float(s.min()) if not s.empty else 0.0,
                "max": float(s.max()) if not s.empty else 0.0,
                "skewness": float(s.skew()) if len(s) > 2 else 0.0,
            }
        else:
            vc = s.value_counts().head(5).to_dict()
            stats = {
                "column_name": column_name,
                "type": "categorical",
                "count": int(s.count()),
                "null_count": int(s.isnull().sum()),
                "unique_count": int(s.nunique()),
                "top_categories": {str(k): int(v) for k, v in vc.items()},
            }
        return json.dumps(stats)
    except Exception as exc:
        return json.dumps({"error": f"Failed to compute stats for {column_name}: {exc}"})


@mcp.tool()

def compute_correlation_matrix(numeric_data: dict, method: str = "pearson") -> str:
    """
    Computes pairwise correlation matrix for numeric columns.
    
    Args:
        numeric_data: Dictionary of column_name -> list of float values.
        method: 'pearson' or 'spearman'.
        
    Returns:
        JSON matrix of correlations.
    """
    try:
        df = pd.DataFrame(numeric_data)
        corr = df.corr(method=method).round(4).to_dict()
        return json.dumps({"method": method, "correlations": corr})
    except Exception as exc:
        return json.dumps({"error": f"Failed to compute correlation matrix: {exc}"})


@mcp.tool()

def analyze_missingness_patterns(data: dict) -> str:
    """
    Analyzes missing value co-occurrences across columns.
    
    Args:
        data: Dictionary of column_name -> list of values.
        
    Returns:
        JSON summary of missingness percentages and co-missing columns.
    """
    try:
        df = pd.DataFrame(data)
        null_counts = df.isnull().sum()
        total = len(df)
        missing_summary = {
            col: {"null_count": int(cnt), "null_pct": round(cnt / total * 100, 2)}
            for col, cnt in null_counts.items()
            if cnt > 0
        }
        return json.dumps({"total_rows": total, "missing_columns": missing_summary})
    except Exception as exc:
        return json.dumps({"error": f"Failed to analyze missingness: {exc}"})


@mcp.tool()
def analyze_target_distribution(target_column: str, values: list, task_type: str) -> str:
    """
    Analyzes target variable distribution and generates resolution recommendations.
    
    Args:
        target_column: Name of target column.
        values: List of target values.
        task_type: 'classification' or 'regression'.
    """
    try:
        df = pd.DataFrame({target_column: values})
        from app.agents.eda import compute_target_analysis
        result = compute_target_analysis(df, target_column, task_type)
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": f"Failed to analyze target distribution: {exc}"})


@mcp.tool()
def detect_outliers_and_skew(column_name: str, values: list) -> str:
    """
    Detects extreme outliers (Z-score > 3, IQR bounds) and distribution skewness.
    
    Args:
        column_name: Column to analyze.
        values: List of numeric values.
    """
    try:
        df = pd.DataFrame({column_name: values})
        cols_spec = [{"column_name": column_name, "role": "feature"}]
        from app.agents.eda import compute_outlier_skew_analysis
        result = compute_outlier_skew_analysis(df, cols_spec)
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": f"Failed to detect outliers/skew for {column_name}: {exc}"})


if __name__ == "__main__":
    mcp.run()

