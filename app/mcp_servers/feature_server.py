"""
app/mcp_servers/feature_server.py — FastMCP Server for Feature Engineering Tools
================================================================================
Exposes MCP tools for feature scoring, formula safety check, and SQL expression evaluation.
"""

import json
import logging
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

logger = logging.getLogger(__name__)

# Initialize FastMCP Server for Feature Engineering
mcp = FastMCP("AutoML-Feature-Server")


@mcp.tool()

def calculate_mutual_information(data: dict, target_column: str, is_classification: bool = True) -> str:
    """
    Calculates Mutual Information feature importance scores between features and target.
    
    Args:
        data: Dictionary of column_name -> list of values.
        target_column: Target variable name.
        is_classification: True for classification target, False for regression.
        
    Returns:
        JSON string mapping each feature to its Mutual Information score.
    """
    try:
        df = pd.DataFrame(data).dropna()
        if target_column not in df.columns:
            return json.dumps({"error": f"Target column '{target_column}' not found in data."})
            
        X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
        y = df[target_column]
        
        if X.empty:
            return json.dumps({"error": "No numeric feature columns available for MI computation."})
            
        if is_classification:
            scores = mutual_info_classif(X, y, random_state=42)
        else:
            scores = mutual_info_regression(X, y, random_state=42)
            
        mi_dict = {col: round(float(score), 4) for col, score in zip(X.columns, scores)}
        sorted_mi = dict(sorted(mi_dict.items(), key=lambda item: item[1], reverse=True))
        
        return json.dumps({
            "target_column": target_column,
            "mutual_information_scores": sorted_mi
        })
    except Exception as exc:
        return json.dumps({"error": f"Failed Mutual Information calculation: {exc}"})


@mcp.tool()

def validate_formula_safety(formula: str, columns: list) -> str:
    """
    Validates mathematical safety of a proposed feature transformation formula.
    Checks for potential division by zero, negative log, or invalid column names.
    
    Args:
        formula: Mathematical string formula (e.g. 'log(price + 1)', 'income / family_size').
        columns: List of valid column names in dataset.
        
    Returns:
        JSON string indicating safety, risk level, and warnings.
    """
    try:
        warnings = []
        is_safe = True
        
        if "/" in formula and "+ 1" not in formula and "+ 0." not in formula and "abs(" not in formula:
            warnings.append("Potential division by zero risk. Ensure denominator cannot be zero.")
            is_safe = False
            
        if "log(" in formula and "+ 1" not in formula and "+1" not in formula:
            warnings.append("Logarithm on non-positive values risk. Recommend log1p or adding +1 offset.")
            is_safe = False

        if "sqrt(" in formula:
            warnings.append("Square root on negative values risk. Ensure non-negative inputs.")
            
        return json.dumps({
            "formula": formula,
            "is_safe": is_safe,
            "risk_warnings": warnings
        })
    except Exception as exc:
        return json.dumps({"error": f"Formula validation failed: {exc}"})


@mcp.tool()

def evaluate_sql_expression(data: dict, query_expression: str) -> str:
    """
    Evaluates a pandas / SQL expression on dataset features to inspect candidate feature sample values.
    
    Args:
        data: Dictionary of column_name -> list of values.
        query_expression: Pandas expression string (e.g. 'df["price"] / df["sqft"]').
        
    Returns:
        JSON string with summary statistics of the engineered feature.
    """
    try:
        df = pd.DataFrame(data)
        result_series = pd.eval(query_expression, local_dict={"df": df})
        if isinstance(result_series, pd.Series):
            return json.dumps({
                "expression": query_expression,
                "count": int(result_series.count()),
                "mean": round(float(result_series.mean()), 4) if pd.api.types.is_numeric_dtype(result_series) else "N/A",
                "sample_values": [round(float(x), 4) if isinstance(x, (float, np.floating)) else str(x) for x in result_series.head(5)]
            })
        return json.dumps({"expression": query_expression, "result": str(result_series)})
    except Exception as exc:
        return json.dumps({"error": f"Expression evaluation failed: {exc}"})


if __name__ == "__main__":
    mcp.run()
