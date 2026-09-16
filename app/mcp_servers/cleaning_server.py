"""
app/mcp_servers/cleaning_server.py — FastMCP Server for Data Cleaning Tools
========================================================================
Exposes MCP tools for dirty data inspection, imputation preview, and outlier bounds.
"""

import json
import logging
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Initialize FastMCP Server for Data Cleaning
mcp = FastMCP("AutoML-Cleaning-Server")


@mcp.tool()

def inspect_dirty_values(column_name: str, values: list) -> str:
    """
    Inspects non-standard strings, mixed types, or unparseable values in a column.
    
    Args:
        column_name: Name of the column to inspect.
        values: List of column values.
        
    Returns:
        JSON string detailing dirty value patterns and affected row counts.
    """
    try:
        s = pd.Series(values)
        dirty_patterns = []
        
        # Check for mixed types or non-numeric strings in numeric columns
        numeric_s = pd.to_numeric(s, errors="coerce")
        coerced_nulls = numeric_s.isnull().sum() - s.isnull().sum()
        
        if coerced_nulls > 0:
            unparseable = s[numeric_s.isnull() & s.notnull()].unique()[:5].tolist()
            dirty_patterns.append({
                "issue": "mixed_types",
                "coerced_nulls": int(coerced_nulls),
                "samples": [str(x) for x in unparseable]
            })

        # Check for empty strings or whitespace
        empty_str_cnt = s.apply(lambda x: isinstance(x, str) and str(x).strip() == "").sum()
        if empty_str_cnt > 0:
            dirty_patterns.append({
                "issue": "empty_whitespace_strings",
                "count": int(empty_str_cnt)
            })

        return json.dumps({
            "column_name": column_name,
            "total_values": len(s),
            "dirty_patterns": dirty_patterns
        })
    except Exception as exc:
        return json.dumps({"error": f"Failed to inspect dirty values for {column_name}: {exc}"})


@mcp.tool()

def preview_imputation_impact(column_name: str, values: list, strategy: str = "median") -> str:
    """
    Simulates imputing missing values and reports impact on distribution metrics.
    
    Args:
        column_name: Name of the column.
        values: Column values containing missing entries.
        strategy: 'mean', 'median', 'mode', or 'constant_zero'.
        
    Returns:
        JSON string comparing pre and post imputation metrics.
    """
    try:
        s = pd.Series(values)
        num_s = pd.to_numeric(s, errors="coerce")
        null_count = int(num_s.isnull().sum())
        
        if null_count == 0:
            return json.dumps({"column_name": column_name, "message": "No missing values present."})
            
        pre_mean = float(num_s.mean()) if not num_s.empty else 0.0
        pre_std = float(num_s.std()) if len(num_s) > 1 else 0.0
        
        if strategy == "median":
            fill_val = float(num_s.median())
        elif strategy == "mean":
            fill_val = pre_mean
        elif strategy == "mode":
            fill_val = float(num_s.mode()[0]) if not num_s.mode().empty else 0.0
        else:
            fill_val = 0.0
            
        post_s = num_s.fillna(fill_val)
        post_mean = float(post_s.mean())
        post_std = float(post_s.std())
        
        return json.dumps({
            "column_name": column_name,
            "strategy": strategy,
            "null_count_filled": null_count,
            "fill_value": round(fill_val, 4),
            "metrics": {
                "pre_mean": round(pre_mean, 4),
                "post_mean": round(post_mean, 4),
                "pre_std": round(pre_std, 4),
                "post_std": round(post_std, 4),
            }
        })
    except Exception as exc:
        return json.dumps({"error": f"Failed preview for {column_name}: {exc}"})


@mcp.tool()

def preview_outlier_clipping(column_name: str, values: list, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> str:
    """
    Previews Winsorization / quantile clipping of extreme numeric outliers.
    
    Args:
        column_name: Name of the column.
        values: List of numeric values.
        lower_quantile: Lower threshold percentile (e.g. 0.01).
        upper_quantile: Upper threshold percentile (e.g. 0.99).
        
    Returns:
        JSON string reporting bounds and rows clipped.
    """
    try:
        s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
        lower_bound = float(s.quantile(lower_quantile))
        upper_bound = float(s.quantile(upper_quantile))
        
        clipped_low = int((s < lower_bound).sum())
        clipped_high = int((s > upper_bound).sum())
        
        return json.dumps({
            "column_name": column_name,
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "rows_clipped_lower": clipped_low,
            "rows_clipped_upper": clipped_high,
            "total_outliers": clipped_low + clipped_high
        })
    except Exception as exc:
        return json.dumps({"error": f"Failed outlier clipping preview for {column_name}: {exc}"})


if __name__ == "__main__":
    mcp.run()
