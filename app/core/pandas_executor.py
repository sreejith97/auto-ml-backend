import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

def _df_to_markdown(sub_df: pd.DataFrame) -> str:
    """Formats DataFrame to Markdown table without tabulate dependency."""
    if sub_df.empty:
        return "*Empty Table*"
    headers = [str(c) for c in sub_df.columns]
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = [
        "| " + " | ".join([str(val) for val in row]) + " |"
        for _, row in sub_df.iterrows()
    ]
    return "\n".join([header_row, separator_row] + data_rows)

def _sanitize_pandas_code(code: str) -> str:
    """Strips safe pre-loaded imports (pd, np, math) and matplotlib calls."""
    clean_lines = []
    for line in code.split("\n"):
        s = line.strip()
        if s.startswith("import pandas") or s.startswith("import numpy") or s.startswith("import math"):
            continue
        if s.startswith("from pandas") or s.startswith("from numpy") or s.startswith("from math"):
            continue
        if s.startswith("plt.") or ".plot(" in s:
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)

def execute_pandas_operation(df: pd.DataFrame, code: str, is_mutation: bool = False) -> Tuple[pd.DataFrame, str, bool]:
    """
    Executes a pandas code snippet in a sandboxed, restricted environment.
    
    Args:
        df: Input pandas DataFrame.
        code: Python snippet operating on 'df'.
        is_mutation: True if operation modifies df structure or rows.
        
    Returns:
        (updated_df, output_markdown_table, success_flag)
    """
    if not code or not code.strip():
        return df, "", False

    code = _sanitize_pandas_code(code)

    # Copy df to prevent accidental in-place mutation if execution fails
    working_df = df.copy()

    # Sandboxed global/local scope — strictly restricting builtins
    safe_globals = {
        "pd": pd,
        "np": np,
        "__builtins__": {
            "len": len,
            "range": range,
            "list": list,
            "dict": dict,
            "set": set,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "round": round,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
        }
    }
    
    safe_locals = {
        "df": working_df,
        "result": None
    }

    try:
        # Prevent dangerous calls
        forbidden_keywords = ["__import__", "os.", "sys.", "subprocess", "open(", "eval(", "exec("]
        for kw in forbidden_keywords:
            if kw in code:
                raise ValueError(f"Security error: Forbidden keyword '{kw}' detected in pandas execution snippet.")


        # Execute in sandboxed scope
        exec(code, safe_globals, safe_locals)

        res = safe_locals.get("result")
        updated_df = safe_locals.get("df", working_df)

        if isinstance(res, dict) and "error" in res:
            return df, f"Error: {res['error']}", False

        output_md = ""
        if res is not None:
            if isinstance(res, pd.DataFrame):
                output_md = _df_to_markdown(res.head(10))
            elif isinstance(res, pd.Series):
                output_md = _df_to_markdown(res.head(10).to_frame())
            elif isinstance(res, (dict, list)):
                output_md = f"```json\n{res}\n```"
            else:
                output_md = f"**Result:** `{res}`"
        elif is_mutation and updated_df is not None:
            output_md = f"**Data Updated:** Applied operation successfully. New shape: `{updated_df.shape[0]} rows × {updated_df.shape[1]} columns`."

        return updated_df, output_md, True

    except Exception as exc:
        logger.error(f"Pandas execution failed for code [{code}]: {exc}")
        err_msg = f"⚠️ **Execution Error:** {exc}"
        return df, err_msg, False



def _format_chart_num(val: float) -> str:
    abs_val = abs(val)
    if abs_val >= 1_000_000:
        s = f"{val / 1_000_000:.1f}M"
        return s.replace(".0M", "M")
    elif abs_val >= 1_000:
        s = f"{val / 1_000:.1f}K"
        return s.replace(".0K", "K")
    elif val == int(val):
        return str(int(val))
    else:
        return f"{val:.1f}"

def _format_bin_range(start: float, end: float) -> str:
    return f"{_format_chart_num(start)} - {_format_chart_num(end)}"

def build_chart_spec(
    df: pd.DataFrame,
    chart_type: str,
    title: str,
    x_axis: str,
    y_axis: str
) -> Optional[Dict[str, Any]]:
    """
    Builds a structured Recharts payload (chart_type, title, x_axis, y_axis, data).
    """
    if df is None or df.empty:
        return None

    try:
        available_cols = df.columns.tolist()
        if not available_cols:
            return None

        # Strictly validate requested columns exist in dataset
        if x_axis and x_axis not in available_cols:
            logger.warning(f"Requested chart x_axis '{x_axis}' not found in columns: {available_cols}")
            return None

        if y_axis and chart_type not in ("histogram", "pie") and y_axis not in available_cols:
            logger.warning(f"Requested chart y_axis '{y_axis}' not found in columns: {available_cols}")
            return None

        x_col = x_axis if x_axis in available_cols else available_cols[0]
        y_col = y_axis if y_axis in available_cols else (available_cols[1] if len(available_cols) > 1 else available_cols[0])

        chart_data = []

        if chart_type == "histogram":
            s = pd.to_numeric(df[x_col], errors='coerce').dropna()
            if not s.empty:
                counts, bin_edges = np.histogram(s, bins=8)
                for i in range(len(counts)):
                    chart_data.append({
                        "bin": _format_bin_range(bin_edges[i], bin_edges[i+1]),
                        "count": int(counts[i])
                    })
                x_col = "bin"
                y_col = "count"

        elif chart_type in ("bar", "pie", "line"):
            # Group or take top 10
            if pd.api.types.is_numeric_dtype(df[y_col]):
                grouped = df.groupby(x_col)[y_col].mean().reset_index().head(10)
                for _, row in grouped.iterrows():
                    chart_data.append({
                        str(x_col): str(row[x_col]),
                        str(y_col): round(float(row[y_col]), 2)
                    })
            else:
                vc = df[x_col].value_counts().head(8).reset_index()
                vc.columns = [x_col, "count"]
                for _, row in vc.iterrows():
                    chart_data.append({
                        str(x_col): str(row[x_col]),
                        "count": int(row["count"])
                    })
                y_col = "count"

        else: # scatter
            sample_df = df[[x_col, y_col]].dropna().head(50)
            for _, row in sample_df.iterrows():
                try:
                    chart_data.append({
                        str(x_col): float(row[x_col]),
                        str(y_col): float(row[y_col])
                    })
                except ValueError:
                    pass

        if not chart_data:
            return None

        return {
            "chart_type": chart_type or "bar",
            "title": title or f"{chart_type.capitalize()} Plot of {y_col} by {x_col}",
            "x_axis": str(x_col),
            "y_axis": str(y_col),
            "data": chart_data
        }

    except Exception as exc:
        logger.error(f"Failed to build chart spec: {exc}")
        return None
