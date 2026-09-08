"""
app/workflows/steps/cleaning_step.py — Agno HITL Cleaning Analysis
====================================================================
The cleaning stage uses the Agno Workflow's HumanReview (HITL) pattern:

1. `cleaning_analysis_fn` runs all detectors (deterministic, no LLM)
2. Issues above CONFIDENCE_THRESHOLD are "auto_applied"
3. Issues below the threshold become "pending" — the workflow PAUSES via
   HumanReview and the frontend shows issue cards
4. The user resolves each card via the frontend
5. POST /clean/{run_id}/resume → workflow.continue_run() → pipeline continues

This replaces the `questions_pending` DB table + `chat.py` answer-parsing loop.

Usage:
    from app.workflows.steps.cleaning_step import (
        run_cleaning_analysis,
        apply_cleaning_decisions,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Confidence threshold: issues >= this are auto-applied, below triggers HITL
CONFIDENCE_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Core cleaning analysis — deterministic detectors
# ---------------------------------------------------------------------------

def run_cleaning_analysis(
    df: pd.DataFrame,
    columns_spec: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Runs all cleaning detectors and splits results into:
      - auto_applied: high-confidence issues fixed automatically
      - pending:      low-confidence issues requiring user decision (HITL)

    Returns (auto_applied, pending)
    """
    from app.agents.cleaning import (
        detect_missing,
        detect_outliers,
        detect_duplicates,
        detect_format_inconsistency,
    )
    from app.agents.confidence import apply_confidence_based_rule

    auto_applied: List[Dict] = []
    pending: List[Dict] = []

    # Per-column detectors
    for col_meta in columns_spec:
        col = col_meta.get("column_name")
        role = col_meta.get("role", "feature")

        if role in ("id", "ignore") or col not in df.columns:
            continue

        # Missing values
        result = detect_missing(df, col, col_meta)
        if result:
            entry = {**result, "column": col, "id": f"missing_{col}"}
            if result["confidence"] >= CONFIDENCE_THRESHOLD:
                entry["action"] = result["proposed_fix"]
                entry["source"] = "auto"
                auto_applied.append(entry)
            else:
                pending.append(entry)

        # Outliers (numeric only)
        result = detect_outliers(df, col, col_meta)
        if result:
            entry = {**result, "column": col, "id": f"outlier_{col}"}
            if result["confidence"] >= CONFIDENCE_THRESHOLD:
                entry["action"] = result["proposed_fix"]
                entry["source"] = "auto"
                auto_applied.append(entry)
            else:
                pending.append(entry)

        # Format inconsistencies (categorical only)
        result = detect_format_inconsistency(df, col)
        if result:
            entry = {**result, "column": col, "id": f"format_{col}"}
            if result["confidence"] >= CONFIDENCE_THRESHOLD:
                entry["action"] = result["proposed_fix"]
                entry["source"] = "auto"
                auto_applied.append(entry)
            else:
                pending.append(entry)

    # Dataset-level: duplicates
    result = detect_duplicates(df)
    if result:
        entry = {**result, "column": None, "id": "duplicate_rows"}
        if result["confidence"] >= CONFIDENCE_THRESHOLD:
            entry["action"] = result["proposed_fix"]
            entry["source"] = "auto"
            auto_applied.append(entry)
        else:
            pending.append(entry)

    return auto_applied, pending


# ---------------------------------------------------------------------------
# Apply cleaning decisions to the DataFrame
# ---------------------------------------------------------------------------

def apply_cleaning_actions(
    df: pd.DataFrame,
    actions: List[Dict],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Applies a list of cleaning actions to the dataframe.
    Returns (modified_df, summary_counts).

    Each action dict: {column, action, issue, ...}
    """
    rows_dropped = 0
    values_imputed = 0
    values_capped = 0

    for action in actions:
        col = action.get("column")
        op = action.get("action")

        if op == "drop_duplicates":
            before = len(df)
            df = df.drop_duplicates()
            rows_dropped += before - len(df)

        elif op == "drop_rows" and col and col in df.columns:
            before = len(df)
            df = df.dropna(subset=[col])
            rows_dropped += before - len(df)

        elif op == "impute_median" and col and col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                med = df[col].median()
                nulls = df[col].isnull().sum()
                df[col] = df[col].fillna(med)
                values_imputed += int(nulls)

        elif op == "impute_mean" and col and col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                mn = df[col].mean()
                nulls = df[col].isnull().sum()
                df[col] = df[col].fillna(mn)
                values_imputed += int(nulls)

        elif op == "fill_zero" and col and col in df.columns:
            nulls = df[col].isnull().sum()
            df[col] = df[col].fillna(0)
            values_imputed += int(nulls)

        elif op == "cap_iqr" and col and col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                mask = (df[col] < lower) | (df[col] > upper)
                df.loc[mask, col] = df[col].clip(lower, upper)
                values_capped += int(mask.sum())

        elif op == "drop" and col and col in df.columns:
            before = len(df)
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                rows_dropped += before - len(df)

        elif op == "lowercase_all" and col and col in df.columns:
            df[col] = df[col].astype(str).str.lower()
            values_imputed += len(df)

        elif op == "ignore":
            pass  # user chose to keep as-is

    return df, {
        "rows_dropped": rows_dropped,
        "values_imputed": values_imputed,
        "values_capped": values_capped,
    }
