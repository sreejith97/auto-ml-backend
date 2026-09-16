"""
app/mcp_servers/training_server.py — FastMCP Server for Training & Diagnostics Tools
=====================================================================================
Exposes MCP tools for computing ROC/PR/Accuracy/RMSE metrics, hyperparameter space recommendation, and model interpretability.
"""

import json
import logging
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

# Initialize FastMCP Server for Training & Diagnostics
mcp = FastMCP("AutoML-Training-Server")


@mcp.tool()

def compute_model_diagnostics(y_true: list, y_pred: list, task_type: str = "classification") -> str:
    """
    Computes comprehensive evaluation metrics for classification or regression predictions.
    
    Args:
        y_true: Ground truth target values.
        y_pred: Model predicted target values.
        task_type: 'classification' or 'regression'.
        
    Returns:
        JSON formatted dictionary of diagnostic metrics.
    """
    try:
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        
        if task_type.lower() == "classification":
            metrics = {
                "task_type": "classification",
                "accuracy": round(float(accuracy_score(y_true_arr, y_pred_arr)), 4),
                "precision_macro": round(float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)), 4),
                "recall_macro": round(float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)), 4),
                "f1_macro": round(float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)), 4),
            }
        else:
            mse = float(mean_squared_error(y_true_arr, y_pred_arr))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
            r2 = float(r2_score(y_true_arr, y_pred_arr))
            metrics = {
                "task_type": "regression",
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "mse": round(mse, 4),
                "r2_score": round(r2, 4),
            }
        return json.dumps(metrics)
    except Exception as exc:
        return json.dumps({"error": f"Failed model diagnostics: {exc}"})


@mcp.tool()

def recommend_hyperparameters(task_type: str, n_rows: int, n_cols: int) -> str:
    """
    Recommends optimal hyperparameter search grid based on dataset dimensions.
    
    Args:
        task_type: 'classification' or 'regression'.
        n_rows: Number of rows in training dataset.
        n_cols: Number of feature columns.
        
    Returns:
        JSON string detailing recommended hyperparameter search bounds for Random Forest and XGBoost/LightGBM.
    """
    try:
        is_large = n_rows > 50000
        recs = {
            "random_forest": {
                "n_estimators": [100, 200] if is_large else [100, 200, 300],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
            },
            "gradient_boosting": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 6, 9],
                "subsample": [0.8, 1.0],
            },
            "recommendation_notes": (
                "For large datasets (>50k rows), capped n_estimators to avoid memory bottleneck."
                if is_large
                else "Full hyperparameter space recommended."
            )
        }
        return json.dumps(recs)
    except Exception as exc:
        return json.dumps({"error": f"Failed hyperparameter recommendation: {exc}"})


if __name__ == "__main__":
    mcp.run()
