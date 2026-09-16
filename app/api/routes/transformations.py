from fastapi import APIRouter, Depends, HTTPException
import asyncpg
import logging
from typing import List, Optional
from pydantic import BaseModel

from app.core.db import get_db
from app.api.deps import get_current_user
from app.core.storage import load_dataframe, save_snapshot
from app.core.pandas_executor import execute_pandas_operation, build_chart_spec

import pandas as pd

logger = logging.getLogger(__name__)
router = APIRouter()

class TransformExecuteRequest(BaseModel):
    prompt: str
    custom_code: Optional[str] = None
    from_step_id: Optional[int] = None

class TransformStepResponse(BaseModel):
    step_id: int
    prompt: str
    code: str
    rows: int
    cols: int
    created_at: str

class TransformResponse(BaseModel):
    summary: str
    table_markdown: str
    rows: int
    cols: int
    chart_spec: Optional[dict] = None
    step: Optional[dict] = None

@router.get("/{run_id}/data")
async def get_transformations_data(
    run_id: int,
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Fetches current DataFrame sample, dimensions, and columns for the transformation studio."""
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    try:
        df = await _replay_transformations(db, run_id)
    except Exception as exc:
        logger.error(f"Failed to load dataset for run={run_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {exc}")

    sample_rows = df.head(15).to_dict(orient="records")
    # Clean NaN for JSON safety
    clean_sample = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in sample_rows
    ] if not df.empty else []

    return {
        "rows": len(df),
        "cols": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "sample": clean_sample
    }

@router.get("/{run_id}/history")
async def get_transformation_history(
    run_id: int,
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Fetches persistent history log of executed transformation steps."""
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    rows = await db.fetch(
        "SELECT id as step_id, prompt, code, rows, cols, TO_CHAR(created_at, 'HH:MI AM') as created_at FROM transformation_history WHERE run_id = $1 ORDER BY id DESC",
        run_id
    )
    return [dict(r) for r in rows]

@router.post("/{run_id}/execute", response_model=TransformResponse)
async def execute_transformation(
    run_id: int,
    req: TransformExecuteRequest,
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Executes a natural language or custom Pandas transformation on the dataset."""
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    run_info = await db.fetchrow("SELECT dataset_id, target_column, task_type FROM pipeline_runs WHERE id = $1", run_id)
    if not run_info:
        raise HTTPException(status_code=404, detail="Run not found")

    if req.from_step_id:
        await db.execute("DELETE FROM transformation_history WHERE run_id = $1 AND id > $2", run_id, req.from_step_id)
    
    try:
        df = await _replay_transformations(db, run_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {exc}")

    code_to_run = req.custom_code or ""
    summary_text = ""
    chart_spec = None
    requires_chart = False
    chart_type = "bar"
    x_axis = ""
    y_axis = ""

    if not code_to_run:
        from app.workflows.steps.chat_step import create_chat_agent, ChatIntent
        agent = create_chat_agent(run_id)
        columns_info = ", ".join(df.columns.tolist())
        prompt = (
            f"Columns: {columns_info}\n"
            f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} cols\n"
            f"User transformation/analysis instruction: {req.prompt}\n"
            "Generate valid Python Pandas code operating on 'df'. "
            "If creating columns or filtering rows, mutate 'df'. "
            "If asking for a plot/chart, set requires_chart=True and chart parameters. "
            "NEVER call matplotlib, plt.show(), or df.plot(). Write aggregation into 'result' if querying."
        )
        try:
            result = await agent.arun(prompt)
            intent: ChatIntent = result.content if isinstance(result.content, ChatIntent) else ChatIntent(intent="general_chat", response=str(result.content))
            code_to_run = intent.pandas_code or ""
            summary_text = intent.response or f"Applied: {req.prompt}"
            requires_chart = getattr(intent, "requires_chart", False)
            chart_type = getattr(intent, "chart_type", "bar") or "bar"
            x_axis = getattr(intent, "x_axis", "") or ""
            y_axis = getattr(intent, "y_axis", "") or ""
        except Exception as exc:
            logger.error(f"Transformation code generation failed: {exc}")
            summary_text = f"Transformation attempted: {req.prompt}"

    # Clean matplotlib calls if present
    if "plt." in code_to_run or ".plot(" in code_to_run:
        lines = [
            l for l in code_to_run.split("\n") 
            if not l.strip().startswith("plt.") and ".plot(" not in l
        ]
        code_to_run = "\n".join(lines)
        requires_chart = True

    # Execute pandas code
    updated_df, table_md, success = execute_pandas_operation(df, code_to_run, is_mutation=True)

    if not success:
        avail_cols = ", ".join([f"'{c}'" for c in df.columns.tolist()])
        friendly_error = (
            f"Could not execute '{req.prompt}'. The request may refer to columns not present in this dataset. "
            f"Available columns: [{avail_cols}]. Try asking to filter rows, create ratio features, or plot these columns."
        )
        return TransformResponse(
            summary=friendly_error,
            table_markdown=f"⚠️ **Notice:** {table_md}",
            rows=len(df),
            cols=len(df.columns),
            chart_spec=None,
            step=None
        )

    # Persist updated snapshot
    snapshot_filename = f"run_{run_id}_transformations.csv"
    saved_key = await save_snapshot(db, updated_df, run_id, snapshot_filename)
    await db.execute(
        "INSERT INTO stage_outputs (run_id, stage, r2_snapshot_path) VALUES ($1, 'cleaning', $2)",
        run_id, saved_key
    )

    # Save to history database table
    step_id = await db.fetchval(
        "INSERT INTO transformation_history (run_id, prompt, code, rows, cols) VALUES ($1, $2, $3, $4, $5) RETURNING id",
        run_id, req.prompt, code_to_run, len(updated_df), len(updated_df.columns)
    )

    # Build chart if requested or detected
    if requires_chart:
        chart_spec = build_chart_spec(updated_df, chart_type, f"{chart_type.capitalize()} Chart", x_axis, y_axis)

    step_info = {
        "step_id": step_id,
        "prompt": req.prompt,
        "code": code_to_run,
        "rows": len(updated_df),
        "cols": len(updated_df.columns),
        "created_at": "Just now"
    }

    return TransformResponse(
        summary=summary_text or f"Successfully transformed dataset to {len(updated_df)} rows × {len(updated_df.columns)} columns.",
        table_markdown=table_md,
        rows=len(updated_df),
        cols=len(updated_df.columns),
        chart_spec=chart_spec,
        step=step_info
    )

async def _replay_transformations(db: asyncpg.Connection, run_id: int) -> pd.DataFrame:
    """Replays active transformation steps for run_id from clean base dataset."""
    clean_df = await load_dataframe(db, run_id, stage="cleaning")
    working_df = clean_df.copy()

    remaining_steps = await db.fetch(
        "SELECT id, code FROM transformation_history WHERE run_id = $1 ORDER BY id ASC",
        run_id
    )

    for step in remaining_steps:
        code = step["code"]
        if code:
            working_df, _, _ = execute_pandas_operation(working_df, code, is_mutation=True)

    snapshot_filename = f"run_{run_id}_transformations.csv"
    saved_key = await save_snapshot(db, working_df, run_id, snapshot_filename)
    await db.execute(
        "INSERT INTO stage_outputs (run_id, stage, r2_snapshot_path) VALUES ($1, 'cleaning', $2)",
        run_id, saved_key
    )
    return working_df

@router.delete("/{run_id}/history/{step_id}")
async def revert_transformation_step(
    run_id: int,
    step_id: int,
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Reverts/deletes a specific transformation step and replays remaining steps."""
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    deleted = await db.execute("DELETE FROM transformation_history WHERE id = $1 AND run_id = $2", step_id, run_id)
    if deleted == "DELETE 0":
        raise HTTPException(status_code=404, detail="Step not found")

    updated_df = await _replay_transformations(db, run_id)
    
    # Return updated history and dataset sample
    rows = await db.fetch(
        "SELECT id as step_id, prompt, code, rows, cols, TO_CHAR(created_at, 'HH:MI AM') as created_at FROM transformation_history WHERE run_id = $1 ORDER BY id DESC",
        run_id
    )
    sample_rows = updated_df.head(15).to_dict(orient="records")
    clean_sample = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in sample_rows
    ] if not updated_df.empty else []

    return {
        "message": "Step reverted successfully",
        "history": [dict(r) for r in rows],
        "rows": len(updated_df),
        "cols": len(updated_df.columns),
        "columns": updated_df.columns.tolist(),
        "dtypes": {c: str(updated_df[c].dtype) for c in updated_df.columns},
        "sample": clean_sample
    }

@router.post("/{run_id}/reset")
async def reset_all_transformations(
    run_id: int,
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Resets all transformation steps and restores original clean dataset."""
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    await db.execute("DELETE FROM transformation_history WHERE run_id = $1", run_id)
    clean_df = await load_dataframe(db, run_id, stage="cleaning")

    snapshot_filename = f"run_{run_id}_transformations.csv"
    saved_key = await save_snapshot(db, clean_df, run_id, snapshot_filename)
    await db.execute(
        "INSERT INTO stage_outputs (run_id, stage, r2_snapshot_path) VALUES ($1, 'cleaning', $2)",
        run_id, saved_key
    )

    sample_rows = clean_df.head(15).to_dict(orient="records")
    clean_sample = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in sample_rows
    ] if not clean_df.empty else []

    return {
        "message": "Reset all transformations to original dataset",
        "history": [],
        "rows": len(clean_df),
        "cols": len(clean_df.columns),
        "columns": clean_df.columns.tolist(),
        "dtypes": {c: str(clean_df[c].dtype) for c in clean_df.columns},
        "sample": clean_sample
    }

@router.get("/{run_id}/history/{step_id}/data")
async def get_step_snapshot_data(
    run_id: int,
    step_id: int,
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Previews dataset state as it was at a specific historical step."""
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    clean_df = await load_dataframe(db, run_id, stage="cleaning")
    working_df = clean_df.copy()

    steps = await db.fetch(
        "SELECT id, code FROM transformation_history WHERE run_id = $1 AND id <= $2 ORDER BY id ASC",
        run_id, step_id
    )

    for step in steps:
        code = step["code"]
        if code:
            working_df, _, _ = execute_pandas_operation(working_df, code, is_mutation=True)

    sample_rows = working_df.head(15).to_dict(orient="records")
    clean_sample = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in sample_rows
    ] if not working_df.empty else []

    return {
        "step_id": step_id,
        "rows": len(working_df),
        "cols": len(working_df.columns),
        "columns": working_df.columns.tolist(),
        "dtypes": {c: str(working_df[c].dtype) for c in working_df.columns},
        "sample": clean_sample
    }

@router.post("/{run_id}/rollback/{step_id}")
async def rollback_to_step(
    run_id: int,
    step_id: int,
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Rolls back dataset and history to step_id (deleting all subsequent steps > step_id)."""
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    await db.execute("DELETE FROM transformation_history WHERE run_id = $1 AND id > $2", run_id, step_id)

    updated_df = await _replay_transformations(db, run_id)

    rows = await db.fetch(
        "SELECT id as step_id, prompt, code, rows, cols, TO_CHAR(created_at, 'HH:MI AM') as created_at FROM transformation_history WHERE run_id = $1 ORDER BY id DESC",
        run_id
    )
    sample_rows = updated_df.head(15).to_dict(orient="records")
    clean_sample = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in sample_rows
    ] if not updated_df.empty else []

    return {
        "message": f"Rolled back history to Step #{step_id}",
        "history": [dict(r) for r in rows],
        "rows": len(updated_df),
        "cols": len(updated_df.columns),
        "columns": updated_df.columns.tolist(),
        "dtypes": {c: str(updated_df[c].dtype) for c in updated_df.columns},
        "sample": clean_sample
    }


