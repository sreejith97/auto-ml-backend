from fastapi import APIRouter, Depends, HTTPException
import asyncpg
from app.core.db import get_db
from app.core.queries import pipeline_runs
from app.models.schemas import TaskCreate, TaskResponse
from app.api.deps import get_current_user
from app.agents import narration

router = APIRouter()

@router.post("/{run_id}", response_model=TaskResponse)
async def create_task(
    run_id: int, 
    task: TaskCreate, 
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    await pipeline_runs.update_run_task(
        conn=db,
        run_id=run_id,
        task_type=task.task_type,
        target_column=task.target_column,
        time_column=task.time_column,
        forecast_horizon=task.forecast_horizon
    )
    
    # Narrate the task creation
    await narration.narrate_task(db, run_id, task.task_type, task.target_column)
    
    return TaskResponse(run_id=run_id, status="task_defined")

@router.get("/{dataset_id}/recommend")
async def recommend_task(
    dataset_id: int, 
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    import json
    
    owner = await db.fetchval("SELECT user_id FROM datasets WHERE id = $1", dataset_id)
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # get project description and columns
    project = await db.fetchrow("SELECT project_name, description FROM pipeline_runs WHERE dataset_id = $1 ORDER BY id DESC LIMIT 1", dataset_id)
    columns = await db.fetch("SELECT column_name, semantic_type, notes FROM columns_spec WHERE dataset_id = $1", dataset_id)
    
    col_info = "\n".join([f"- {c['column_name']} ({c['semantic_type']}): {c['notes']}" for c in columns])
    desc = project['description'] if project and project['description'] else 'None'
    
    system_prompt = f"""
    You are an AI assistant helping a user configure a machine learning task.
    Project Description: {desc}
    Columns:
    {col_info}
    
    Based on the project description and the dataset columns, recommend the best machine learning task type ('classification', 'regression', or 'forecasting') and the target column to predict.
    
    Return ONLY a strict JSON object with this exact shape:
    {{
      "task_type": "classification" | "regression" | "forecasting",
      "target_column": "exact_column_name_from_list",
      "reasoning": "A 1-sentence explanation of why you chose this."
    }}
    """
    
    try:
        from agno.agent import Agent
        from pydantic import BaseModel, Field
        from app.core.llm import get_model

        class TaskRecommendation(BaseModel):
            task_type: str = Field(description="One of 'classification', 'regression', or 'forecasting'")
            target_column: str = Field(description="Exact column name from the dataset list")
            reasoning: str = Field(description="1-sentence explanation of why you chose this")

        rec_agent = Agent(
            model=get_model(),
            instructions="Recommend the best ML task type and target column based on the dataset structure.",
            output_schema=TaskRecommendation,
        )
        result = await rec_agent.arun(system_prompt)
        rec: TaskRecommendation = result.content if isinstance(result.content, TaskRecommendation) else TaskRecommendation.model_validate_json(str(result.content))
        return rec.model_dump()
    except Exception as e:
        return {"task_type": "regression", "target_column": "", "reasoning": f"Failed to generate recommendation: {e}"}
