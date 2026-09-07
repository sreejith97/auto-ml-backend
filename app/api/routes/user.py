from fastapi import APIRouter, Depends, HTTPException
import asyncpg
from typing import List, Optional
from pydantic import BaseModel
from app.core.db import get_db
from app.api.deps import get_current_user

router = APIRouter()

class ProjectCreate(BaseModel):
    project_name: str
    description: str

@router.get("/runs")
async def get_user_runs(
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    query = """
        SELECT pr.id, pr.dataset_id, pr.status, pr.task_type, pr.target_column, 
               pr.created_at, pr.project_name, pr.description, pr.model_type,
               d.name as dataset_name
        FROM pipeline_runs pr
        LEFT JOIN datasets d ON pr.dataset_id = d.id
        WHERE pr.user_id = $1
        ORDER BY pr.created_at DESC
    """
    records = await db.fetch(query, current_user['id'])
    return [dict(r) for r in records]

@router.post("/runs")
async def create_project(
    project: ProjectCreate,
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    query = """
        INSERT INTO pipeline_runs (user_id, project_name, description, status)
        VALUES ($1, $2, $3, 'pending')
        RETURNING id
    """
    run_id = await db.fetchval(
        query, 
        current_user['id'], 
        project.project_name, 
        project.description
    )
    return {"run_id": run_id}

@router.delete("/runs/{run_id}")
async def delete_project(
    run_id: int,
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    # Verify ownership
    owner_id = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if not owner_id:
        raise HTTPException(status_code=404, detail="Project not found")
    if owner_id != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized to delete this project")
        
    await db.execute("DELETE FROM pipeline_runs WHERE id = $1", run_id)
    return {"message": "Project deleted successfully"}
