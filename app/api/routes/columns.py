from fastapi import APIRouter, Depends, HTTPException
import asyncpg
from typing import List
from app.core.db import get_db
from app.core.queries import columns as columns_queries
from app.models.schemas import ColumnSpecCreate, ColumnSpecResponse, ColumnSpecUpsertResponse
from app.api.deps import get_current_user

router = APIRouter()

@router.get("/{dataset_id}", response_model=List[ColumnSpecResponse])
async def get_columns(
    dataset_id: int, 
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM datasets WHERE id = $1", dataset_id)
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    records = await columns_queries.get_columns_for_dataset(db, dataset_id)
    return [dict(r) for r in records]

@router.post("/{dataset_id}", response_model=ColumnSpecUpsertResponse)
async def upsert_column(
    dataset_id: int, 
    col_spec: ColumnSpecCreate, 
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM datasets WHERE id = $1", dataset_id)
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    # 1. Upsert the column specification
    await columns_queries.upsert_column_spec(
        conn=db,
        dataset_id=dataset_id,
        column_name=col_spec.column_name,
        role=col_spec.role,
        semantic_type=col_spec.semantic_type,
        missing_meaning=col_spec.missing_meaning,
        notes=col_spec.notes
    )
    
    # 2. Check if exactly one target column exists to allow pipeline advancement
    target_count = await columns_queries.count_target_columns(db, dataset_id)
    can_advance = (target_count == 1)
    
    if can_advance and col_spec.role == 'target':
        from app.agents import narration
        run_id = await db.fetchval("SELECT id FROM pipeline_runs WHERE dataset_id = $1 ORDER BY id DESC LIMIT 1", dataset_id)
        if run_id:
            await narration.narrate_columns(db, run_id, col_spec.column_name)
    
    return ColumnSpecUpsertResponse(status="success", can_advance=can_advance)
