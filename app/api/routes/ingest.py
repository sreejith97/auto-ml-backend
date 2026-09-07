from fastapi import APIRouter, Depends
import asyncpg
from app.core.db import get_db
from app.models.schemas import HelloResponse
from app.core.storage import get_s3_client
import os

from app.core.queries import datasets
from app.agents import narration

router = APIRouter()

@router.post("/hello", response_model=HelloResponse)
async def hello_pipeline(db: asyncpg.Connection = Depends(get_db)):
    """A 'Hello Pipeline' round trip to test Postgres (Raw SQL) and R2."""
    
    # 1. DB Write (Raw SQL via helper)
    new_dataset_id = await datasets.create_dataset(
        conn=db,
        name="hello_dataset",
        source_type="dummy",
        r2_raw_path="dummy_path.csv",
        row_count=1
    )
    
    # Create a pipeline_run so conversation messages have a valid run_id FK
    run_id = await db.fetchval(
        "INSERT INTO pipeline_runs (dataset_id, status) VALUES ($1, 'ingested') RETURNING id",
        new_dataset_id
    )
    
    # Trigger Narration
    await narration.narrate_ingestion(db, run_id, "hello_dataset", 1)
    
    # 2. DB Read
    db_test_res = f"Created dataset with id {new_dataset_id} and run id {run_id} using raw SQL!"
    
    # 3. R2 Write & Read
    r2_test_res = "R2 client not configured"
    s3_client = get_s3_client()
    if s3_client:
        bucket_name = os.getenv("R2_BUCKET_NAME")
        try:
            # write
            s3_client.put_object(Bucket=bucket_name, Key="hello.txt", Body=b"Hello from pipeline!")
            # read
            obj = s3_client.get_object(Bucket=bucket_name, Key="hello.txt")
            data = obj['Body'].read().decode('utf-8')
            r2_test_res = f"Wrote and read from R2: '{data}'"
        except Exception as e:
            r2_test_res = f"R2 Error: {e}"
            
    return HelloResponse(
        message="Round trip complete", 
        db_test=db_test_res, 
        r2_test=r2_test_res,
        dataset_id=new_dataset_id,
        run_id=run_id
    )

from fastapi import UploadFile, File, Form, HTTPException
import uuid
from typing import Dict, Any, Optional

from app.api.deps import get_current_user

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    run_id: Optional[int] = Form(None),
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Uploads a CSV to R2 and creates the dataset and updates the pipeline_run."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    s3_client = get_s3_client()
    if not s3_client:
        raise HTTPException(status_code=500, detail="R2 client is not configured.")
        
    bucket_name = os.getenv("R2_BUCKET_NAME")
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    
    # Read the file into memory (fine for < 500MB as per frontend limit)
    content = await file.read()
    
    import pandas as pd
    import io
    from app.agents.schema_infer import infer_schema
    from app.core.queries import columns as col_queries
    
    try:
        df = pd.read_csv(io.BytesIO(content))
        row_count = len(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
    if row_count == 0: row_count = 1
    
    try:
        s3_client.put_object(Bucket=bucket_name, Key=unique_filename, Body=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload to R2: {str(e)}")
        
    # Create DB rows
    new_dataset_id = await db.fetchval(
        """
        INSERT INTO datasets (name, source_type, r2_raw_path, row_count, user_id)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
        """,
        file.filename, "csv_upload", unique_filename, row_count, current_user['id']
    )
    
    # Run Schema Inference
    inferred_schema = await infer_schema(df)
    
    # Save schema to database
    for col in inferred_schema:
        await col_queries.upsert_column_spec(
            db, 
            new_dataset_id, 
            col['column_name'], 
            col['role'], 
            col['semantic_type'], 
            None, 
            col['notes']
        )
    
    if run_id:
        # Update existing run
        await db.execute(
            "UPDATE pipeline_runs SET dataset_id = $1, status = 'ingested' WHERE id = $2 AND user_id = $3",
            new_dataset_id, run_id, current_user['id']
        )
    else:
        # Create new run (fallback if no run_id provided)
        run_id = await db.fetchval(
            "INSERT INTO pipeline_runs (dataset_id, status, user_id) VALUES ($1, 'ingested', $2) RETURNING id",
            new_dataset_id, current_user['id']
        )
    
    # Trigger Narration
    await narration.narrate_ingestion(db, run_id, file.filename, row_count, inferred_schema)
    
    return {
        "message": "Upload successful",
        "dataset_id": new_dataset_id,
        "run_id": run_id
    }
