import os
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

def get_s3_client():
    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
        return None # Return None or mock client for local testing if not configured

    endpoint_url = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto", 
        config=Config(signature_version="s3v4"),
    )

import logging
import pandas as pd
import io
import asyncpg

logger = logging.getLogger(__name__)

async def test_r2_connection():
    client = get_s3_client()
    if not client:
        logger.warning("R2 credentials missing. Skipping R2 connection.")
        return
    try:
        # Actually make a network request to verify credentials
        client.head_bucket(Bucket=R2_BUCKET_NAME)
        logger.info(f"Successfully connected to Cloudflare R2 bucket: {R2_BUCKET_NAME}")
    except Exception as e:
        logger.error(f"Failed to connect to Cloudflare R2 bucket '{R2_BUCKET_NAME}': {e}")

def get_local_path(dataset_id: int, filename: str) -> str:
    path = os.path.join("local_storage", str(dataset_id))
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)

async def load_dataframe(db: asyncpg.Connection, run_id: int, stage: str = None) -> pd.DataFrame:
    dataset_id = await db.fetchval("SELECT dataset_id FROM pipeline_runs WHERE id = $1", run_id)
    if not dataset_id:
        raise Exception(f"No dataset found for run_id {run_id}")

    r2_path = None
    if stage:
        r2_path = await db.fetchval(
            "SELECT r2_snapshot_path FROM stage_outputs WHERE run_id = $1 AND stage = $2 ORDER BY created_at DESC LIMIT 1", 
            run_id, stage
        )

    if not r2_path:
        r2_path = await db.fetchval("SELECT r2_raw_path FROM datasets WHERE id = $1", dataset_id)
        
    if not r2_path:
        raise Exception(f"No valid data path found for run_id {run_id}")

    s3 = get_s3_client()
    if s3:
        try:
            obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=r2_path)
            return pd.read_csv(io.BytesIO(obj['Body'].read()))
        except Exception as e:
            logger.warning(f"R2 read failed for {r2_path}: {e}. Falling back to local storage.")
    
    local_file = get_local_path(dataset_id, os.path.basename(r2_path))
    if os.path.exists(local_file):
        return pd.read_csv(local_file)
    else:
        # Emergency fallback for testing if dummy_dataset exists in root
        if os.path.exists(os.path.basename(r2_path)):
            return pd.read_csv(os.path.basename(r2_path))
        raise Exception(f"Data file not found locally: {local_file}")

async def save_snapshot(db: asyncpg.Connection, df: pd.DataFrame, run_id: int, filename: str) -> str:
    dataset_id = await db.fetchval("SELECT dataset_id FROM pipeline_runs WHERE id = $1", run_id)
    s3 = get_s3_client()
    
    if s3:
        try:
            csv_buffer = io.BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            s3.put_object(Bucket=R2_BUCKET_NAME, Key=filename, Body=csv_buffer.getvalue())
            return filename
        except Exception as e:
            logger.warning(f"R2 write failed for {filename}: {e}. Falling back to local storage.")
            
    local_file = get_local_path(dataset_id, filename)
    df.to_csv(local_file, index=False)
    return filename

import joblib

async def save_model(db: asyncpg.Connection, model: any, run_id: int, filename: str) -> str:
    dataset_id = await db.fetchval("SELECT dataset_id FROM pipeline_runs WHERE id = $1", run_id)
    s3 = get_s3_client()
    
    if s3:
        try:
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)
            s3.put_object(Bucket=R2_BUCKET_NAME, Key=filename, Body=buffer.getvalue())
            return filename
        except Exception as e:
            logger.warning(f"R2 write failed for model {filename}: {e}. Falling back to local storage.")
            
    local_file = get_local_path(dataset_id, filename)
    joblib.dump(model, local_file)
    return filename

async def load_model(db: asyncpg.Connection, run_id: int, path: str) -> any:
    dataset_id = await db.fetchval("SELECT dataset_id FROM pipeline_runs WHERE id = $1", run_id)
    s3 = get_s3_client()
    
    if s3:
        try:
            obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=path)
            return joblib.load(io.BytesIO(obj['Body'].read()))
        except Exception as e:
            logger.warning(f"R2 read failed for model {path}: {e}. Falling back to local storage.")
            
    local_file = get_local_path(dataset_id, path)
    if os.path.exists(local_file):
        return joblib.load(local_file)
    raise Exception(f"Model file not found locally or in R2: {path}")
