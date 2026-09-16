import asyncio
import asyncpg
import os
import pandas as pd
from dotenv import load_dotenv

from app.core.db import init_db_pool, get_db
from app.api.routes.clean import run_cleaning

async def verify_dataset(conn, user_id, name, path):
    print(f"\n--- Verifying Dataset: {name} ---")
    df = pd.read_csv(path)
    
    dataset_id = await conn.fetchval(
        "INSERT INTO datasets (name, source_type, r2_raw_path, row_count, user_id) VALUES ($1, 'upload', $2, $3, $4) RETURNING id",
        name, path, len(df), user_id
    )
    
    # Insert column specs
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        semantic = 'numeric' if 'int' in dtype_str or 'float' in dtype_str else 'text'
        await conn.execute("INSERT INTO columns_spec (dataset_id, column_name, role, semantic_type) VALUES ($1, $2, 'feature', $3)", dataset_id, col, semantic)
        
    run_id = await conn.fetchval("INSERT INTO pipeline_runs (dataset_id, user_id, task_type, target_column, status) VALUES ($1, $2, 'classification', $3, 'ingested') RETURNING id", dataset_id, user_id, df.columns[-1])
    
    os.makedirs(f"local_storage/{dataset_id}", exist_ok=True)
    df.to_csv(f"local_storage/{dataset_id}/{path}", index=False)
    await conn.execute("INSERT INTO stage_outputs (run_id, stage, r2_snapshot_path) VALUES ($1, 'ingestion', $2)", run_id, path)
    
    user_dict = {"id": user_id, "email": "verify@verify.com"}
    res = await run_cleaning(run_id, conn, user_dict)
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Auto-Applied Issues ({len(res.auto_applied)}):")
    for a in res.auto_applied:
        print(f"  - [{a['column']}] {a['issue']} (Conf: {a['confidence']:.2f}) -> {a['action']}")
    print(f"Pending Issues ({len(res.pending)}):")
    for p in res.pending:
        print(f"  - [{p['column']}] {p['issue']} (Conf: {p['confidence']:.2f})")
    print(f"Rule Reused Issues ({len(res.rule_reused)}):")
    for r in res.rule_reused:
        print(f"  - [{r['column']}] {r['issue']} (Conf: {r['confidence']:.2f}) -> {r['action']}")

async def main():
    load_dotenv()
    await init_db_pool()
    db_gen = get_db()
    conn = await anext(db_gen)
    try:
        user_id = await conn.fetchval("INSERT INTO users (email, password_hash) VALUES ('verify2@verify.com', 'pwd') ON CONFLICT (email) DO UPDATE SET email='verify2@verify.com' RETURNING id")
        await verify_dataset(conn, user_id, "Pima", "pima_perturbed.csv")
        await verify_dataset(conn, user_id, "SECOM", "secom_perturbed.csv")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
