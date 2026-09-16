import asyncio
import asyncpg
import os
import json
from dotenv import load_dotenv
import pandas as pd

from app.core.db import init_db_pool, get_db
from app.api.routes.clean import run_cleaning, answer_question, get_clean_state
from app.core.storage import save_snapshot

async def verify():
    load_dotenv()
    await init_db_pool()
    db_gen = get_db()
    conn = await anext(db_gen)
    
    try:
        # Create a user
        user_id = await conn.fetchval("INSERT INTO users (email, password_hash) VALUES ('verify@verify.com', 'pwd') ON CONFLICT (email) DO UPDATE SET email='verify@verify.com' RETURNING id")
        
        # Create a dummy dataframe with missing values to trigger HITL
        df = pd.DataFrame({
            "income": [50000, 60000, None, 55000, 65000, 52000, None, 48000, 51000, 59000] # 20% missing, should be low confidence (0.3)
        })
        
        # Insert dataset
        dataset_id = await conn.fetchval("INSERT INTO datasets (name, source_type, r2_raw_path, row_count, user_id) VALUES ('Verify DS', 'upload', 'dataset_raw.csv', 10, $1) RETURNING id", user_id)
        
        # Insert column spec
        await conn.execute("INSERT INTO columns_spec (dataset_id, column_name, role, semantic_type) VALUES ($1, 'income', 'feature', 'numeric')", dataset_id)
        
        # ---------------- RUN 1 ----------------
        run_id_1 = await conn.fetchval("INSERT INTO pipeline_runs (dataset_id, user_id, task_type, target_column, status) VALUES ($1, $2, 'regression', 'income', 'ingested') RETURNING id", dataset_id, user_id)
        # Mock the dataframe in storage
        os.makedirs(f"local_storage/{dataset_id}", exist_ok=True)
        df.to_csv(f"local_storage/{dataset_id}/dataset_raw.csv", index=False)
        await conn.execute("INSERT INTO stage_outputs (run_id, stage, r2_snapshot_path) VALUES ($1, 'ingestion', 'dataset_raw.csv')", run_id_1)
        
        print(f"--- RUN 1 (Run ID: {run_id_1}) ---")
        user_dict = {"id": user_id, "email": "verify@verify.com"}
        res1 = await run_cleaning(run_id_1, conn, user_dict)
        print(f"Run 1 auto_applied count: {len(res1.auto_applied)}")
        print(f"Run 1 rule_reused count: {len(res1.rule_reused)}")
        print(f"Run 1 pending count: {len(res1.pending)}")
        
        if len(res1.pending) > 0:
            q_id = res1.pending[0]["id"]
            print(f"Resolving pending issue {q_id} with 'impute_median'...")
            await answer_question(run_id_1, q_id, "impute_median", conn)
        
        # ---------------- RUN 2 ----------------
        # Same dataset, new run
        run_id_2 = await conn.fetchval("INSERT INTO pipeline_runs (dataset_id, user_id, task_type, target_column, status) VALUES ($1, $2, 'regression', 'income', 'ingested') RETURNING id", dataset_id, user_id)
        os.makedirs(f"local_storage/{dataset_id}", exist_ok=True)
        df.to_csv(f"local_storage/{dataset_id}/dataset_raw.csv", index=False)
        await conn.execute("INSERT INTO stage_outputs (run_id, stage, r2_snapshot_path) VALUES ($1, 'ingestion', 'dataset_raw.csv')", run_id_2)
        
        print(f"\n--- RUN 2 (Run ID: {run_id_2}) ---")
        res2 = await run_cleaning(run_id_2, conn, user_dict)
        print(f"Run 2 auto_applied count: {len(res2.auto_applied)}")
        print(f"Run 2 rule_reused count: {len(res2.rule_reused)}")
        print(f"Run 2 pending count: {len(res2.pending)}")
        
        if len(res2.rule_reused) > 0:
            print(f"Success! Reused rule for column: {res2.rule_reused[0]['column']}, action: {res2.rule_reused[0]['action']}")
        else:
            print("Failed to reuse rule!")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(verify())
