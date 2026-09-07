import asyncio, asyncpg, os
from dotenv import load_dotenv
load_dotenv()

async def main():
    conn = await asyncpg.connect('postgresql://postgres:postgres@localhost:5432/postgres')
    
    rows = await conn.fetch('SELECT stage, r2_snapshot_path FROM stage_outputs WHERE run_id = 11')
    print('=== stage_outputs for run 11 ===')
    for r in rows:
        print(f'  {r["stage"]}: {r["r2_snapshot_path"]}')
    
    rows = await conn.fetch('SELECT id, column_name, status, final_transform FROM feature_proposals WHERE run_id = 11')
    print('=== feature_proposals for run 11 ===')
    for r in rows:
        print(f'  id={r["id"]} col={r["column_name"]} status={r["status"]} transform={r["final_transform"]}')
    
    status = await conn.fetchval('SELECT status FROM pipeline_runs WHERE id = 11')
    print(f'=== pipeline status: {status}')
    
    # Check unique constraint on stage_outputs
    constraint = await conn.fetch("""
        SELECT conname, contype FROM pg_constraint 
        WHERE conrelid = 'stage_outputs'::regclass
    """)
    print('=== stage_outputs constraints ===')
    for c in constraint:
        print(f'  {c["conname"]}: {c["contype"]}')
    
    await conn.close()

asyncio.run(main())
