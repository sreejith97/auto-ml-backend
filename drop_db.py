import asyncio
import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

async def drop():
    url = os.getenv('DATABASE_URL').replace('postgresql+asyncpg://', 'postgresql://')
    conn = await asyncpg.connect(url)
    await conn.execute('DROP TABLE IF EXISTS model_runs, questions_pending, rules, stage_outputs, pipeline_runs, columns_spec, datasets CASCADE;')
    await conn.close()
    print("Tables dropped.")

if __name__ == "__main__":
    asyncio.run(drop())
