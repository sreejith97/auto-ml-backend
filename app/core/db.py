import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

# asyncpg expects postgresql:// or postgres://
if DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

# Global connection pool
db_pool: asyncpg.Pool = None

import logging

logger = logging.getLogger(__name__)

async def init_db_pool():
    global db_pool
    db_pool = await asyncpg.create_pool(dsn=DATABASE_URL)
    logger.info("Successfully connected to the PostgreSQL database.")

async def close_db_pool():
    global db_pool
    if db_pool:
        await db_pool.close()

async def get_db():
    if not db_pool:
        raise Exception("Database pool is not initialized")
    async with db_pool.acquire() as connection:
        yield connection
