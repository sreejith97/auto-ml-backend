import asyncio
import asyncpg
import os
from app.api.routes.chat import post_chat_message
from app.models.schemas import ChatRequest
import app.core.llm as llm

async def main():
    llm.model_name = "groq/compound-mini"
    
    dsn = os.getenv("DATABASE_URL").replace("+asyncpg", "")
    conn = await asyncpg.connect(dsn)
    try:
        req = ChatRequest(message="has_pool is the target column")
        res = await post_chat_message(run_id=1, req=req, db=conn)
        print("SUCCESS:", res)
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        await conn.close()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())
