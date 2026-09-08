from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import asyncpg
from app.core.db import get_db, init_db_pool, close_db_pool
from app.core.storage import get_s3_client, test_r2_connection
from app.models.schemas import HealthResponse
from app.api.routes import ingest, columns, chat, task, clean, eda, auth, user, features, training, models, predict, pipeline
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db_pool()
    await test_r2_connection()
    yield
    # Shutdown
    await close_db_pool()

app = FastAPI(
    title="Human-Guided ML Pipeline",
    description="A full-stack ML Pipeline application. Interactive documentation is available here.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(user.router, prefix="/api/v1/user", tags=["user"])
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingest"])
app.include_router(columns.router, prefix="/api/v1/columns", tags=["columns"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(task.router, prefix="/api/v1/task", tags=["task"])
app.include_router(clean.router, prefix="/api/v1/clean", tags=["clean"])
app.include_router(eda.router, prefix="/api/v1/eda", tags=["eda"])
app.include_router(features.router, prefix="/api/v1/features", tags=["features"])
app.include_router(training.router, prefix="/api/v1/training", tags=["training"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(predict.router, prefix="/api/v1/predict", tags=["predict"])
app.include_router(pipeline.router, prefix="/api/v1/pipeline", tags=["pipeline"])

@app.get("/health", response_model=HealthResponse)
async def health_check(db: asyncpg.Connection = Depends(get_db)):
    db_connected = False
    r2_connected = False
    
    # Check DB
    try:
        await db.execute("SELECT 1")
        db_connected = True
    except Exception as e:
        print(f"DB Error: {e}")
        
    # Check R2
    try:
        s3_client = get_s3_client()
        if s3_client:
            s3_client.list_buckets()
            r2_connected = True
    except Exception as e:
        print(f"R2 Error: {e}")
        
    status = "ok" if db_connected and r2_connected else "error"
    return HealthResponse(status=status, db_connected=db_connected, r2_connected=r2_connected)
