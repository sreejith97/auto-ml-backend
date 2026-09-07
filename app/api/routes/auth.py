from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
import asyncpg
from pydantic import BaseModel, EmailStr
from app.core.db import get_db
from app.core.security import get_password_hash, verify_password, create_access_token

router = APIRouter()

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

@router.post("/register", response_model=Token)
async def register(user: UserCreate, db: asyncpg.Connection = Depends(get_db)):
    # Check if user exists
    existing = await db.fetchval("SELECT id FROM users WHERE email = $1", user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    hashed_password = get_password_hash(user.password)
    
    # Insert new user
    query = """
        INSERT INTO users (email, password_hash)
        VALUES ($1, $2)
        RETURNING id, email
    """
    new_user = await db.fetchrow(query, user.email, hashed_password)
    
    # Create token
    access_token = create_access_token(data={"sub": str(new_user['id'])})
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user": {"id": new_user['id'], "email": new_user['email']}
    }

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: asyncpg.Connection = Depends(get_db)):
    # OAuth2PasswordRequestForm uses 'username' instead of 'email'
    user = await db.fetchrow("SELECT id, email, password_hash FROM users WHERE email = $1", form_data.username)
    
    if not user or not verify_password(form_data.password, user['password_hash']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token = create_access_token(data={"sub": str(user['id'])})
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user": {"id": user['id'], "email": user['email']}
    }
