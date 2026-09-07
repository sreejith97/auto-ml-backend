from fastapi import APIRouter, Depends, HTTPException
import asyncpg
from typing import List
from app.core.db import get_db
from app.core.queries import conversation as conv_queries
from app.api.deps import get_current_user
from app.models.schemas import ChatMessage, ChatRequest, ChatResponse
from pydantic import BaseModel
from app.agents import nl_answer_parser

router = APIRouter()

@router.get("/{run_id}", response_model=List[ChatMessage])
async def get_chat_history(
    run_id: int, 
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    # Verify ownership
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    records = await conv_queries.get_messages(db, run_id)
    return [dict(r) for r in records]

@router.post("/{run_id}", response_model=ChatResponse)
async def post_chat_message(
    run_id: int, 
    req: ChatRequest, 
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    # Verify ownership
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    # 1. Insert user message
    # For now we will assume the current stage is 'general' if we can't infer it, 
    # but the frontend could pass it. Let's default to 'chat'
    stage = "chat" 
    
    # Check if there are any pending questions for this run (simplified for Phase 2.5 scaffold)
    # Ideally we fetch the pending questions from questions_pending table
    query_pending = "SELECT * FROM questions_pending WHERE run_id = $1 AND status = 'pending' LIMIT 1"
    pending_q = await db.fetchrow(query_pending, run_id)
    
    await conv_queries.insert_message(db, run_id, 'user', req.message, stage, pending_q['id'] if pending_q else None)
    
    # 2 & 3. Process Message
    assistant_reply = ""
    action_taken = False
    
    if pending_q:
        # We have a pending question, attempt to parse the free text answer
        allowed_options = ["drop", "impute_mean", "impute_median", "drop_rows", "cap_iqr", "lowercase_all"] # Extended for phase 4
        matched_option, conf, clarification = await nl_answer_parser.parse_answer(pending_q['issue_type'], req.message, allowed_options)
        
        if matched_option:
            # CALL THE ACTUAL CLEANING LOGIC!
            from app.api.routes.clean import answer_question
            await answer_question(run_id, pending_q['id'], matched_option, db)
            
            assistant_reply = f"Understood. I have applied '{matched_option}' to resolve the issue for '{pending_q['column_name']}'."
            action_taken = True
        else:
            assistant_reply = clarification
    else:
        # General Read-Only Q&A or Schema Editing
        run_info = await db.fetchrow("SELECT dataset_id, status FROM pipeline_runs WHERE id = $1", run_id)
        
        if not run_info:
            assistant_reply = "Run not found."
        else:
            dataset_id = run_info['dataset_id']
            current_stage = run_info['status']
            
            intent_data = await nl_answer_parser.parse_chat_intent(req.message, current_stage)
            
            if intent_data.get('intent') == 'update_column_role':
                col = intent_data.get('column_name')
                role = intent_data.get('new_role')
                
                exists = await db.fetchval("SELECT id FROM columns_spec WHERE dataset_id = $1 AND column_name = $2", dataset_id, col)
                if exists and role in ['target', 'feature', 'ignore', 'id']:
                    # If target, maybe we should clear other targets? (Optional, but good practice)
                    if role == 'target':
                        await db.execute("UPDATE columns_spec SET role = 'feature' WHERE dataset_id = $1 AND role = 'target'", dataset_id)
                        
                    await db.execute("UPDATE columns_spec SET role = $1 WHERE dataset_id = $2 AND column_name = $3", role, dataset_id, col)
                    assistant_reply = f"I've updated the '{col}' column to be the {role}!"
                    action_taken = True
                else:
                    assistant_reply = f"I couldn't find a column named '{col}' or the role is invalid."
            else:
                # Give actual context to answer general questions
                from app.core import llm
                
                columns = await db.fetch("SELECT column_name, role, semantic_type, notes FROM columns_spec WHERE dataset_id = $1", dataset_id)
                project = await db.fetchrow("SELECT project_name, description FROM pipeline_runs WHERE id = $1", run_id)
                
                col_info = "\n".join([f"- {c['column_name']} (Role: {c['role']}, Type: {c['semantic_type']})" for c in columns])
                proj_name = project['project_name'] if project and project['project_name'] else 'Unnamed Project'
                proj_desc = project['description'] if project and project['description'] else 'No description'
                
                context = f"Project: {proj_name}\nGoal/Description: {proj_desc}\n\nDataset Columns:\n{col_info}"
                
                system_prompt = f"You are a helpful AI data engineering assistant for an AutoML platform. Use the following context to answer the user's question concisely and naturally in plain text.\n\nContext:\n{context}\n\nDO NOT output raw JSON arrays. Be helpful and conversational."
                
                assistant_reply = await llm.complete(system_prompt=system_prompt, user_prompt=req.message)
    # 4. Insert Assistant Reply
    await conv_queries.insert_message(db, run_id, 'assistant', assistant_reply, stage)
    
    return ChatResponse(
        assistant_reply=assistant_reply,
        action_taken=action_taken,
        updated_state=None
    )

class SystemMessageRequest(BaseModel):
    message: str

@router.post("/{run_id}/system")
async def post_system_message(
    run_id: int, 
    req: SystemMessageRequest, 
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    await conv_queries.insert_message(db, run_id, 'assistant', req.message, 'system')
    return {"status": "success"}
