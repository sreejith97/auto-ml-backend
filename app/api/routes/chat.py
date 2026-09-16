"""
app/api/routes/chat.py — Chat route (Agno PipelineChatAgent-backed)
====================================================================
Simplified from ~137 lines to ~80 using the typed ChatIntent model from
app/workflows/steps/chat_step.py.

The agent has persistent memory (Postgres-backed, session_id=str(run_id))
so it can reference previous pipeline stages without extra context injection.
"""

from fastapi import APIRouter, Depends, HTTPException
import asyncpg
from typing import List
from pydantic import BaseModel

from app.core.db import get_db
from app.core.queries import conversation as conv_queries
from app.api.deps import get_current_user
from app.models.schemas import ChatMessage, ChatRequest, ChatResponse

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /chat/{run_id} — fetch history (unchanged)
# ---------------------------------------------------------------------------

@router.get("/{run_id}", response_model=List[ChatMessage])
async def get_chat_history(
    run_id: int,
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    records = await conv_queries.get_messages(db, run_id)
    return [dict(r) for r in records]


# ---------------------------------------------------------------------------
# POST /chat/{run_id} — Agno-powered intent + response
# ---------------------------------------------------------------------------

@router.post("/{run_id}", response_model=ChatResponse)
async def post_chat_message(
    run_id: int,
    req: ChatRequest,
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    # 1. Fetch prior conversation history BEFORE inserting current message
    all_msgs = await conv_queries.get_messages(db, run_id)
    history_lines = [f"{m['role'].capitalize()}: {m['content']}" for m in all_msgs[-10:]]
    history_str = "\n".join(history_lines) if history_lines else "None"

    # 2. Insert current user message into database
    await conv_queries.insert_message(db, run_id, "user", req.message, "chat")

    # 3. Build dataset context for the agent
    run_info = await db.fetchrow(
        "SELECT dataset_id, status, project_name, description FROM pipeline_runs WHERE id = $1",
        run_id,
    )
    context = await _build_context(db, run_info)

    # 4. Run the Agno chat agent with full context & conversation history
    from app.workflows.steps.chat_step import create_chat_agent, ChatIntent

    agent = create_chat_agent(run_id)
    try:
        prompt = (
            f"Dataset context:\n{context}\n\n"
            f"Recent Conversation History (including pipeline stage narrations):\n{history_str}\n\n"
            f"Current User message: {req.message}"
        )
        result = await agent.arun(prompt)
        intent: ChatIntent = result.content if isinstance(result.content, ChatIntent) else ChatIntent(intent="general_chat", response=str(result.content))
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error(f"Chat agent failed: {exc}")
        intent = ChatIntent(intent="general_chat", response="I'm having trouble right now. Please try again.")

    # Act on the classified intent
    assistant_reply, action_taken, extra_meta = await _handle_intent(intent, run_id, run_info, db)

    # Insert assistant reply
    await conv_queries.insert_message(db, run_id, "assistant", assistant_reply, "chat")

    return ChatResponse(
        assistant_reply=assistant_reply,
        action_taken=action_taken,
        pandas_code=extra_meta.get("pandas_code"),
        table_markdown=extra_meta.get("table_markdown"),
        chart_spec=extra_meta.get("chart_spec"),
        updated_state=extra_meta.get("updated_state"),
    )


# ---------------------------------------------------------------------------
# POST /chat/{run_id}/system — internal system messages (unchanged)
# ---------------------------------------------------------------------------

class SystemMessageRequest(BaseModel):
    message: str


@router.post("/{run_id}/system")
async def post_system_message(
    run_id: int,
    req: SystemMessageRequest,
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    await conv_queries.insert_message(db, run_id, "assistant", req.message, "system")
    return {"status": "success"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _build_context(db: asyncpg.Connection, run_info) -> str:
    """Build a concise dataset context string for the chat agent."""
    if not run_info:
        return "No project context available."

    dataset_id = run_info["dataset_id"]
    proj_name = run_info.get("project_name") or "Unnamed Project"
    proj_desc = run_info.get("description") or "No description"

    columns = []
    if dataset_id:
        columns = await db.fetch(
            "SELECT column_name, role, semantic_type FROM columns_spec WHERE dataset_id = $1",
            dataset_id,
        )

    col_info = "\n".join(
        [f"- {c['column_name']} (Role: {c['role']}, Type: {c['semantic_type']})" for c in columns]
    ) or "No columns inferred yet."

    return f"Project: {proj_name}\nGoal: {proj_desc}\n\nColumns:\n{col_info}"


async def _handle_intent(
    intent,
    run_id: int,
    run_info,
    db: asyncpg.Connection,
) -> tuple[str, bool, dict]:
    """Dispatch on ChatIntent and return (reply, action_taken, extra_metadata)."""
    from app.workflows.steps.chat_step import ChatIntent

    extra = {}

    if intent.intent == "update_column_role":
        col = intent.column_name
        role = intent.new_role
        dataset_id = run_info["dataset_id"] if run_info else None

        if not col or not role or not dataset_id:
            return "I couldn't determine which column or role you meant. Could you be more specific?", False, extra

        exists = await db.fetchval(
            "SELECT id FROM columns_spec WHERE dataset_id = $1 AND column_name = $2",
            dataset_id, col,
        )
        if exists and role in ("target", "feature", "ignore", "id"):
            if role == "target":
                await db.execute(
                    "UPDATE columns_spec SET role = 'feature' WHERE dataset_id = $1 AND role = 'target'",
                    dataset_id,
                )
            await db.execute(
                "UPDATE columns_spec SET role = $1 WHERE dataset_id = $2 AND column_name = $3",
                role, dataset_id, col,
            )
            return f"Done! I've updated '{col}' to be the {role}.", True, extra
        else:
            return f"I couldn't find a column named '{col}', or the role '{role}' is invalid.", False, extra

    elif intent.intent == "answer_cleaning_question":
        if intent.confidence < 0.7 or not intent.matched_option:
            return intent.response or "Could you clarify your cleaning decision?", False, extra
        return f"Got it — I'll apply '{intent.matched_option}' to that issue.", True, extra

    elif intent.intent == "data_analysis_and_op":
        from app.core.storage import load_dataframe, save_snapshot
        from app.core.pandas_executor import execute_pandas_operation, build_chart_spec

        try:
            current_stage = run_info.get("status") or "cleaning"
            df = await load_dataframe(db, run_id, stage=current_stage)
        except Exception as err:
            return f"Unable to load dataset for analysis: {err}", False, extra

        pandas_code = intent.pandas_code or ""
        is_mutation = getattr(intent, "is_mutation", False)
        
        updated_df, table_md, success = execute_pandas_operation(df, pandas_code, is_mutation=is_mutation)

        extra["pandas_code"] = pandas_code
        extra["table_markdown"] = table_md

        # If data was mutated, save updated snapshot
        if success and is_mutation:
            try:
                snapshot_filename = f"run_{run_id}_{current_stage}_chat_mod.csv"
                saved_key = await save_snapshot(db, updated_df, run_id, snapshot_filename)
                await db.execute(
                    "INSERT INTO stage_outputs (run_id, stage, r2_snapshot_path) VALUES ($1, $2, $3)",
                    run_id, current_stage, saved_key
                )
                extra["updated_state"] = {"mutated": True, "rows": len(updated_df), "cols": len(updated_df.columns)}
            except Exception as save_err:
                import logging
                logging.getLogger(__name__).error(f"Failed to persist mutated dataframe: {save_err}")

        # Build chart payload if requested
        if getattr(intent, "requires_chart", False) and intent.chart_type:
            chart_spec = build_chart_spec(
                df=updated_df if success else df,
                chart_type=intent.chart_type,
                title=intent.chart_title or f"{intent.chart_type.capitalize()} Chart",
                x_axis=intent.x_axis or "",
                y_axis=intent.y_axis or ""
            )
            if chart_spec:
                extra["chart_spec"] = chart_spec

        reply_parts = []
        if intent.response:
            reply_parts.append(intent.response)
        if table_md:
            reply_parts.append(f"\n{table_md}")

        final_reply = "\n\n".join(reply_parts) if reply_parts else "Analysis operation completed successfully."
        return final_reply, True, extra

    else:  # general_chat
        return intent.response or "I'm not sure how to help with that.", False, extra

