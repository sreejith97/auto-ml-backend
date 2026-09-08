"""
app/workflows/steps/chat_step.py — Agno PipelineChatAgent
==========================================================
Replaces the scattered llm.complete() calls in:
  - app/api/routes/chat.py (inline general-chat + intent parsing)
  - app/agents/nl_answer_parser.py (parse_answer + parse_chat_intent)

Key improvements over the old approach:
  - Pydantic `response_model=ChatIntent` → no more json.loads(), no markdown stripping
  - `session_id=str(run_id)` + `add_history_to_messages=True` → agent remembers the
    entire conversation so users can say "go back and change that"
  - Single coherent agent rather than two separate parse_ functions

Usage:
    from app.workflows.steps.chat_step import create_chat_agent, ChatIntent
    agent = create_chat_agent(run_id)
    result = await agent.arun(f"Context: {ctx}\n\nUser: {message}")
    intent: ChatIntent = result.content
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.core.llm import get_model, get_agent_storage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic response model — eliminates json.loads() entirely
# ---------------------------------------------------------------------------

class ChatIntent(BaseModel):
    intent: Literal[
        "update_column_role",
        "answer_cleaning_question",
        "general_chat",
    ] = Field(
        description=(
            "Classified intent of the user's message. "
            "Use 'update_column_role' if they want to change a column's role. "
            "Use 'answer_cleaning_question' if they are resolving a cleaning decision. "
            "Use 'general_chat' for any other question or comment."
        )
    )

    # update_column_role fields
    column_name: Optional[str] = Field(
        default=None,
        description="Exact column name to update (only for update_column_role intent)."
    )
    new_role: Optional[Literal["target", "feature", "ignore", "id"]] = Field(
        default=None,
        description="New role for the column (only for update_column_role intent)."
    )

    # answer_cleaning_question fields
    matched_option: Optional[str] = Field(
        default=None,
        description=(
            "The cleaning action chosen. One of: "
            "drop, impute_mean, impute_median, drop_rows, cap_iqr, lowercase_all, drop_duplicates, ignore."
        )
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the matched option (0.0–1.0)."
    )

    # general_chat / clarification
    response: Optional[str] = Field(
        default=None,
        description=(
            "A concise, plain-English reply to the user. "
            "Required for general_chat. Also used for clarification when confidence < 0.7."
        )
    )


# ---------------------------------------------------------------------------
# Agent factory — cached per run_id for persistent memory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def create_chat_agent(run_id: int):
    """
    Returns (and caches) a PipelineChatAgent for a given run_id.
    Uses the same session_id as the NarrationAgent so the chat agent
    can naturally reference what happened in earlier pipeline stages.
    """
    from agno.agent import Agent

    return Agent(
        name="PipelineChatAgent",
        model=get_model(),
        db=get_agent_storage("chat_sessions"),
        session_id=str(run_id),
        add_history_to_context=True,
        num_history_runs=10,
        output_schema=ChatIntent,
        structured_outputs=True,
        description=(
            "You are an intelligent assistant embedded in an AutoML pipeline. "
            "You help non-technical users understand their data, correct column roles, "
            "and make cleaning decisions. You remember everything said in this session."
        ),
        instructions=[
            "Classify every user message into one of three intents: "
            "update_column_role | answer_cleaning_question | general_chat.",
            "For update_column_role: extract the exact column name and new role.",
            "For answer_cleaning_question: map the user's answer to one of the allowed "
            "cleaning actions (drop, impute_mean, impute_median, drop_rows, cap_iqr, "
            "lowercase_all, drop_duplicates, ignore). Set confidence < 0.7 if unsure "
            "and provide a clarifying response.",
            "For general_chat: answer concisely using the provided dataset context. "
            "Never expose raw JSON or internal pipeline identifiers.",
            "Always keep responses under 3 sentences unless more detail is explicitly asked.",
        ],
    )
