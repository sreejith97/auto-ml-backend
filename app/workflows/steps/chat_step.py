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
        "data_analysis_and_op",
        "general_chat",
    ] = Field(
        description=(
            "Classified intent of the user's message. "
            "Use 'update_column_role' if they want to change a column's role. "
            "Use 'answer_cleaning_question' if they are resolving a cleaning decision. "
            "Use 'data_analysis_and_op' if they ask to analyze data, run pandas operations, filter rows, compute statistics, or generate charts/plots. "
            "Use 'general_chat' for general questions or pipeline status comments."
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

    # data_analysis_and_op fields
    pandas_code: Optional[str] = Field(
        default=None,
        description="Valid Pandas code snippet operating on DataFrame 'df' (e.g. df.groupby('col')['target'].mean().reset_index() or df['new_col'] = ...)."
    )
    is_mutation: bool = Field(
        default=False,
        description="Set to True if pandas_code modifies df rows/columns or structure, False if it is read-only analysis."
    )
    requires_chart: bool = Field(
        default=False,
        description="Set to True if the user asks for a chart/plot/graph or if a visual plot would enhance the analysis."
    )
    chart_type: Optional[Literal["bar", "line", "scatter", "pie", "histogram"]] = Field(
        default=None,
        description="Chart type requested or best suited: bar, line, scatter, pie, histogram."
    )
    chart_title: Optional[str] = Field(
        default=None,
        description="Title for the chart."
    )
    x_axis: Optional[str] = Field(
        default=None,
        description="DataFrame column name for x-axis."
    )
    y_axis: Optional[str] = Field(
        default=None,
        description="DataFrame column name for y-axis."
    )

    # general_chat / text analysis summary
    response: Optional[str] = Field(
        default=None,
        description=(
            "A concise, plain-English text analysis, explanation, or general response to the user. "
            "Provide insightful commentary on data distributions or trends when performing analysis."
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
            "You are an intelligent data analyst and assistant embedded in an AutoML pipeline. "
            "You help users analyze data, write sandboxed Pandas code, generate visual charts, "
            "correct column roles, and make cleaning decisions."
        ),
        instructions=[
            "Classify user messages into one of four intents: "
            "update_column_role | answer_cleaning_question | data_analysis_and_op | general_chat.",
            "For data_analysis_and_op: write clean, executable Pandas code operating on 'df'.",
            "If the user asks to create columns, filter rows, or mutate data, set is_mutation=True.",
            "If the user asks for a plot, chart, histogram, or graph, set requires_chart=True and specify chart_type, chart_title, x_axis, y_axis.",
            "If the user enters unrelated, off-topic, or nonsensical inputs (e.g. referencing columns not in the dataset or general trivia), classify as general_chat or return a polite guidance response in 'response' indicating the available dataset columns and suggesting valid actions.",
            "Always include insightful, plain-English text commentary in the 'response' field explaining statistical findings or operational summaries.",
        ],
    )

