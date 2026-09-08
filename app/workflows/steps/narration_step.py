"""
app/workflows/steps/narration_step.py — Agno NarrationAgent
============================================================
Replaces the six standalone `narrate_*()` functions in narration.py with a
single memory-aware Agno Agent that:
  - Remembers previous narrations (add_history_to_messages=True) to avoid
    repetitive phrasing across pipeline stages
  - Stores conversation history in Postgres via agent_storage
  - Uses session_id=str(run_id) for shared memory with the chat agent

Usage:
    from app.workflows.steps.narration_step import get_narration_agent

    agent = get_narration_agent(run_id)
    result = await agent.arun("Narrate the EDA stage completion. ...")
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.core.llm import get_model, get_agent_storage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent factory — one agent per run_id so history is isolated per pipeline
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def get_narration_agent(run_id: int):
    """
    Returns (and caches) a NarrationAgent for a given run_id.
    The agent uses Postgres storage for persistent cross-stage memory.
    """
    from agno.agent import Agent

    return Agent(
        name="NarrationAgent",
        model=get_model(),
        db=get_agent_storage("narration_sessions"),
        session_id=str(run_id),
        add_history_to_context=True,
        num_history_runs=6,          # last 6 narrations kept in context
        description=(
            "You are a friendly AI assistant narrating the progress of an AutoML "
            "pipeline to a non-technical user. You remember what you said earlier "
            "and avoid being repetitive."
        ),
        instructions=[
            "Keep responses short: 1-3 encouraging sentences in plain English.",
            "Never expose internal JSON, model names, or technical pipeline details.",
            "Refer back to earlier stages naturally (e.g. 'now that we've cleaned...') "
            "to give the user a sense of progress.",
            "Always end with a clear next-step invitation.",
        ],
        markdown=False,
    )


# ---------------------------------------------------------------------------
# Stage-specific narration helpers
# These are drop-in async replacements for the functions in narration.py.
# Each inserts the narration message into the conversation history via DB.
# ---------------------------------------------------------------------------

async def narrate_stage(
    conn,
    run_id: int,
    stage: str,
    prompt: str,
    fallback: str,
) -> str:
    """Core narration helper — delegates to agent and stores result."""
    from app.core.queries import conversation

    try:
        agent = get_narration_agent(run_id)
        result = await agent.arun(prompt)
        # Extract string content from Agno RunResponse
        msg = result.content if isinstance(result.content, str) else str(result.content)
    except Exception as exc:
        logger.error(f"NarrationAgent failed for stage={stage} run={run_id}: {exc}")
        msg = fallback

    await conversation.insert_message(conn, run_id, "assistant", msg, stage)
    return msg


async def narrate_ingestion(
    conn, run_id: int, dataset_name: str, row_count: int, schema_list: list | None = None
) -> str:
    target_cols = [c["column_name"] for c in (schema_list or []) if c["role"] == "target"]
    target_str = (
        f"I think the target variable to predict is '{target_cols[0]}'."
        if target_cols
        else "I couldn't find an obvious target variable — please review."
    )
    prompt = (
        f"The dataset '{dataset_name}' ({row_count} rows, "
        f"{len(schema_list or [])} columns) was just ingested. "
        f"{target_str} "
        "Narrate this success and invite the user to review the column roles."
    )
    fallback = (
        f"I've successfully ingested '{dataset_name}' ({row_count} rows). "
        "Please review the inferred column roles and correct me if needed."
    )
    return await narrate_stage(conn, run_id, "ingestion", prompt, fallback)


async def narrate_columns(conn, run_id: int, target_col: str) -> str:
    prompt = (
        f"The user just finished specifying column metadata. "
        f"The target variable to predict is '{target_col}'. "
        "Confirm this and invite them to define the ML task next."
    )
    fallback = (
        f"Column specs saved! We'll be predicting '{target_col}'. "
        "Ready to define the ML task."
    )
    return await narrate_stage(conn, run_id, "columns", prompt, fallback)


async def narrate_task(conn, run_id: int, task_type: str, target_col: str) -> str:
    prompt = (
        f"The user locked in the ML task as '{task_type}' predicting '{target_col}'. "
        "Confirm this enthusiastically and invite them to start the Data Cleaning stage."
    )
    fallback = (
        f"Task locked: {task_type} on '{target_col}'. Let's clean the data!"
    )
    return await narrate_stage(conn, run_id, "task", prompt, fallback)


async def narrate_cleaning(conn, run_id: int, auto_applied: int, pending: int) -> str:
    if pending > 0:
        prompt = (
            f"Cleaning analysis done. I auto-applied {auto_applied} obvious fixes, "
            f"but found {pending} ambiguous issues requiring the user's judgment. "
            "Ask them to review the issue cards and make a decision for each one."
        )
        fallback = (
            f"I applied {auto_applied} automatic fixes. "
            f"There are {pending} issues that need your decision."
        )
    else:
        prompt = (
            f"Cleaning complete — I auto-applied {auto_applied} fixes with no ambiguity. "
            "Invite the user to proceed to EDA."
        )
        fallback = f"All cleaned! Applied {auto_applied} fixes automatically."
    return await narrate_stage(conn, run_id, "cleaning", prompt, fallback)


async def narrate_eda(conn, run_id: int, findings: list) -> str:
    if findings:
        findings_text = "\n".join([f"- {f['type']}: {f['detail']}" for f in findings])
        prompt = (
            f"EDA is complete. Notable findings:\n{findings_text}\n"
            "Summarize these insights in a friendly, non-technical sentence or two."
        )
        fallback = f"EDA done! Key findings: {findings_text[:200]}"
    else:
        prompt = "EDA is complete with nothing highly notable to report. Tell the user."
        fallback = "Exploratory analysis complete — data looks standard."
    return await narrate_stage(conn, run_id, "eda", prompt, fallback)


async def narrate_features(conn, run_id: int, proposals_list: list) -> str:
    count = len(proposals_list)
    prompt = (
        f"Feature engineering proposed {count} new features based on your data. "
        "Invite the user to review and approve them."
    )
    fallback = f"I've proposed {count} new features. Please review them!"
    return await narrate_stage(conn, run_id, "features", prompt, fallback)


async def narrate_results(conn, run_id: int, top_model: str, accuracy: float) -> str:
    prompt = (
        f"Model training is done! The best model is {top_model} "
        f"with a score of {accuracy:.4f}. "
        "Congratulate the user warmly and point them to the results charts."
    )
    fallback = (
        f"Training complete! Best model: {top_model} (score: {accuracy:.4f}). "
        "Check the results page!"
    )
    return await narrate_stage(conn, run_id, "training", prompt, fallback)
