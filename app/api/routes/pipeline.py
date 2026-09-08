"""
app/api/routes/pipeline.py — SSE streaming endpoint for the Agno workflow
=========================================================================
Provides a single persistent SSE connection per pipeline run that pushes
real-time workflow events to the frontend.

Endpoint:
    GET /api/v1/pipeline/{run_id}/stream?input=<optional_stage_input>

SSE Event Format:
    event: <event_type>
    data: <json_payload>

Event types emitted:
    workflow_started        → WorkflowStarted
    step_started            → StepStarted (includes step_name → agent name mapping)
    step_completed          → StepCompleted
    parallel_started        → ParallelExecutionStarted
    parallel_completed      → ParallelExecutionCompleted
    condition_started       → ConditionExecutionStarted
    condition_completed     → ConditionExecutionCompleted
    token                   → RunContent (agent token chunks)
    workflow_paused         → StepPausedEvent (HITL)
    workflow_completed      → WorkflowCompleted
    workflow_error          → WorkflowError
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.api.deps import get_current_user
from app.core.db import get_db

import asyncpg

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Step name → User-facing agent name mapping
# ---------------------------------------------------------------------------
STEP_AGENT_NAMES: dict[str, str] = {
    "schema_inference":          "SchemaInferenceAgent",
    "narrate_ingestion":         "NarrationAgent",
    "cleaning_analysis":         "CleaningAgent",
    "narrate_cleaning":          "NarrationAgent",
    "eda_stats":                 "EDAStatsAnalyzer",
    "eda_correlation":           "CorrelationAnalyzer",
    "eda_class_balance":         "ClassBalanceAnalyzer",
    "eda_synthesis":             "EDASynthesisAgent",
    "feature_proposals":         "FeatureEngineeringAgent",
    "narrate_features":          "NarrationAgent",
    "classification_training":   "ClassificationTrainingAgent",
    "regression_training":       "RegressionTrainingAgent",
    "narrate_results":           "NarrationAgent",
}


def _map_event(event) -> tuple[str, dict]:
    """Maps an Agno WorkflowRunEvent to an SSE (event_type, data) pair."""
    try:
        event_name = event.event if hasattr(event, "event") else type(event).__name__

        if "workflow_started" in event_name:
            return "workflow_started", {"message": "Pipeline started"}

        elif "step_started" in event_name:
            step_name = getattr(event, "step_name", "") or ""
            agent_name = STEP_AGENT_NAMES.get(step_name, step_name)
            return "step_started", {"step_name": step_name, "agent_name": agent_name}

        elif "step_completed" in event_name:
            step_name = getattr(event, "step_name", "") or ""
            return "step_completed", {"step_name": step_name}

        elif "parallel_execution_started" in event_name:
            return "parallel_started", {"message": "Running EDA in parallel..."}

        elif "parallel_execution_completed" in event_name:
            return "parallel_completed", {"message": "EDA computations complete"}

        elif "condition_execution_started" in event_name:
            return "condition_started", {"message": "Evaluating training branch..."}

        elif "condition_execution_completed" in event_name:
            return "condition_completed", {"message": "Training branch selected"}

        elif "step_paused" in event_name or "workflow_paused" in event_name:
            step_name = getattr(event, "paused_step_name", "cleaning_analysis")
            return "workflow_paused", {
                "paused_step": step_name,
                "message": "Workflow paused — awaiting your decisions",
            }

        elif "workflow_completed" in event_name:
            return "workflow_completed", {"message": "Pipeline complete!"}

        elif "workflow_error" in event_name or "step_error" in event_name:
            msg = getattr(event, "error", str(event))
            return "workflow_error", {"message": str(msg)}

        elif hasattr(event, "content") and isinstance(event.content, str) and event.content:
            # Token-level streaming from agent executor
            return "token", {"content": event.content}

        else:
            return "ping", {"event": event_name}

    except Exception as exc:
        logger.debug(f"Could not map event {event}: {exc}")
        return "ping", {}


async def _sse_event(event_type: str, data: dict) -> str:
    """Formats a single SSE event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------

@router.get("/{run_id}/stream")
async def stream_pipeline(
    run_id: int,
    input: str = Query(default="", description="Optional initial input for the workflow"),
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Opens a persistent SSE connection that streams Agno workflow events
    for the given pipeline run in real-time.

    The frontend connects once per pipeline run (or reconnects after HITL resume).
    The stream closes naturally when the workflow completes or pauses for HITL.
    """
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    async def event_generator() -> AsyncIterator[str]:
        try:
            from app.workflows.registry import get_or_create_workflow
            wf = get_or_create_workflow(run_id)

            # Send initial connected event
            yield await _sse_event("connected", {"run_id": run_id})

            # Run the workflow with full streaming
            async for event in wf.arun(
                input=input or f"Run AutoML pipeline for run_id={run_id}",
                stream=True,
                stream_events=True,
                stream_executor_events=True,
            ):
                event_type, data = _map_event(event)

                # Always emit the event
                yield await _sse_event(event_type, data)

                # Close after terminal events
                if event_type in ("workflow_completed", "workflow_paused", "workflow_error"):
                    return

        except Exception as exc:
            logger.error(f"SSE stream error for run={run_id}: {exc}")
            yield await _sse_event("workflow_error", {"message": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # Disable nginx buffering
        },
    )
