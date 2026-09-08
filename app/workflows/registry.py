"""
app/workflows/registry.py — In-process Workflow instance cache
=============================================================
Workflow objects are expensive to create (they connect to PostgresDb).
This module provides a simple per-process cache keyed by run_id so that
the SSE stream endpoint and the HITL /resume endpoint share the exact
same Workflow instance (and its in-memory state).

Usage:
    from app.workflows.registry import get_or_create_workflow
    wf = get_or_create_workflow(run_id)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agno.workflow import Workflow

logger = logging.getLogger(__name__)

# run_id → Workflow instance
_workflows: dict[int, "Workflow"] = {}


def get_or_create_workflow(run_id: int) -> "Workflow":
    """
    Return the cached Workflow for this run, creating one if needed.
    The workflow is created fresh on first access and reused for all
    subsequent requests (SSE stream, HITL resume, status polling).
    """
    if run_id not in _workflows:
        # Lazy import to avoid circular imports at module load time
        from app.workflows.pipeline_workflow import create_pipeline_workflow
        logger.info(f"Creating new Workflow instance for run_id={run_id}")
        _workflows[run_id] = create_pipeline_workflow(run_id)
    return _workflows[run_id]


def evict_workflow(run_id: int) -> None:
    """Remove a completed or errored workflow from the cache."""
    _workflows.pop(run_id, None)
    logger.info(f"Evicted Workflow for run_id={run_id}")
