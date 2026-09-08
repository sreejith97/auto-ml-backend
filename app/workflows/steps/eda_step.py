"""
app/workflows/steps/eda_step.py — Agno EDA Steps (Parallel + Synthesis)
========================================================================
Wraps the deterministic EDA functions from `app/agents/eda.py` as Agno
Workflow steps, plus an EDASynthesisAgent that narrates the findings
with full memory of the pipeline's prior narrations.

Architecture:
  Parallel:
    ├── Step(eda_stats_fn)       ← compute_summary_stats
    ├── Step(eda_correlation_fn) ← compute_correlation_matrix
    └── Step(eda_class_balance_fn) ← compute_class_balance (classification only)
  Step(eda_synthesis_agent)    ← synthesizes + narrates all 3 outputs

These step functions are also called directly from app/api/routes/eda.py
so the existing REST endpoint continues to work during the migration.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd

from app.core.llm import get_model, get_agent_storage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic EDA step functions (Parallel candidates)
# ---------------------------------------------------------------------------

def compute_eda_stats(df: pd.DataFrame, columns_spec: List[Dict]) -> Dict[str, Any]:
    """Wraps eda.compute_summary_stats for direct use."""
    from app.agents.eda import compute_summary_stats
    return compute_summary_stats(df, columns_spec)


def compute_eda_correlation(df: pd.DataFrame, columns_spec: List[Dict]) -> Dict[str, Any]:
    """Wraps eda.compute_correlation_matrix for direct use."""
    from app.agents.eda import compute_correlation_matrix
    return compute_correlation_matrix(df, columns_spec)


def compute_eda_class_balance(
    df: pd.DataFrame, target_col: str, task_type: str
) -> Optional[Dict[str, Any]]:
    """Wraps eda.compute_class_balance for direct use."""
    from app.agents.eda import compute_class_balance
    return compute_class_balance(df, target_col, task_type)


def compute_eda_findings(
    stats: Dict, correlation: Dict, class_balance: Optional[Dict]
) -> List[Dict]:
    """Deterministic findings from all three EDA outputs."""
    from app.agents.eda import generate_findings_summary
    return generate_findings_summary(stats, correlation, class_balance)


# ---------------------------------------------------------------------------
# EDASynthesisAgent — narrates findings with pipeline memory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def get_eda_synthesis_agent(run_id: int):
    """
    Returns a cached EDASynthesisAgent for a given run_id.
    Shares session_id with the NarrationAgent so it can reference prior stages.
    """
    from agno.agent import Agent

    return Agent(
        name="EDASynthesisAgent",
        model=get_model(),
        db=get_agent_storage("eda_sessions"),
        session_id=str(run_id),
        add_history_to_context=True,
        num_history_runs=4,
        description=(
            "You are an AI data analyst that synthesizes Exploratory Data Analysis "
            "findings and explains them to non-technical users. You remember what "
            "was discussed earlier in the pipeline."
        ),
        instructions=[
            "Summarize the key EDA findings in 2-4 plain English sentences.",
            "Highlight class imbalance, high correlation, or skew if present — these "
            "affect model quality.",
            "Never output raw numbers, JSON, or statistical jargon.",
            "If no notable findings exist, say so briefly and positively.",
            "Reference any relevant cleaning decisions made earlier if applicable.",
        ],
        markdown=False,
    )


async def synthesize_eda(
    run_id: int,
    conn,
    stats: Dict,
    correlation: Dict,
    class_balance: Optional[Dict],
    findings: List[Dict],
) -> str:
    """
    Runs the EDASynthesisAgent and stores the narration.
    Returns the narration string.
    """
    from app.core.queries import conversation

    if findings:
        findings_text = "\n".join([f"- {f['type']}: {f['detail']}" for f in findings])
        prompt = (
            f"EDA is complete. Notable findings:\n{findings_text}\n\n"
            "Summarize these insights for the user in 2-3 friendly, non-technical sentences."
        )
    else:
        prompt = (
            "EDA completed with no notable issues. "
            "Tell the user briefly and invite them to proceed to feature engineering."
        )

    try:
        agent = get_eda_synthesis_agent(run_id)
        result = await agent.arun(prompt)
        msg = result.content if isinstance(result.content, str) else str(result.content)
    except Exception as exc:
        logger.error(f"EDASynthesisAgent failed for run={run_id}: {exc}")
        msg = (
            f"EDA complete. Found {len(findings)} notable insights."
            if findings
            else "EDA complete. Data looks clean!"
        )

    await conversation.insert_message(conn, run_id, "assistant", msg, "eda")
    return msg


# ---------------------------------------------------------------------------
# All-in-one EDA runner — used by the existing REST route
# ---------------------------------------------------------------------------

async def run_eda(
    run_id: int,
    conn,
    df: pd.DataFrame,
    columns_spec: List[Dict],
    target_col: str,
    task_type: str,
) -> Dict[str, Any]:
    """
    Runs all EDA computations and synthesis narration.
    Returns the full EDA payload for the frontend.
    """
    stats = compute_eda_stats(df, columns_spec)
    correlation = compute_eda_correlation(df, columns_spec)
    class_balance = compute_eda_class_balance(df, target_col, task_type)
    findings = compute_eda_findings(stats, correlation, class_balance)

    narration = await synthesize_eda(run_id, conn, stats, correlation, class_balance, findings)

    return {
        "stats": stats,
        "correlation": correlation,
        "class_balance": class_balance,
        "findings": findings,
        "narration": narration,
    }
