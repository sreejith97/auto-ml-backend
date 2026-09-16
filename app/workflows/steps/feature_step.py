"""
app/workflows/steps/feature_step.py — Agno FeatureEngineeringAgent
===================================================================
Wraps the deterministic `propose_transform()` rule engine as an Agno tool,
then uses a tool-using Agent that can read EDA findings from session memory
to make smarter feature proposals (e.g. "I saw high skew during EDA...").

The agent shares session_id=str(run_id) with the NarrationAgent and
EDASynthesisAgent — so it can naturally reference "what I found during EDA"
without any extra context injection.

Usage (from features route):
    from app.workflows.steps.feature_step import propose_features_agno
    proposals = await propose_features_agno(run_id, conn, df, columns_spec)
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
# Agno tool wrapping the deterministic rule engine
# ---------------------------------------------------------------------------

def _make_feature_tools():
    """Build Agno tool functions for the feature engineering rule engine."""
    from agno.tools import tool

    @tool
    def propose_column_transform(
        column_name: str,
        role: str,
        semantic_type: str,
        is_numeric: bool,
        nunique: int,
        skew: float,
        has_non_positive: bool,
    ) -> Dict[str, Any]:
        """
        Proposes the optimal feature transformation for a single column using
        the deterministic rule engine. Returns a proposal dict with keys:
        column_name, semantic_type, proposed_transform, params, rationale.

        Args:
            column_name: Name of the column.
            role: One of feature/target/id/ignore.
            semantic_type: One of numeric/categorical/datetime/text/boolean.
            is_numeric: True if the column dtype is numeric.
            nunique: Number of unique values.
            skew: Skewness of the column (numeric only, 0.0 otherwise).
            has_non_positive: True if column contains values <= 0 (numeric only).
        """
        col_meta = {"role": role, "semantic_type": semantic_type}

        # Skip non-feature columns
        if role in ("id", "ignore", "target"):
            return {"column_name": column_name, "proposed_transform": "skip", "rationale": f"Role is '{role}' — skipped."}

        if semantic_type in ("text", "free_text"):
            return {"column_name": column_name, "semantic_type": semantic_type, "proposed_transform": "manual_handling",
                    "rationale": "Text columns require manual NLP handling (TF-IDF or embeddings)."}

        if semantic_type == "boolean":
            return {"column_name": column_name, "semantic_type": "boolean", "proposed_transform": "passthrough",
                    "rationale": "Boolean — already model-ready."}

        if semantic_type == "datetime":
            return {"column_name": column_name, "semantic_type": "datetime", "proposed_transform": "datetime_decompose",
                    "rationale": "Decompose into year/month/day/day_of_week for temporal patterns."}

        if semantic_type == "categorical" or not is_numeric:
            if nunique <= 10:
                return {"column_name": column_name, "semantic_type": "categorical", "proposed_transform": "one_hot_encode",
                        "rationale": f"Low cardinality ({nunique} ≤ 10) — OneHotEncoding optimal."}
            else:
                return {"column_name": column_name, "semantic_type": "categorical", "proposed_transform": "ordinal_encode",
                        "rationale": f"High cardinality ({nunique} > 10) — OrdinalEncoding avoids sparse columns."}

        # Numeric
        if abs(skew) > 1.0:
            if has_non_positive:
                return {"column_name": column_name, "semantic_type": "numeric", "proposed_transform": "standard_scale",
                        "rationale": f"High skew ({skew:.2f}) but contains non-positive values — standard scaling only."}
            else:
                return {"column_name": column_name, "semantic_type": "numeric", "proposed_transform": "log1p_standard_scale",
                        "rationale": f"High skew ({skew:.2f}) — log1p + standard scaling proposed."}
        else:
            return {"column_name": column_name, "semantic_type": "numeric", "proposed_transform": "standard_scale",
                    "rationale": "Normal variance — standard scaling to center the data."}

    return [propose_column_transform]


# ---------------------------------------------------------------------------
# FeatureEngineeringAgent — tool-using, EDA-aware
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def get_feature_agent(run_id: int):
    """Returns (and caches) a FeatureEngineeringAgent for a run_id."""
    from agno.agent import Agent
    from app.mcp_servers.factory import get_mcp_toolkit
    feature_mcp = get_mcp_toolkit("feature")
    tools = _make_feature_tools()
    if feature_mcp:
        tools.append(feature_mcp)

    return Agent(
        name="FeatureEngineeringAgent",
        model=get_model(),
        tools=tools,
        db=get_agent_storage("feature_sessions"),
        session_id=str(run_id),
        add_history_to_context=True,
        num_history_runs=4,
        description=(
            "You are an expert ML feature engineer. You use the propose_column_transform "
            "tool to decide the optimal transformation for each feature column. "
            "You remember EDA findings from earlier in this pipeline session and can "
            "override rule-engine defaults when you spot something unusual."
        ),
        instructions=[
            "For every feature column provided, call propose_column_transform with the "
            "correct arguments to get the transformation proposal.",
            "If EDA found high skew in a column, ensure the proposal reflects that.",
            "Skip columns with role in ['id', 'ignore', 'target'].",
            "Return ONLY the proposals from the tool — do not add commentary.",
        ],
        markdown=False,
    )


# ---------------------------------------------------------------------------
# Direct deterministic proposal (used as fast path / fallback)
# ---------------------------------------------------------------------------

def _deterministic_proposals(df: pd.DataFrame, columns_spec: List[Dict]) -> List[Dict]:
    """Fast path: deterministic proposals without agent overhead."""
    from app.agents.feature_engineering import propose_transform

    proposals = []
    for col_meta in columns_spec:
        col = col_meta.get("column_name")
        if col not in df.columns:
            continue
        proposal = propose_transform(col, col_meta, df[col])
        if proposal:
            proposals.append(proposal)
    return proposals


# ---------------------------------------------------------------------------
# Public API — used by features route
# ---------------------------------------------------------------------------

async def propose_features_agno(
    run_id: int,
    conn,
    df: pd.DataFrame,
    columns_spec: List[Dict],
) -> List[Dict]:
    """
    Proposes feature transformations using the Agno FeatureEngineeringAgent.
    Falls back to direct deterministic proposals if the agent fails.
    """
    # Build per-column summaries for the agent prompt
    col_summaries = []
    for col_meta in columns_spec:
        col = col_meta.get("column_name")
        role = col_meta.get("role", "feature")
        if role in ("id", "ignore", "target") or col not in df.columns:
            continue

        series = df[col]
        try:
            skew = float(series.skew()) if pd.api.types.is_numeric_dtype(series) else 0.0
        except Exception:
            skew = 0.0

        col_summaries.append({
            "column_name": col,
            "role": role,
            "semantic_type": col_meta.get("semantic_type", "numeric"),
            "is_numeric": bool(pd.api.types.is_numeric_dtype(series)),
            "nunique": int(series.nunique()),
            "skew": round(skew, 3),
            "has_non_positive": bool((series <= 0).any()) if pd.api.types.is_numeric_dtype(series) else False,
        })

    if not col_summaries:
        return []

    prompt = (
        f"Propose feature transformations for the following {len(col_summaries)} columns. "
        f"Call propose_column_transform for each one:\n"
        + json.dumps(col_summaries, indent=2)
    )

    try:
        agent = get_feature_agent(run_id)
        result = await agent.arun(prompt)
        # The agent's tool calls already produced the proposals as content
        # Parse JSON array from content if possible, else fall back
        content = result.content
        if isinstance(content, list):
            return content
        if isinstance(content, str):
            import re
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                return json.loads(match.group())
        # If parsing fails, fall through to deterministic
        raise ValueError("Could not parse agent feature proposals")

    except Exception as exc:
        logger.warning(
            f"FeatureEngineeringAgent failed for run={run_id}: {exc}. "
            "Falling back to deterministic proposals."
        )
        return _deterministic_proposals(df, columns_spec)
