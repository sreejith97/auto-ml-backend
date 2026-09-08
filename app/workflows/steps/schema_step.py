"""
app/workflows/steps/schema_step.py — Agno SchemaInferenceAgent
==============================================================
Replaces the brittle `json.loads()` approach in schema_infer.py with an
Agno Agent that uses a Pydantic `response_model` for guaranteed structured output.

The agent object is module-level (singleton) for performance — Agno reuses the
underlying model client across calls. Call `infer_schema_agno(df)` from routes.
"""

from __future__ import annotations

import json
import logging
from typing import List, Literal

import pandas as pd
from pydantic import BaseModel, Field

from app.core.llm import get_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic response model — no more json.loads()
# ---------------------------------------------------------------------------

class ColumnSpec(BaseModel):
    column_name: str = Field(description="Exact column name as it appears in the dataset.")
    role: Literal["feature", "target", "id", "ignore"] = Field(
        description=(
            "Role of the column. Only use 'target' if it is OBVIOUSLY the prediction "
            "target (e.g. price, is_churned, survival). Use 'id' for primary keys. "
            "Use 'ignore' for fully null or useless columns. Default to 'feature'."
        )
    )
    semantic_type: Literal["numeric", "categorical", "datetime", "text", "boolean"] = Field(
        description="The semantic data type of the column's values."
    )
    notes: str = Field(
        description="A 1-sentence, plain-English description for a non-technical user."
    )


class SchemaResult(BaseModel):
    columns: List[ColumnSpec] = Field(
        description="Schema specification for every column in the dataset."
    )


# ---------------------------------------------------------------------------
# Agent (module-level singleton)
# ---------------------------------------------------------------------------

schema_agent = None  # lazy-initialized below


def _get_schema_agent():
    """Lazy-initialise the schema agent (avoids import-time model client creation)."""
    global schema_agent
    if schema_agent is None:
        from agno.agent import Agent
        schema_agent = Agent(
            name="SchemaInferenceAgent",
            model=get_model(),
            instructions=(
                "Analyze the columns of the uploaded dataset and assign role, semantic_type, and notes to each.\n"
                "Roles: 'target' (only ONE primary target column if obvious, else 'feature'), "
                "'feature', 'ignore' (high-cardinality IDs, constant values), 'id'.\n"
                "Semantic types: 'numeric', 'categorical', 'datetime', 'text'."
            ),
            output_schema=SchemaResult,
            structured_outputs=True,
            markdown=False,
        )
    return schema_agent


# ---------------------------------------------------------------------------
# Public API — drop-in async replacement for schema_infer.infer_schema()
# ---------------------------------------------------------------------------

def _build_prompt(df: pd.DataFrame) -> str:
    sample_data = df.head(5).to_dict(orient="list")
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    return (
        f"Column dtypes:\n{json.dumps(dtypes, indent=2)}\n\n"
        f"First 5 rows sample:\n{json.dumps(sample_data, indent=2)}\n\n"
        "Infer the schema for every column listed above."
    )


async def infer_schema_agno(df: pd.DataFrame) -> list[dict]:
    """
    Agno-powered schema inference with Pydantic structured output.
    Returns a list of dicts compatible with the existing schema store.
    Falls back to heuristics if the agent call fails.
    """
    try:
        agent = _get_schema_agent()
        prompt = _build_prompt(df)
        result: SchemaResult = await agent.arun(prompt)

        if isinstance(result.content, SchemaResult):
            schema = result.content
        else:
            # Agno may wrap in RunResponse.content
            schema = result.content if isinstance(result.content, SchemaResult) else result

        return [col.model_dump() for col in schema.columns]

    except Exception as exc:
        logger.error(f"SchemaInferenceAgent failed: {exc}. Falling back to heuristics.")
        return _heuristic_fallback(df)


def _heuristic_fallback(df: pd.DataFrame) -> list[dict]:
    """Simple fallback if the agent call fails entirely."""
    return [
        {
            "column_name": col,
            "role": "feature",
            "semantic_type": "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "text",
            "notes": f"The {col} column.",
        }
        for col in df.columns
    ]
