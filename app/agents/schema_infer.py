"""
app/agents/schema_infer.py — Schema Inference (Agno-backed)
============================================================
This module is now a thin shim over the Agno-powered implementation in
`app/workflows/steps/schema_step.py`.

The public function `infer_schema(df)` is unchanged so that the existing
ingest route (`app/api/routes/ingest.py`) requires no modifications in Phase 3.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


async def infer_schema(df: pd.DataFrame) -> list:
    """
    Infer column schema using the Agno SchemaInferenceAgent.
    Returns a list of dicts:
      [{"column_name": ..., "role": ..., "semantic_type": ..., "notes": ...}]
    """
    from app.workflows.steps.schema_step import infer_schema_agno
    return await infer_schema_agno(df)
