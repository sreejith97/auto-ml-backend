"""
app/workflows/steps/training_step.py — Agno Training Condition Agents
======================================================================
Creates classification and regression training agents for use in the
Agno Workflow Condition step. These agents orchestrate the existing
deterministic training logic from `app/agents/training.py`.

The actual model fitting is deterministic (scikit-learn). The agents
provide narration and can be extended to tune hyperparameters via tools.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.core.llm import get_model, get_agent_storage

logger = logging.getLogger(__name__)


@lru_cache(maxsize=64)
def get_classification_agent(run_id: int):
    from agno.agent import Agent
    from app.mcp_servers.factory import get_mcp_toolkit

    training_mcp = get_mcp_toolkit("training")
    tools = [training_mcp] if training_mcp else []

    return Agent(
        name="ClassificationTrainingAgent",
        model=get_model(),
        tools=tools,
        db=get_agent_storage("training_sessions"),
        session_id=str(run_id),
        add_history_to_context=True,
        description=(
            "You coordinate the classification model training stage. "
            "You use FastMCP training tools to compute diagnostics and recommend hyperparameter grids."
        ),
        instructions=[
            "Summarize the classification results in 2 plain-English sentences.",
            "Mention the best model name and its score without technical jargon.",
            "Reference feature engineering decisions from earlier in the session if relevant.",
        ],
        markdown=False,
    )


@lru_cache(maxsize=64)
def get_regression_agent(run_id: int):
    from agno.agent import Agent
    from app.mcp_servers.factory import get_mcp_toolkit

    training_mcp = get_mcp_toolkit("training")
    tools = [training_mcp] if training_mcp else []

    return Agent(
        name="RegressionTrainingAgent",
        model=get_model(),
        tools=tools,
        db=get_agent_storage("training_sessions"),
        session_id=str(run_id),
        add_history_to_context=True,
        description=(
            "You coordinate the regression model training stage. "
            "You use FastMCP training tools to compute diagnostics and recommend hyperparameter grids."
        ),
        instructions=[
            "Summarize the regression results in 2 plain-English sentences.",
            "Mention the best model name and its R² score without technical jargon.",
            "Reference feature engineering decisions from earlier in the session if relevant.",
        ],
        markdown=False,
    )
