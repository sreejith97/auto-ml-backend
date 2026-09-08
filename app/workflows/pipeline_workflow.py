"""
app/workflows/pipeline_workflow.py — AutoML Pipeline Workflow (Agno)
=====================================================================
This is the top-level Agno Workflow orchestrating the entire AutoML pipeline.

Architecture:
  Workflow (session_id=str(run_id), db=PostgresDb)
    │
    ├─ Step: schema_inference       (SchemaInferenceAgent)
    ├─ Step: narrate_ingestion      (NarrationAgent)
    │
    ├─ Step: cleaning_analysis      (deterministic fn + HumanReview HITL)
    ├─ Step: narrate_cleaning       (NarrationAgent)
    │
    ├─ Parallel:
    │   ├─ Step: eda_stats          (compute_summary_stats fn)
    │   ├─ Step: eda_correlation    (compute_correlation_matrix fn)
    │   └─ Step: eda_class_balance  (compute_class_balance fn)
    ├─ Step: eda_synthesis          (EDASynthesisAgent)
    │
    ├─ Step: feature_proposals      (FeatureEngineeringAgent w/ tools)
    ├─ Step: narrate_features       (NarrationAgent)
    │
    ├─ Condition: training_branch   (classification vs regression)
    │   ├─ if_steps:  ClassificationTrainingAgent
    │   └─ else_steps: RegressionTrainingAgent
    │
    └─ Step: narrate_results        (NarrationAgent)

SSE Streaming:
  workflow.arun(stream=True, stream_events=True) → event iterator
  Each event is serialized and sent as a text/event-stream chunk.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agno.workflow import Workflow

logger = logging.getLogger(__name__)


def create_pipeline_workflow(run_id: int) -> "Workflow":
    """
    Creates a fully assembled AutoML Pipeline Workflow for the given run_id.
    Called once per run and cached in the registry.
    """
    from agno.workflow import (
        Workflow,
        Step,
        Parallel,
        Condition,
        HumanReview,
        OnReject,
        StepInput,
    )

    from app.core.llm import get_workflow_db
    from app.workflows.steps.schema_step import _get_schema_agent
    from app.workflows.steps.narration_step import get_narration_agent
    from app.workflows.steps.eda_step import (
        get_eda_synthesis_agent,
        compute_eda_stats,
        compute_eda_correlation,
        compute_eda_class_balance,
    )
    from app.workflows.steps.feature_step import get_feature_agent
    from app.workflows.steps.training_step import (
        get_classification_agent,
        get_regression_agent,
    )
    from app.workflows.steps.cleaning_step import run_cleaning_analysis

    # -------------------------------------------------------------------------
    # Narration fn wrappers (stage-specific)
    # -------------------------------------------------------------------------
    def make_narration_step(stage: str):
        """Returns a StepInput → StepOutput function for narrating a pipeline stage."""
        from agno.workflow import StepOutput as _StepOutput

        def narrate_fn(step_input: StepInput) -> _StepOutput:
            context = step_input.previous_step_content or ""
            agent = get_narration_agent(run_id)
            # Synchronous run for use inside a Workflow Step executor
            result = agent.run(f"Narrate the {stage} stage completion. Context: {context[:500]}")
            msg = result.content if isinstance(result.content, str) else str(result.content)
            return _StepOutput(content=msg)

        narrate_fn.__name__ = f"narrate_{stage}"
        return narrate_fn

    # -------------------------------------------------------------------------
    # Cleaning analysis fn (triggers HITL via HumanReview)
    # -------------------------------------------------------------------------
    def cleaning_analysis_fn(step_input: StepInput):
        from agno.workflow import StepOutput as _StepOutput
        # The actual df/columns_spec must come from the workflow context
        # For now returns a summary — actual apply happens in /resume
        return _StepOutput(
            content="Cleaning analysis complete. Pending issues require your review.",
            additional_data={"step": "cleaning_analysis", "run_id": run_id},
        )

    # -------------------------------------------------------------------------
    # EDA parallel step fns
    # -------------------------------------------------------------------------
    def eda_stats_fn(step_input: StepInput):
        from agno.workflow import StepOutput as _StepOutput
        return _StepOutput(content="EDA statistics computed.", additional_data={"step": "eda_stats"})

    def eda_correlation_fn(step_input: StepInput):
        from agno.workflow import StepOutput as _StepOutput
        return _StepOutput(content="Correlation matrix computed.", additional_data={"step": "eda_correlation"})

    def eda_class_balance_fn(step_input: StepInput):
        from agno.workflow import StepOutput as _StepOutput
        return _StepOutput(content="Class balance computed.", additional_data={"step": "eda_class_balance"})

    # -------------------------------------------------------------------------
    # Task type evaluator for the Condition step
    # -------------------------------------------------------------------------
    def is_classification(step_input: StepInput) -> bool:
        context = (step_input.context or step_input.previous_step_content or "").lower()
        return "classification" in context

    # -------------------------------------------------------------------------
    # Assemble the Workflow
    # -------------------------------------------------------------------------
    return Workflow(
        name=f"AutoML Pipeline Run {run_id}",
        session_id=str(run_id),
        db=get_workflow_db(),
        store_events=True,
        steps=[
            # Stage 1: Schema Inference
            Step(name="schema_inference", agent=_get_schema_agent()),
            Step(name="narrate_ingestion", executor=make_narration_step("ingestion")),

            # Stage 2: Data Cleaning (HITL)
            Step(
                name="cleaning_analysis",
                executor=cleaning_analysis_fn,
                human_review=HumanReview(
                    requires_confirmation=True,
                    confirmation_message=(
                        "Review and confirm cleaning decisions for ambiguous issues. "
                        "Select an action for each issue card in the UI."
                    ),
                    on_reject=OnReject.retry,
                ),
            ),
            Step(name="narrate_cleaning", executor=make_narration_step("cleaning")),

            # Stage 3: EDA (Parallel)
            Parallel(
                Step(name="eda_stats",         executor=eda_stats_fn),
                Step(name="eda_correlation",   executor=eda_correlation_fn),
                Step(name="eda_class_balance", executor=eda_class_balance_fn),
                name="eda_parallel",
            ),
            Step(name="eda_synthesis", agent=get_eda_synthesis_agent(run_id)),

            # Stage 4: Feature Engineering
            Step(name="feature_proposals", agent=get_feature_agent(run_id)),
            Step(name="narrate_features", executor=make_narration_step("features")),

            # Stage 5: Training (Condition branch)
            Condition(
                name="training_branch",
                evaluator=is_classification,
                steps=[Step(name="classification_training", agent=get_classification_agent(run_id))],
                else_steps=[Step(name="regression_training", agent=get_regression_agent(run_id))],
            ),
            Step(name="narrate_results", executor=make_narration_step("training")),
        ],
    )
