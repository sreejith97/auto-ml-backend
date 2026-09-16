from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Any, Dict
from datetime import datetime

class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    r2_connected: bool

class DatasetBase(BaseModel):
    name: str
    source_type: str
    r2_raw_path: str
    row_count: int

class DatasetCreate(DatasetBase):
    pass

class DatasetResponse(DatasetBase):
    id: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class PipelineRunBase(BaseModel):
    dataset_id: int
    task_type: str
    target_column: str
    time_column: Optional[str] = None
    forecast_horizon: Optional[int] = None

class PipelineRunCreate(PipelineRunBase):
    pass

class PipelineRunResponse(PipelineRunBase):
    id: int
    status: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class HelloResponse(BaseModel):
    message: str
    db_test: str
    r2_test: str
    dataset_id: Optional[int] = None
    run_id: Optional[int] = None

class ColumnSpecBase(BaseModel):
    column_name: str
    role: str
    semantic_type: str
    missing_meaning: Optional[str] = None
    notes: Optional[str] = None

class ColumnSpecCreate(ColumnSpecBase):
    pass

class ColumnSpecResponse(ColumnSpecBase):
    id: int
    dataset_id: int
    
class ColumnSpecUpsertResponse(BaseModel):
    status: str
    can_advance: bool

class ChatMessage(BaseModel):
    id: int
    role: str
    content: str
    stage: str
    related_question_id: Optional[int] = None
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class ChatRequest(BaseModel):
    message: str

class ChartSpec(BaseModel):
    chart_type: str # bar, line, scatter, pie, histogram
    title: str
    x_axis: str
    y_axis: str
    data: List[dict]

class ChatResponse(BaseModel):
    assistant_reply: str
    action_taken: bool
    pandas_code: Optional[str] = None
    table_markdown: Optional[str] = None
    chart_spec: Optional[ChartSpec] = None
    updated_state: Optional[dict] = None


class TaskCreate(BaseModel):
    task_type: str
    target_column: str
    time_column: Optional[str] = None
    forecast_horizon: Optional[int] = None

class TaskResponse(BaseModel):
    run_id: int
    status: str

class CleanRunResponse(BaseModel):
    auto_applied: list
    rule_reused: list
    pending: list

class QuestionAnswerRequest(BaseModel):
    question_id: int
    answer: str

class CleanStateResponse(BaseModel):
    auto_applied: list
    rule_reused: list
    pending: list
    resolved: list

class EDAStepResult(BaseModel):
    metrics: dict
    findings: str
    recommendations: List[str] = []

class EDAResponse(BaseModel):
    summary_stats: dict
    correlation_matrix: dict
    class_balance: Optional[dict] = None
    dataset_health: Optional[dict] = None
    target_analysis: Optional[dict] = None
    outlier_analysis: Optional[dict] = None
    feature_target_relations: Optional[dict] = None
    categorical_profiling: Optional[dict] = None
    findings_summary: list
    narration: str


