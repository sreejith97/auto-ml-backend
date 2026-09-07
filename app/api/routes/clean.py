from fastapi import APIRouter, Depends, HTTPException
import asyncpg
import pandas as pd
from app.core.db import get_db
from app.core.queries import columns as col_queries
from app.core.queries import rules as rule_queries
from app.core.queries import questions as q_queries
from app.models.schemas import CleanRunResponse, QuestionAnswerRequest, CleanStateResponse
from app.agents import cleaning, confidence, narration
from app.api.deps import get_current_user
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/{run_id}/run", response_model=CleanRunResponse)
async def run_cleaning(
    run_id: int, 
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    # 1. Fetch run info
    run_info = await db.fetchrow("SELECT dataset_id, user_id FROM pipeline_runs WHERE id = $1", run_id)
    if not run_info:
        raise HTTPException(status_code=404, detail="Run not found")
    if run_info['user_id'] != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    dataset_id = run_info['dataset_id']
    
    # Prevent duplication: if we already generated questions for this run, skip re-running detectors
    existing = await db.fetchval("SELECT COUNT(*) FROM questions_pending WHERE run_id = $1", run_id)
    if existing > 0:
        state = await get_clean_state(run_id, db, current_user)
        return CleanRunResponse(auto_applied=[], pending=state.pending)
    
    from app.core.storage import load_dataframe
    # Load dataset
    try:
        df = await load_dataframe(db, run_id)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")
        
    cols_spec = await col_queries.get_columns_for_dataset(db, dataset_id)
    
    auto_applied = []
    pending_list = []
    
    # 2. Run Detectors
    for col_record in cols_spec:
        col_name = col_record['column_name']
        if col_name not in df.columns:
            continue
            
        col_meta = dict(col_record)
        dtype_str = str(df[col_name].dtype)
        null_pct = df[col_name].isnull().sum() / len(df)
        
        # Calculate Signature
        sig = confidence.column_signature(col_name, dtype_str, null_pct)
        
        # Run the column-level detectors
        detectors = [
            cleaning.detect_missing(df, col_name, col_meta),
            cleaning.detect_outliers(df, col_name, col_meta),
            cleaning.detect_format_inconsistency(df, col_name)
        ]
        
        for det in detectors:
            if not det:
                continue
                
            # Check for existing rule
            matched_rule = await rule_queries.find_matching_rule(db, sig)
            
            if matched_rule and matched_rule['transformation']:
                # Reuse rule
                await rule_queries.increment_rule_usage(db, matched_rule['id'])
                auto_applied.append({
                    "column": col_name,
                    "issue": det["issue"],
                    "confidence": 1.0,
                    "source": "rule",
                    "action": matched_rule['transformation']
                })
            elif det["confidence"] >= confidence.CONFIDENCE_THRESHOLD:
                # High confidence -> Auto apply
                auto_applied.append({
                    "column": col_name,
                    "issue": det["issue"],
                    "confidence": det["confidence"],
                    "source": "detector",
                    "action": det["proposed_fix"]
                })
            else:
                # Low confidence -> Ask human
                q_id = await q_queries.insert_pending(
                    db, run_id, col_name, det["issue"], det["confidence"], det["evidence"]
                )
                pending_list.append({
                    "id": q_id,
                    "column": col_name,
                    "issue": det["issue"],
                    "confidence": det["confidence"]
                })
                
    # Run Dataset-level detectors
    dup_det = cleaning.detect_duplicates(df)
    if dup_det:
        if dup_det["confidence"] >= confidence.CONFIDENCE_THRESHOLD:
            auto_applied.append({
                "column": "Entire Dataset",
                "issue": dup_det["issue"],
                "confidence": dup_det["confidence"],
                "source": "detector",
                "action": dup_det["proposed_fix"]
            })
        else:
            q_id = await q_queries.insert_pending(
                db, run_id, "Entire Dataset", dup_det["issue"], dup_det["confidence"], dup_det["evidence"]
            )
            pending_list.append({
                "id": q_id,
                "column": "Entire Dataset",
                "issue": dup_det["issue"],
                "confidence": dup_det["confidence"]
            })
    
    # 3. Save snapshot (For run_cleaning, we just return the pending state, but wait, the previous code saved dummy_dataset_cleaned.csv here. I will just skip saving here because finalize handles it)
    # df.to_csv("dummy_dataset_cleaned.csv", index=False)
    
    # 4. Narrate
    await narration.narrate_cleaning(db, run_id, len(auto_applied), len(pending_list))
    
    return CleanRunResponse(auto_applied=auto_applied, pending=pending_list)

async def answer_question(run_id: int, question_id: int, answer: str, db: asyncpg.Connection):
    """Extracted core logic so chat.py can call it directly."""
    # Resolve the question
    await q_queries.resolve_question(db, question_id, answer)
    
    # Get question info to save the rule
    q = await db.fetchrow("SELECT * FROM questions_pending WHERE id = $1", question_id)
    if not q:
        return
        
    col_name = q['column_name']
    
    from app.core.storage import load_dataframe
    df = await load_dataframe(db, run_id)
    dtype_str = str(df[col_name].dtype)
    null_pct = df[col_name].isnull().sum() / len(df)
    sig = confidence.column_signature(col_name, dtype_str, null_pct)
    
    # Transformation mapping (stubbed)
    transformation = f"apply_{answer}"
    
    # Save Reusable Rule!
    await rule_queries.upsert_rule(db, sig, f"How to handle {q['issue_type']} in {col_name}?", answer, transformation)
    
    # Log narration confirming action
    msg = f"I have applied '{answer}' to resolve the issue in '{col_name}'. This rule has been saved for future datasets with similar structures."
    from app.core.queries import conversation
    await conversation.insert_message(db, run_id, 'assistant', msg, 'cleaning')

@router.post("/{run_id}/answer", response_model=dict)
async def post_answer(
    run_id: int, 
    req: QuestionAnswerRequest, 
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    await answer_question(run_id, req.question_id, req.answer, db)
    return {"status": "success"}

@router.get("/{run_id}", response_model=CleanStateResponse)
async def get_clean_state(
    run_id: int, 
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    pending = await q_queries.get_pending(db, run_id)
    resolved_records = await q_queries.get_resolved(db, run_id)
    
    # Extract resolved answer from evidence_jsonb
    resolved = []
    for r in resolved_records:
        evidence = json.loads(r['evidence_jsonb']) if r['evidence_jsonb'] else {}
        action = evidence.get("resolved_answer", "Unknown")
        resolved.append({
            "id": r["id"],
            "column": r["column_name"],
            "issue": r["issue_type"],
            "action": action,
            "confidence": r["confidence"]
        })
        
    return CleanStateResponse(
        auto_applied=[], 
        pending=[{
            "id": p["id"],
            "column": p["column_name"],
            "issue": p["issue_type"],
            "confidence": p["confidence"]
        } for p in pending],
        resolved=resolved
    )

@router.post("/{run_id}/finalize", response_model=dict)
async def finalize_cleaning(
    run_id: int, 
    db: asyncpg.Connection = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    owner = await db.fetchval("SELECT user_id FROM pipeline_runs WHERE id = $1", run_id)
    if owner != current_user['id']:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    from app.core.storage import load_dataframe, save_snapshot
    # 1. Load dataset
    df = await load_dataframe(db, run_id)
    original_rows = len(df)
    values_imputed = 0
    values_capped = 0
    
    # 2. Fetch all resolved rules
    resolved_records = await q_queries.get_resolved(db, run_id)
    
    for r in resolved_records:
        evidence = json.loads(r['evidence_jsonb']) if r['evidence_jsonb'] else {}
        action = evidence.get("resolved_answer")
        col = r['column_name']
        
        if not action or col == "Entire Dataset":
            if action == 'drop_duplicates':
                df = df.drop_duplicates()
            continue
            
        if col not in df.columns:
            continue
            
        if action in ['impute_median', 'impute_mean']:
            missing_count = df[col].isnull().sum()
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median() if action == 'impute_median' else df[col].mean()
            else:
                fill_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(fill_val)
            values_imputed += int(missing_count)
        elif action == 'drop_rows':
            df = df.dropna(subset=[col])
        elif action == 'cap_iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower, upper=upper)
            values_capped += int(((df[col] < lower) | (df[col] > upper)).sum())
        elif action == 'drop':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        elif action == 'lowercase_all':
            df[col] = df[col].astype(str).str.lower()

    # 3. Save snapshot
    snapshot_filename = f"dataset_{run_id}_cleaned.csv"
    snapshot_path = await save_snapshot(db, df, run_id, snapshot_filename)
    
    new_rows = len(df)
    rows_dropped = original_rows - new_rows
    
    # 4. Update pipeline status
    await db.execute("UPDATE pipeline_runs SET status = 'cleaned' WHERE id = $1", run_id)
    await db.execute("INSERT INTO stage_outputs (run_id, stage, r2_snapshot_path) VALUES ($1, 'cleaning', $2)", run_id, snapshot_path)
    
    return {
        "status": "success",
        "original_rows": original_rows,
        "new_rows": new_rows,
        "rows_dropped": rows_dropped,
        "values_imputed": values_imputed,
        "values_capped": values_capped
    }
