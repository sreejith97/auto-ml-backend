import asyncpg
import logging
from app.core.queries import conversation
from app.core import llm

logger = logging.getLogger(__name__)

async def generate_narration(prompt: str, fallback_msg: str) -> str:
    system_prompt = "You are a helpful AI assistant narrating the progress of a machine learning pipeline to a non-technical user. Keep your responses short, encouraging, and in plain English (1-3 sentences max)."
    try:
        return await llm.complete(system_prompt, prompt)
    except Exception as e:
        logger.error(f"Narration LLM failed: {e}")
        return fallback_msg

async def narrate_ingestion(conn: asyncpg.Connection, run_id: int, dataset_name: str, row_count: int, schema_list: list = None):
    fallback = f"I have successfully ingested the dataset '{dataset_name}' containing {row_count} rows. I've automatically analyzed your columns. Please review my findings."
    
    if schema_list:
        # Create a brief summary of what we found
        target_cols = [c['column_name'] for c in schema_list if c['role'] == 'target']
        target_str = f"I think the target variable to predict is '{target_cols[0]}'." if target_cols else "I couldn't find an obvious target variable to predict."
        prompt = f"The dataset '{dataset_name}' with {row_count} rows was just successfully ingested. I have also inferred the schema for its {len(schema_list)} columns. {target_str} Narrate this success, mention the target guess, and invite the user to review the columns or correct me in the chat."
    else:
        prompt = f"The dataset '{dataset_name}' with {row_count} rows was just successfully ingested into the pipeline. Narrate this success and invite the user to review column semantics."
        
    msg = await generate_narration(prompt, fallback)
    await conversation.insert_message(conn, run_id, 'assistant', msg, 'ingestion')

async def narrate_columns(conn: asyncpg.Connection, run_id: int, target_col: str):
    fallback = f"I've recorded your column specifications. We will be trying to predict '{target_col}'. I am now ready to move on to define the task."
    prompt = f"The user just finished specifying column metadata. The target variable to predict is '{target_col}'. Narrate this and invite them to define the ML task next."
    msg = await generate_narration(prompt, fallback)
    await conversation.insert_message(conn, run_id, 'assistant', msg, 'columns')

async def narrate_task(conn: asyncpg.Connection, run_id: int, task_type: str, target_col: str):
    fallback = f"I have locked in the task as {task_type} predicting '{target_col}'. Let's move on to the cleaning stage!"
    prompt = f"The user just chose the ML task type as '{task_type}' for predicting '{target_col}'. Narrate this and enthusiastically invite them to proceed to the Data Cleaning stage."
    msg = await generate_narration(prompt, fallback)
    await conversation.insert_message(conn, run_id, 'assistant', msg, 'task')

async def narrate_cleaning(conn: asyncpg.Connection, run_id: int, auto_applied: int, pending: int):
    fallback = f"Data cleaning complete. I auto-applied {auto_applied} obvious cleaning rules."
    if pending > 0:
        fallback += f" However, I found {pending} ambiguous issues that I need your help with. Please let me know how you want to handle them."
    
    prompt = f"Data cleaning is paused. I auto-applied {auto_applied} rules, but found {pending} ambiguous issues (missing values, outliers) that need the user's manual judgment. Ask them to look at the cards on the left to resolve them."
    msg = await generate_narration(prompt, fallback)
    await conversation.insert_message(conn, run_id, 'assistant', msg, 'cleaning')

async def narrate_eda(conn: asyncpg.Connection, run_id: int, findings: list) -> str:
    if not findings:
        fallback = "I've analyzed the data distributions and correlations. Everything looks standard."
        prompt = "Tell the user you finished EDA but found nothing highly notable to report."
    else:
        # Convert structured findings to a string summary
        findings_text = "\n".join([f"- {f['type']}: {f['detail']}" for f in findings])
        fallback = f"I've completed the exploratory data analysis. Here is what I found:\n{findings_text}"
        prompt = f"I just finished EDA and found the following notable insights:\n{findings_text}\nSummarize these insights in a very brief, friendly, non-technical paragraph."
        
    msg = await generate_narration(prompt, fallback)
    await conversation.insert_message(conn, run_id, 'assistant', msg, 'eda')
    return msg

async def narrate_features(conn: asyncpg.Connection, run_id: int, proposals_list: list):
    proposals_count = len(proposals_list)
    fallback = f"I've finished analyzing your features. I generated {proposals_count} new feature proposals based on your data semantics."
    prompt = f"I just engineered {proposals_count} new feature proposals for the dataset. Tell the user this and invite them to review them."
    msg = await generate_narration(prompt, fallback)
    await conversation.insert_message(conn, run_id, 'assistant', msg, 'features')
async def narrate_results(conn: asyncpg.Connection, run_id: int, top_model: str, accuracy: float):
    fallback = f"Training complete! The best performing model was {top_model} achieving an accuracy/R2 score of {accuracy:.4f}. You can review the feature importance and metrics on the left."
    prompt = f"Model training is done! The best model is {top_model} with a score of {accuracy:.4f}. Congratulate the user and point them to the charts on the left."
    msg = await generate_narration(prompt, fallback)
    await conversation.insert_message(conn, run_id, 'assistant', msg, 'training')
