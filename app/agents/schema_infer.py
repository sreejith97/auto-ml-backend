import json
from app.core.llm import complete
import pandas as pd
import logging

logger = logging.getLogger(__name__)

async def infer_schema(df: pd.DataFrame) -> list:
    """
    Passes a sample of the dataframe to the LLM to infer column schema.
    Returns a list of dicts:
    [
        {"column_name": "...", "role": "...", "semantic_type": "...", "notes": "..."}
    ]
    """
    # Create a string representation of the first 5 rows and column dtypes
    sample_data = df.head(5).to_dict(orient='list')
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    prompt = f"""
You are an expert Data Scientist. I have loaded a dataset. I need you to infer the schema for each column.
Here are the column names and their pandas dtypes:
{json.dumps(dtypes, indent=2)}

Here is a sample of the first 5 rows:
{json.dumps(sample_data, indent=2)}

For each column, determine:
1. role: Must be exactly one of: ['feature', 'target', 'id', 'ignore'].
   - Only choose 'target' if it is OBVIOUSLY the target variable (like 'price', 'is_churned', 'sales'). If none are obvious, set all non-ID columns to 'feature'.
   - Choose 'id' for primary keys.
   - Choose 'ignore' for columns that are entirely null or obviously useless.
2. semantic_type: Must be exactly one of: ['numeric', 'categorical', 'datetime', 'text', 'boolean']
3. notes: A very brief 1-sentence description of what this column represents, meant for a non-technical user.

Respond ONLY with a valid JSON array of objects. No markdown formatting, no code blocks, just raw JSON.
Example format:
[
  {{"column_name": "id", "role": "id", "semantic_type": "numeric", "notes": "A unique identifier for each row."}},
  {{"column_name": "price", "role": "target", "semantic_type": "numeric", "notes": "The sale price of the property."}}
]
    """
    
    try:
        response_text = await complete(
            system_prompt="You are a JSON-only schema inference bot. Output raw JSON only.",
            user_prompt=prompt,
            max_tokens=2000
        )
        
        # Strip potential markdown formatting if the LLM misbehaves
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
            
        inferred = json.loads(cleaned.strip())
        
        # Basic validation
        valid_roles = {'feature', 'target', 'id', 'ignore'}
        valid_semantics = {'numeric', 'categorical', 'datetime', 'text', 'boolean'}
        
        for col in inferred:
            if col.get('role') not in valid_roles:
                col['role'] = 'feature'
            if col.get('semantic_type') not in valid_semantics:
                col['semantic_type'] = 'text'
                
        return inferred
        
    except Exception as e:
        logger.error(f"Failed to infer schema: {e}")
        # Fallback to dumb defaults if LLM fails
        fallback = []
        for col in df.columns:
            fallback.append({
                "column_name": col,
                "role": "feature",
                "semantic_type": "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "text",
                "notes": f"The {col} column."
            })
        return fallback
