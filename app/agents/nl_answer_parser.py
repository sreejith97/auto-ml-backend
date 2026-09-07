from typing import List, Optional, Tuple
import json
import logging
from app.core import llm

logger = logging.getLogger(__name__)

async def parse_answer(question_text: str, user_text: str, allowed_options: List[str]) -> Tuple[Optional[str], float, Optional[str]]:
    """
    Uses the LLM to map a free-text user response to one of the structured `allowed_options`.
    Returns (matched_option, confidence, clarification_needed).
    """
    system_prompt = f"""
    You are an AI data engineering assistant. A user was asked a question and gave a free-text answer.
    Your job is to classify their answer into one of the allowed options.
    
    Question: "{question_text}"
    Allowed Options: {allowed_options}
    
    Return ONLY a strict JSON object with this exact shape:
    {{
      "matched_option": "exact_string_from_allowed_options_or_null",
      "confidence": float_between_0_and_1,
      "clarification_needed": "a polite string asking for clarification if confidence < 0.7 or null"
    }}
    Do not return any markdown wrappers, just the raw JSON string.
    """
    
    try:
        response_text = await llm.complete(system_prompt=system_prompt, user_prompt=user_text)
        
        # Strip potential markdown blocks if the LLM adds them despite instructions
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
        data = json.loads(response_text)
        
        matched_option = data.get("matched_option")
        confidence = float(data.get("confidence", 0.0))
        clarification_needed = data.get("clarification_needed")
        
        if confidence < 0.7 or matched_option not in allowed_options:
            fallback = clarification_needed or f"I couldn't confidently map your answer to {allowed_options}. Could you clarify?"
            return None, confidence, fallback
            
        return matched_option, confidence, None
        
    except Exception as e:
        logger.error(f"Failed to parse LLM answer: {e}")
        return None, 0.0, "I'm having trouble processing that right now. Could you please use the buttons, or try again later?"

async def parse_chat_intent(user_text: str, current_stage: str) -> dict:
    """
    Parses a free-text chat message to determine the user's intent.
    Used for generic chat routing, especially in the 'columns' stage to update schema.
    Returns a dict with 'intent' and associated metadata.
    """
    system_prompt = f"""
    You are an AI data engineering assistant.
    Analyze their message and determine their intent.
    
    If they are asking to change the role of a column (e.g., "Make price the target", "Ignore the id column", "has_pool is the target column"), return an intent of "update_column_role".
    
    CRITICAL: Keep your 'response' extremely concise (under 15 words) to save context window space.
    
    Return ONLY a strict JSON object with this exact shape:
    {{
      "intent": "update_column_role" | "general_chat",
      "column_name": "the exact name of the column (if intent is update_column_role), else null",
      "new_role": "target" | "feature" | "ignore" | "id" (if intent is update_column_role), else null,
      "response": "A highly concise response under 15 words (only if intent is general_chat), else null"
    }}
    Do not return any markdown wrappers, just the raw JSON string.
    """
    
    response_text = "<not generated>"
    try:
        response_text = await llm.complete(system_prompt=system_prompt, user_prompt=user_text)
        
        import re
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL)
        if match:
            response_text = match.group(1)
            
        data = json.loads(response_text.strip())
        return data
        
    except Exception as e:
        logger.error(f"Failed to parse chat intent: {e}. Raw response: {response_text}")
        return {"intent": "error", "response": "Sorry, I had trouble understanding that."}
