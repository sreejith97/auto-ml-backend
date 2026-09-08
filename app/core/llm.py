"""
app/core/llm.py — Agno model + storage factory + legacy shim
=============================================================
Phase 1 of the Agno workflow migration.

Public surface
--------------
get_model()            → Agno model object (Groq | OpenAI | Gemini)
get_workflow_db()      → Agno PostgresDb for workflow HITL state
get_agent_storage(tbl) → Agno PostgresAgentStorage for per-agent memory
complete(...)          → LEGACY: raw string completion (removed in Phase 10)
"""

import os
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------
_groq_key   = os.getenv("GROQ_API_KEY")
_openai_key = os.getenv("OPENAI_API_KEY")
_gemini_key = os.getenv("GEMINI_API_KEY")

# Raw DATABASE_URL from env (asyncpg-style: postgresql:// or postgresql+asyncpg://)
_raw_db_url = os.getenv("DATABASE_URL", "")

# Agno's PostgresDb / PostgresAgentStorage expect the psycopg driver URL:
#   postgresql+psycopg://user:pass@host/db
def _agno_db_url() -> str:
    url = _raw_db_url
    # Convert asyncpg-style URL to psycopg-style for Agno
    url = url.replace("postgresql+asyncpg://", "postgresql+psycopg://")
    url = url.replace("postgresql://", "postgresql+psycopg://")
    url = url.replace("postgres://", "postgresql+psycopg://")
    return url


# ---------------------------------------------------------------------------
# Agno Model Factory
# ---------------------------------------------------------------------------
def get_model():
    """
    Returns the primary Agno model based on configured API keys.
    Priority: Groq → OpenAI → Gemini
    """
    if _groq_key:
        from agno.models.groq import Groq
        return Groq(id="openai/gpt-oss-120b", api_key=_groq_key)
    if _openai_key:
        from agno.models.openai import OpenAIChat
        return OpenAIChat(id="gpt-4o-mini", api_key=_openai_key)
    if _gemini_key:
        from agno.models.google import Gemini
        return Gemini(id="gemini-2.0-flash", api_key=_gemini_key)
    raise ValueError(
        "No LLM provider configured. "
        "Set at least one of GROQ_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY."
    )


# ---------------------------------------------------------------------------
# Agno PostgresDb Factory (for Workflow HITL state persistence)
# ---------------------------------------------------------------------------
def get_workflow_db():
    """
    Returns an Agno PostgresDb instance connected to the application's
    Postgres database. Used for Workflow HITL pause/resume state storage.
    """
    from agno.db.postgres import PostgresDb
    db_url = _agno_db_url()
    if not db_url:
        raise ValueError("DATABASE_URL must be set for Agno workflow persistence.")
    return PostgresDb(db_url=db_url)


# ---------------------------------------------------------------------------
# Agno Agent Storage Factory (for per-agent conversation memory)
# ---------------------------------------------------------------------------
def get_agent_storage(table_name: str):
    """
    Returns an Agno PostgresDb for storing per-agent message history.
    Each agent type uses its own table (e.g. 'narration_sessions', 'chat_sessions').
    """
    from agno.db.postgres import PostgresDb
    db_url = _agno_db_url()
    if not db_url:
        raise ValueError("DATABASE_URL must be set for Agno agent storage.")
    return PostgresDb(db_url=db_url, session_table=table_name)


# ---------------------------------------------------------------------------
# LEGACY shim — to be removed in Phase 10
# ---------------------------------------------------------------------------
# Existing agents/routes still call `from app.core.llm import complete`.
# This shim delegates through an ephemeral Agno Agent so the migration
# can proceed incrementally without breaking existing callers.
# ---------------------------------------------------------------------------

# Keep old raw clients alive for the shim so we don't cold-start Agno on
# every legacy call (it would be slow).
from openai import AsyncOpenAI as _AsyncOpenAI

_primary_client: "_AsyncOpenAI | None" = None
_primary_model_id = "gpt-4o-mini"

if _groq_key:
    _primary_client = _AsyncOpenAI(
        api_key=_groq_key,
        base_url="https://api.groq.com/openai/v1",
    )
    _primary_model_id = "openai/gpt-oss-120b"
elif _openai_key:
    _primary_client = _AsyncOpenAI(api_key=_openai_key)
    _primary_model_id = "gpt-4o-mini"

_gemini_client = None
if _gemini_key:
    try:
        from google import genai as _genai
        _gemini_client = _genai.Client(api_key=_gemini_key)
    except ImportError:
        logger.warning("GEMINI_API_KEY set but google-genai not installed.")

_GEMINI_MODEL = "gemini-3.7-flash"


def _is_rate_limit_error(exc: Exception) -> bool:
    exc_str = str(exc).lower()
    if "429" in exc_str or "rate" in exc_str:
        return True
    if hasattr(exc, "status_code") and exc.status_code == 429:
        return True
    return False


async def _gemini_complete(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    if not _gemini_client:
        raise ValueError("Gemini fallback not available — set GEMINI_API_KEY.")
    combined = f"{system_prompt}\n\n{user_prompt}"
    response = await _gemini_client.aio.models.generate_content(
        model=_GEMINI_MODEL,
        contents=combined,
        config={"max_output_tokens": max_tokens, "temperature": 0.3},
    )
    return response.text.strip()


async def complete(system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
    """
    LEGACY: Raw string completion. Kept for backward-compat during migration.
    Will be removed in Phase 10 once all callers are migrated to Agno Agents.
    """
    if not _primary_client and not _gemini_client:
        raise ValueError("No LLM provider configured.")

    if _primary_client:
        try:
            response = await _primary_client.chat.completions.create(
                model=_primary_model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
                extra_body={
                    "reasoning_format": "hidden",
                    "reasoning_effort": "medium",
                },
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            if _gemini_client:
                logger.warning(f"Primary LLM failed ({exc}). Falling back to Gemini.")
            else:
                raise

    try:
        result = await _gemini_complete(system_prompt, user_prompt, max_tokens)
        logger.info("Gemini fallback succeeded.")
        return result
    except Exception as fallback_exc:
        logger.error(f"Gemini fallback also failed: {fallback_exc}")
        raise
