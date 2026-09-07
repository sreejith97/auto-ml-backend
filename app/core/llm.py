import os
import logging
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider 1 (primary): Groq via OpenAI-compatible client
# ---------------------------------------------------------------------------
_groq_key = os.getenv("GROQ_API_KEY")
_openai_key = os.getenv("OPENAI_API_KEY")

_primary_client = None
_primary_model = "gpt-4o-mini"

if _groq_key:
    _primary_client = AsyncOpenAI(
        api_key=_groq_key,
        base_url="https://api.groq.com/openai/v1"
    )
    _primary_model = "openai/gpt-oss-120b"
elif _openai_key:
    _primary_client = AsyncOpenAI(api_key=_openai_key)
    _primary_model = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Provider 2 (fallback): Google Gemini Flash (free tier)
# Uses the official google-genai SDK with async support.
# ---------------------------------------------------------------------------
_gemini_key = os.getenv("GEMINI_API_KEY")
_gemini_client = None

if _gemini_key:
    try:
        from google import genai
        _gemini_client = genai.Client(api_key=_gemini_key)
        logger.info("Gemini fallback LLM initialized (model: gemini-2.0-flash)")
    except ImportError:
        logger.warning(
            "GEMINI_API_KEY is set but 'google-genai' package is not installed. "
            "Run: pip install google-genai"
        )

_GEMINI_MODEL = "gemini-3.7-flash"

# ---------------------------------------------------------------------------
# Rate-limit detection helpers
# ---------------------------------------------------------------------------
def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception signals a 429 / rate-limit from Groq."""
    exc_str = str(exc).lower()
    if "429" in exc_str or "rate" in exc_str:
        return True
    # openai SDK raises specific status-code errors
    if hasattr(exc, "status_code") and exc.status_code == 429:
        return True
    return False


# ---------------------------------------------------------------------------
# Fallback call through Gemini
# ---------------------------------------------------------------------------
async def _gemini_complete(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    """Call Gemini Flash as a fallback provider."""
    if not _gemini_client:
        raise ValueError(
            "Gemini fallback is not available — "
            "set GEMINI_API_KEY and pip install google-genai"
        )

    combined_prompt = f"{system_prompt}\n\n{user_prompt}"

    response = await _gemini_client.aio.models.generate_content(
        model=_GEMINI_MODEL,
        contents=combined_prompt,
        config={
            "max_output_tokens": max_tokens,
            "temperature": 0.3,
        },
    )
    return response.text.strip()


# ---------------------------------------------------------------------------
# Public API — unchanged signature, automatic fallback
# ---------------------------------------------------------------------------
async def complete(system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
    """
    Thin wrapper around the LLM API.

    Strategy:
      1. Try the primary provider (Groq / OpenAI).
      2. If it returns a 429 rate-limit error **and** a Gemini key is
         configured, transparently retry through Gemini Flash.
      3. If no fallback is available, re-raise the original error.
    """
    if not _primary_client and not _gemini_client:
        raise ValueError(
            "No LLM provider configured. "
            "Set at least one of GROQ_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY."
        )

    # --- Primary attempt ---
    if _primary_client:
        try:
            response = await _primary_client.chat.completions.create(
                model=_primary_model,
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

        except Exception as primary_exc:
            if _is_rate_limit_error(primary_exc) and _gemini_client:
                logger.warning(
                    "Primary LLM hit rate limit (429). Falling back to Gemini Flash."
                )
            else:
                # Not a rate-limit error, or no fallback configured → propagate
                if not _gemini_client:
                    logger.error(f"Primary LLM failed and no fallback configured: {primary_exc}")
                    raise
                # For non-rate-limit errors, still try Gemini as a safety net
                logger.warning(
                    f"Primary LLM failed ({primary_exc}). Trying Gemini fallback."
                )

    # --- Fallback attempt ---
    try:
        result = await _gemini_complete(system_prompt, user_prompt, max_tokens)
        logger.info("Gemini fallback succeeded.")
        return result
    except Exception as fallback_exc:
        logger.error(f"Gemini fallback also failed: {fallback_exc}")
        raise

