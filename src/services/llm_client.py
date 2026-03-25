import asyncio
import anthropic
from typing import Optional, List, Dict, Any
from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)

# Default retry settings for rate-limit (429) errors
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_INITIAL_BACKOFF = 30.0  # seconds — matches a 1-minute token bucket reset
_DEFAULT_BACKOFF_MULTIPLIER = 2.0
_DEFAULT_MAX_BACKOFF = 120.0


class LLMClient:
    """
    Unified interface for LLM interactions, currently backing to Anthropic.

    Automatically retries on 429 (rate-limit) responses using exponential
    backoff so that callers don't have to handle throttling themselves.
    """
    def __init__(
        self,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        initial_backoff: float = _DEFAULT_INITIAL_BACKOFF,
        backoff_multiplier: float = _DEFAULT_BACKOFF_MULTIPLIER,
        max_backoff: float = _DEFAULT_MAX_BACKOFF,
    ):
        if not settings.ANTHROPIC_API_KEY:
            log.warning("ANTHROPIC_API_KEY is not set. LLM calls will fail.")

        self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.LLM_MODEL

        # Retry knobs
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff = max_backoff

    async def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """
        Simple generation call with automatic retry on 429 rate-limit errors.

        The backoff starts at *initial_backoff* seconds (default 30 s) and
        doubles on each consecutive 429 up to *max_backoff* (default 120 s).
        If the API returns a ``retry-after`` header the client honours that
        value instead.
        """
        backoff = self.initial_backoff

        for attempt in range(1, self.max_retries + 1):
            try:
                message = await self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return message.content[0].text

            except anthropic.RateLimitError as exc:
                # Honour retry-after header when present
                retry_after = getattr(exc, "response", None)
                if retry_after is not None:
                    retry_after = exc.response.headers.get("retry-after")
                if retry_after is not None:
                    wait = float(retry_after)
                else:
                    wait = backoff

                if attempt == self.max_retries:
                    log.error(
                        "Rate-limit retries exhausted",
                        attempts=attempt,
                        error=str(exc),
                    )
                    raise

                log.warning(
                    "Rate-limited by API, backing off",
                    attempt=attempt,
                    wait_seconds=wait,
                )
                await asyncio.sleep(wait)
                backoff = min(backoff * self.backoff_multiplier, self.max_backoff)

            except Exception as e:
                log.error("LLM Generation failed", error=str(e))
                raise

        # Should not be reached, but just in case:
        raise RuntimeError("LLM generate: retry loop exited unexpectedly")

    # Future: support tool use / structured output here
