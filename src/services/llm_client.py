import anthropic
from typing import Optional, List, Dict, Any
from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)

class LLMClient:
    """
    Unified interface for LLM interactions, currently backing to Anthropic.
    """
    def __init__(self):
        if not settings.ANTHROPIC_API_KEY:
            log.warning("ANTHROPIC_API_KEY is not set. LLM calls will fail.")
        
        self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.LLM_MODEL

    async def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """
        Simple generation call.
        """
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
        except Exception as e:
            log.error("LLM Generation failed", error=str(e))
            raise

    # Future: support tool use / structured output here
