import json
from typing import List
from src.services.llm_client import LLMClient
from src.models.schema import AlphaSignal, RetrievalResult, FilingChunk
from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


def _truncate(text: str, max_chars: int) -> str:
    """Truncate *text* to *max_chars*, appending an ellipsis marker if trimmed."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[…truncated]"


class AnalystAgent:
    """
    Analyzes retrieved SEC chunks to produce an alpha signal.
    """
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def analyze(self, ticker: str, chunks: List[FilingChunk]) -> AlphaSignal:
        """
        Synthesize chunks into a signal.
        """
        max_chars = settings.MAX_CHUNK_CHARS
        context = "\n\n".join([
            f"--- SECTION: {c.section_name} ({c.filing_date}) ---\n{_truncate(c.content, max_chars)}"
            for c in chunks
        ])
        
        system_prompt = """
        You are a Senior Quantitative Analyst at a top-tier hedge fund.
        Your goal is to extract an "Alpha Signal" from the provided SEC filing excerpts.
        
        Focus on:
        1. Material changes in Risk Factors (Item 1A).
        2. Subtle shifts in Management sentiment (MD&A).
        3. Specific legal or regulatory threats.
        
        Output valid JSON adhering to this schema:
        {
            "signal_score": float (0.0 to 1.0, where 1.0 is high conviction of drift/risk),
            "confidence": float (0.0 to 1.0),
            "summary": "string",
            "risk_factors": ["string", "string"],
            "key_quotes": ["string", "string"]
        }
        """
        
        user_prompt = f"Analyze the following excerpts for ticker {ticker}:\n\n{context}"
        
        response_text = await self.llm.generate(system_prompt, user_prompt, temperature=0.1)
        
        # Simple parsing for now. In prod, use Pydantic output parsers or tool use.
        try:
            # clean potential markdown fences
            cleaned = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            
            return AlphaSignal(
                ticker=ticker,
                signal_score=data.get("signal_score", 0.5),
                confidence=data.get("confidence", 0.5),
                summary=data.get("summary", "Analysis failed to parse."),
                risk_factors=data.get("risk_factors", []),
                key_quotes=data.get("key_quotes", [])
            )
        except Exception as e:
            log.error("Failed to parse Analyst output", error=str(e), response=response_text)
            # Return a fallback signal
            return AlphaSignal(
                ticker=ticker,
                signal_score=0.0,
                confidence=0.0,
                summary=f"Parsing error: {str(e)}",
            )
