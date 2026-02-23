import json
from typing import List, Tuple
from src.services.llm_client import LLMClient
from src.models.schema import AlphaSignal, FilingChunk
from src.core.logger import get_logger

log = get_logger(__name__)

class CriticAgent:
    """
    Reviews the Analyst's output for hallucinations or weak evidence.
    """
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def critique(self, signal: AlphaSignal, chunks: List[FilingChunk]) -> Tuple[bool, str]:
        """
        Returns (is_approved, critique_notes).
        """
        context = "\n\n".join([f"--- SECTION: {c.section_name} ---\n{c.content}" for c in chunks])
        
        system_prompt = """
        You are a Compliance Officer and Risk Manager.
        Your job is to verify the Analyst's report against the source text.
        
        Rules:
        1. If the Analyst cites a quote that DOES NOT exist in the text, REJECT it.
        2. If the signal score is high (>0.7) but the evidence is weak, REJECT it.
        3. If the analysis is sound, APPROVE it.
        
        Output JSON:
        {
            "approved": boolean,
            "critique": "string explanation"
        }
        """
        
        user_prompt = f"""
        Source Text:
        {context}
        
        Analyst Report:
        Score: {signal.signal_score}
        Summary: {signal.summary}
        Quotes: {signal.key_quotes}
        """
        
        response_text = await self.llm.generate(system_prompt, user_prompt, temperature=0.0)
        
        try:
            cleaned = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            return data.get("approved", False), data.get("critique", "No critique provided.")
        except Exception as e:
            log.error("Failed to parse Critic output", error=str(e))
            return False, "Critic parsing failed."
