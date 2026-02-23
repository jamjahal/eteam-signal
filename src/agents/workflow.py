from typing import List, Optional
from src.services.llm_client import LLMClient
from src.agents.analyst import AnalystAgent
from src.agents.critic import CriticAgent
from src.models.schema import AlphaSignal, FilingChunk
from src.core.logger import get_logger

log = get_logger(__name__)

class AgentWorkflow:
    """
    Orchestrates the Analyst -> Critic loop.
    """
    def __init__(self):
        self.llm = LLMClient()
        self.analyst = AnalystAgent(self.llm)
        self.critic = CriticAgent(self.llm)

    async def run_analysis(self, ticker: str, chunks: List[FilingChunk], max_retries: int = 1) -> AlphaSignal:
        """
        Run the analysis loop. 
        If Critic rejects, we could loop back to Analyst (simplification: just append critique).
        """
        log.info("Starting Agent Workflow", ticker=ticker)
        
        # 1. Analyst Pass
        signal = await self.analyst.analyze(ticker, chunks)
        log.info("Analyst Signal", score=signal.signal_score)
        
        # 2. Critic Pass
        approved, critique = await self.critic.critique(signal, chunks)
        log.info("Critic Result", approved=approved, critique=critique)
        
        # 3. Finalize
        # In a full loop, we would re-prompt the Analyst with the critique.
        # For this version, we attach the critique to the signal.
        signal.critic_notes = critique
        
        # Adjust confidence if rejected?
        if not approved:
            signal.confidence = signal.confidence * 0.5
            
        return signal
