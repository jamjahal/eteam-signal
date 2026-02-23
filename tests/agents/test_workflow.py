import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.agents.workflow import AgentWorkflow
from src.models.schema import AlphaSignal, FilingChunk
from datetime import date

@pytest.mark.asyncio
async def test_run_analysis_flow():
    with patch("src.agents.workflow.LLMClient"), \
         patch("src.agents.workflow.AnalystAgent") as MockAnalyst, \
         patch("src.agents.workflow.CriticAgent") as MockCritic:
        
        # Setup mocks
        mock_analyst = MockAnalyst.return_value
        mock_critic = MockCritic.return_value
        
        signal_out = AlphaSignal(ticker="A", signal_score=0.8, confidence=1.0, summary="High")
        mock_analyst.analyze = AsyncMock(return_value=signal_out)
        
        # Critic rejects it
        mock_critic.critique = AsyncMock(return_value=(False, "Too aggressive"))
        
        workflow = AgentWorkflow()
        chunk = FilingChunk(ticker="A", cik="1", form_type="10-K", 
                            period_end_date=date.today(), filing_date=date.today(), content="c")
        
        final_signal = await workflow.run_analysis("A", [chunk])
        
        # Check that logic reduced confidence
        assert final_signal.confidence == 0.5 
        assert final_signal.critic_notes == "Too aggressive"
