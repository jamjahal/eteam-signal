import pytest
from unittest.mock import Mock, AsyncMock
from src.agents.critic import CriticAgent
from src.models.schema import AlphaSignal, FilingChunk
from datetime import date

@pytest.mark.asyncio
async def test_critique_approved():
    mock_llm = Mock()
    mock_llm.generate = AsyncMock(return_value="""
    {
        "approved": true,
        "critique": "Looks good"
    }
    """)
    
    agent = CriticAgent(mock_llm)
    signal = AlphaSignal(ticker="A", signal_score=0.5, confidence=0.5, summary="s")
    chunk = FilingChunk(ticker="A", cik="1", form_type="10-K", 
                        period_end_date=date.today(), filing_date=date.today(), content="c")
    
    approved, notes = await agent.critique(signal, [chunk])
    
    assert approved is True
    assert notes == "Looks good"

@pytest.mark.asyncio
async def test_critique_rejected():
    mock_llm = Mock()
    mock_llm.generate = AsyncMock(return_value="""
    {
        "approved": false,
        "critique": "Hallucination"
    }
    """)
    
    agent = CriticAgent(mock_llm)
    signal = AlphaSignal(ticker="A", signal_score=0.5, confidence=0.5, summary="s")
    chunk = FilingChunk(ticker="A", cik="1", form_type="10-K", 
                        period_end_date=date.today(), filing_date=date.today(), content="c")
    
    approved, notes = await agent.critique(signal, [chunk])
    
    assert approved is False
    assert notes == "Hallucination"
