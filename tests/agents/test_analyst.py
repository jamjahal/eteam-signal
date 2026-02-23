import pytest
from unittest.mock import Mock, AsyncMock
from src.agents.analyst import AnalystAgent
from src.models.schema import FilingChunk
from datetime import date

@pytest.mark.asyncio
async def test_analyze_success():
    mock_llm = Mock()
    # Mock successful JSON response
    mock_llm.generate = AsyncMock(return_value="""
    {
        "signal_score": 0.8,
        "confidence": 0.9,
        "summary": "Bad news",
        "risk_factors": ["Risk A"],
        "key_quotes": ["Quote B"]
    }
    """)
    
    agent = AnalystAgent(mock_llm)
    chunk = FilingChunk(
        ticker="A", cik="1", form_type="10-K", 
        period_end_date=date.today(), filing_date=date.today(), 
        content="abc"
    )
    
    signal = await agent.analyze("A", [chunk])
    
    assert signal.signal_score == 0.8
    assert signal.summary == "Bad news"
    assert "Risk A" in signal.risk_factors

@pytest.mark.asyncio
async def test_analyze_parse_failure():
    mock_llm = Mock()
    mock_llm.generate = AsyncMock(return_value="Not JSON")
    
    agent = AnalystAgent(mock_llm)
    chunk = FilingChunk(
        ticker="A", cik="1", form_type="10-K", 
        period_end_date=date.today(), filing_date=date.today(), 
        content="abc"
    )
    
    signal = await agent.analyze("A", [chunk])
    
    # specific fallback behavior
    assert signal.signal_score == 0.0
    assert "Parsing error" in signal.summary
