import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, Mock
from src.main import app
from src.models.schema import AlphaSignal, RetrievalResult, FilingChunk
from datetime import date

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("src.api.routes.Retriever")
@patch("src.api.routes.AgentWorkflow")
def test_analyze_ticker_success(MockWorkflow, MockRetriever):
    # Mock retrieval
    mock_retriever_instance = MockRetriever.return_value
    mock_chunk = FilingChunk(
        ticker="AAPL", cik="1", form_type="10-K", 
        period_end_date=date.today(), filing_date=date.today(), content="c"
    )
    mock_retriever_instance.search.return_value = [
        RetrievalResult(chunk=mock_chunk, score=0.9, source="dense", rank=1)
    ]
    
    # Mock workflow
    mock_workflow_instance = MockWorkflow.return_value
    mock_signal = AlphaSignal(
        ticker="AAPL", signal_score=0.8, confidence=0.9, summary="Summary", 
        risk_factors=[], key_quotes=[]
    )
    mock_workflow_instance.run_analysis = AsyncMock(return_value=mock_signal)
    
    response = client.post("/api/v1/analyze/AAPL")
    
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["signal_score"] == 0.8

@patch("src.api.routes.Retriever")
def test_analyze_ticker_not_found(MockRetriever):
    mock_retriever_instance = MockRetriever.return_value
    mock_retriever_instance.search.return_value = [] # No results
    
    response = client.post("/api/v1/analyze/AAPL")
    
    assert response.status_code == 404
