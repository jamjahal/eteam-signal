import pytest
from datetime import date, datetime
from src.models.schema import FilingChunk, RetrievalResult, AlphaSignal

def test_filing_chunk_creation():
    chunk = FilingChunk(
        ticker="AAPL",
        cik="12345",
        form_type="10-K",
        period_end_date=date(2023, 12, 31),
        filing_date=date(2024, 2, 1),
        section_name="Item 1A",
        content="Risk factors content...",
        tokens=100
    )
    assert chunk.ticker == "AAPL"
    assert chunk.id is not None
    assert isinstance(chunk.id, str)

def test_retrieval_result_creation():
    chunk = FilingChunk(
        ticker="AAPL",
        cik="12345",
        form_type="10-K",
        period_end_date=date(2023, 12, 31),
        filing_date=date(2024, 2, 1),
        content="Content",
        tokens=10
    )
    result = RetrievalResult(
        chunk=chunk,
        score=0.95,
        source="dense",
        rank=1
    )
    assert result.score == 0.95
    assert result.chunk.ticker == "AAPL"

def test_alpha_signal_creation():
    signal = AlphaSignal(
        ticker="AAPL",
        signal_score=0.8,
        confidence=0.9,
        summary="High risk",
        risk_factors=["Factor 1", "Factor 2"],
        key_quotes=["Quote 1"]
    )
    assert signal.signal_score == 0.8
    assert len(signal.risk_factors) == 2
    assert isinstance(signal.analysis_date, datetime)
