from typing import List, Optional, Dict, Literal
from datetime import date, datetime
from pydantic import BaseModel, Field
import uuid

class FilingChunk(BaseModel):
    """
    Represents a semantically meaningful chunk of an SEC filing.
    Ideally corresponds to a section or subsection (e.g., Item 1A).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ticker: str
    cik: str
    form_type: Literal["10-K", "10-Q"]
    period_end_date: date
    filing_date: date
    section_name: Optional[str] = None
    content: str
    tokens: int = 0
    metadata: Dict[str, str] = Field(default_factory=dict)

    class Config:
        frozen = True

class RetrievalResult(BaseModel):
    """
    A single result from the hybrid retrieval engine.
    """
    chunk: FilingChunk
    score: float
    source: Literal["dense", "sparse", "hybrid"]
    rank: int

class AlphaSignal(BaseModel):
    """
    The output of the Analyst/Critic loop.
    """
    ticker: str
    analysis_date: datetime = Field(default_factory=datetime.now)
    signal_score: float = Field(..., description="0.0 to 1.0 score indicating alpha potential (risk/sentiment drift).")
    confidence: float = Field(..., description="0.0 to 1.0 confidence in the assessment.")
    summary: str = Field(..., description="Executive summary of the findings.")
    risk_factors: List[str] = Field(default_factory=list, description="Key risk factors identified.")
    key_quotes: List[str] = Field(default_factory=list, description="Verbatim quotes supporting the signal.")
    critic_notes: Optional[str] = Field(None, description="Notes from the critic agent regarding potential hallucinations or gaps.")
