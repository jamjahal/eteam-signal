from typing import List, Optional
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class TransactionCode(str, Enum):
    """SEC Form 4 transaction codes."""
    PURCHASE = "P"
    SALE = "S"
    AWARD = "A"
    DISPOSITION = "D"
    CONVERSION = "C"
    EXERCISE = "M"
    OTHER = "O"


class AnomalyType(str, Enum):
    VOLUME = "VOLUME"
    FREQUENCY = "FREQUENCY"
    CLUSTER = "CLUSTER"
    HOLDINGS_PERCENTAGE = "HOLDINGS_PERCENTAGE"
    TIMING = "TIMING"


class InsiderSentiment(str, Enum):
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"


class InsiderTransaction(BaseModel):
    """Single Form 4 transaction parsed from SEC EDGAR."""

    ticker: str
    insider_name: str
    insider_title: str = ""
    is_officer: bool = False
    is_director: bool = False
    transaction_date: date
    transaction_code: TransactionCode
    shares: float
    price_per_share: Optional[float] = None
    total_value: Optional[float] = None
    shares_owned_after: Optional[float] = None
    is_10b5_1: bool = False
    filing_date: date

    model_config = ConfigDict(frozen=True)


class InsiderProfile(BaseModel):
    """Aggregated executive baseline built from historical Form 4 data."""

    insider_name: str
    ticker: str
    avg_transaction_size: float = 0.0
    avg_frequency_days: float = 0.0
    total_transactions: int = 0
    typical_sell_percentage: float = 0.0
    last_transaction_date: Optional[date] = None


class InsiderAnomaly(BaseModel):
    """A single detected anomaly in insider trading behaviour."""

    ticker: str
    insider_name: str
    anomaly_type: AnomalyType
    severity_score: float = Field(ge=0.0, le=1.0)
    z_score: float = 0.0
    description: str = ""
    transactions: List[InsiderTransaction] = Field(default_factory=list)


class InsiderSignal(BaseModel):
    """Aggregated insider-trading signal for a single ticker."""

    ticker: str
    analysis_date: datetime = Field(default_factory=datetime.now)
    anomaly_score: float = Field(0.0, ge=0.0, le=1.0)
    anomalies: List[InsiderAnomaly] = Field(default_factory=list)
    insider_sentiment: InsiderSentiment = InsiderSentiment.NEUTRAL
    recommendation: str = ""
    composite_alpha_score: Optional[float] = Field(None, ge=0.0, le=1.0)
