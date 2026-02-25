import pytest
from datetime import date, datetime

from src.models.insider_schema import (
    AnomalyType,
    InsiderAnomaly,
    InsiderProfile,
    InsiderSentiment,
    InsiderSignal,
    InsiderTransaction,
    TransactionCode,
)


def _make_tx(**overrides) -> InsiderTransaction:
    defaults = dict(
        ticker="AAPL",
        insider_name="Tim Cook",
        insider_title="CEO",
        is_officer=True,
        is_director=False,
        transaction_date=date(2025, 6, 1),
        transaction_code=TransactionCode.SALE,
        shares=10000.0,
        price_per_share=190.0,
        total_value=1_900_000.0,
        shares_owned_after=50000.0,
        is_10b5_1=False,
        filing_date=date(2025, 6, 3),
    )
    defaults.update(overrides)
    return InsiderTransaction(**defaults)


class TestInsiderTransaction:
    def test_basic_creation(self):
        tx = _make_tx()
        assert tx.ticker == "AAPL"
        assert tx.transaction_code == TransactionCode.SALE
        assert tx.shares == 10000.0

    def test_frozen_model(self):
        tx = _make_tx()
        with pytest.raises(Exception):
            tx.ticker = "GOOG"

    def test_optional_fields(self):
        tx = _make_tx(price_per_share=None, total_value=None, shares_owned_after=None)
        assert tx.price_per_share is None

    def test_transaction_code_enum(self):
        assert TransactionCode.PURCHASE.value == "P"
        assert TransactionCode.SALE.value == "S"
        tx = _make_tx(transaction_code=TransactionCode.PURCHASE)
        assert tx.transaction_code == TransactionCode.PURCHASE


class TestInsiderProfile:
    def test_creation(self):
        profile = InsiderProfile(
            insider_name="Tim Cook",
            ticker="AAPL",
            avg_transaction_size=1_500_000.0,
            avg_frequency_days=45.0,
            total_transactions=20,
            typical_sell_percentage=0.05,
            last_transaction_date=date(2025, 6, 1),
        )
        assert profile.total_transactions == 20
        assert profile.avg_frequency_days == 45.0

    def test_defaults(self):
        profile = InsiderProfile(insider_name="Nobody", ticker="XYZ")
        assert profile.avg_transaction_size == 0.0
        assert profile.last_transaction_date is None


class TestInsiderAnomaly:
    def test_score_bounds(self):
        a = InsiderAnomaly(
            ticker="AAPL",
            insider_name="Tim Cook",
            anomaly_type=AnomalyType.VOLUME,
            severity_score=0.85,
        )
        assert 0.0 <= a.severity_score <= 1.0

    def test_invalid_score(self):
        with pytest.raises(Exception):
            InsiderAnomaly(
                ticker="AAPL",
                insider_name="X",
                anomaly_type=AnomalyType.VOLUME,
                severity_score=1.5,
            )

    def test_anomaly_types(self):
        for at in AnomalyType:
            a = InsiderAnomaly(
                ticker="T", insider_name="X", anomaly_type=at, severity_score=0.5
            )
            assert a.anomaly_type == at


class TestInsiderSignal:
    def test_defaults(self):
        sig = InsiderSignal(ticker="AAPL")
        assert sig.anomaly_score == 0.0
        assert sig.insider_sentiment == InsiderSentiment.NEUTRAL
        assert sig.anomalies == []
        assert sig.composite_alpha_score is None

    def test_with_anomalies(self):
        anomaly = InsiderAnomaly(
            ticker="AAPL",
            insider_name="X",
            anomaly_type=AnomalyType.CLUSTER,
            severity_score=0.9,
        )
        sig = InsiderSignal(
            ticker="AAPL",
            anomaly_score=0.8,
            anomalies=[anomaly],
            insider_sentiment=InsiderSentiment.BEARISH,
        )
        assert len(sig.anomalies) == 1
        assert sig.insider_sentiment == InsiderSentiment.BEARISH
