import pytest
from unittest.mock import AsyncMock, patch
from datetime import date, timedelta

from src.agents.insider_analyzer import InsiderAnalyzer, VOLUME_Z_THRESHOLD
from src.models.insider_schema import (
    AnomalyType,
    InsiderProfile,
    InsiderSentiment,
    InsiderTransaction,
    TransactionCode,
)


def _make_tx(days_ago=0, shares=1000, price=150.0, code=TransactionCode.SALE, **kw):
    defaults = dict(
        ticker="AAPL",
        insider_name="Tim Cook",
        insider_title="CEO",
        is_officer=True,
        is_director=False,
        transaction_date=date.today() - timedelta(days=days_ago),
        transaction_code=code,
        shares=shares,
        price_per_share=price,
        total_value=shares * price,
        shares_owned_after=50000.0,
        is_10b5_1=False,
        filing_date=date.today() - timedelta(days=max(0, days_ago - 2)),
    )
    defaults.update(kw)
    return InsiderTransaction(**defaults)


def _make_profile(**kw):
    defaults = dict(
        insider_name="Tim Cook",
        ticker="AAPL",
        avg_transaction_size=150_000.0,
        avg_frequency_days=30.0,
        total_transactions=20,
        typical_sell_percentage=0.05,
        last_transaction_date=date.today() - timedelta(days=5),
    )
    defaults.update(kw)
    return InsiderProfile(**defaults)


@pytest.fixture
def store():
    s = AsyncMock()
    s.save_anomaly = AsyncMock(return_value=1)
    return s


@pytest.fixture
def analyzer(store):
    return InsiderAnalyzer(store)


class TestTier1VolumeAnomaly:
    def test_detects_large_volume(self, analyzer):
        # Build history of small trades + one massive outlier
        txns = [_make_tx(days_ago=i * 30, shares=1000, price=150.0) for i in range(5)]
        txns.insert(0, _make_tx(days_ago=0, shares=50000, price=150.0))
        profile = _make_profile()

        anomalies = analyzer._tier1_detect(txns, profile, "AAPL")
        volume_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.VOLUME]
        assert len(volume_anomalies) >= 1

    def test_no_volume_anomaly_for_normal_trades(self, analyzer):
        txns = [_make_tx(days_ago=i * 30, shares=1000, price=150.0) for i in range(5)]
        profile = _make_profile()
        anomalies = analyzer._tier1_detect(txns, profile, "AAPL")
        volume_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.VOLUME]
        assert len(volume_anomalies) == 0


class TestTier1HoldingsPercentage:
    def test_detects_large_pct_sale(self, analyzer):
        tx = _make_tx(shares=40000, shares_owned_after=10000)
        profile = _make_profile()
        anomalies = analyzer._tier1_detect([tx], profile, "AAPL")
        pct_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.HOLDINGS_PERCENTAGE]
        assert len(pct_anomalies) == 1
        assert pct_anomalies[0].severity_score == pytest.approx(0.8, abs=0.01)


class TestClusterSelling:
    @pytest.mark.asyncio
    async def test_detects_cluster(self, analyzer, store):
        store.get_recent_sellers = AsyncMock(return_value=["A", "B", "C"])
        anomaly = await analyzer._detect_cluster_selling("AAPL")
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.CLUSTER

    @pytest.mark.asyncio
    async def test_no_cluster(self, analyzer, store):
        store.get_recent_sellers = AsyncMock(return_value=["A"])
        anomaly = await analyzer._detect_cluster_selling("AAPL")
        assert anomaly is None


class TestTier2IsolationForest:
    def test_with_sufficient_data(self, analyzer):
        txns = [_make_tx(days_ago=i * 15, shares=1000 + i * 100, price=150.0) for i in range(15)]
        score = analyzer._tier2_score(txns)
        assert 0.0 <= score <= 1.0

    def test_insufficient_data_returns_zero(self, analyzer):
        txns = [_make_tx()]
        assert analyzer._tier2_score(txns) == 0.0


class TestSentimentDerivation:
    def test_bearish_on_high_score_and_sells(self):
        txns = [_make_tx(code=TransactionCode.SALE) for _ in range(5)]
        sentiment = InsiderAnalyzer._derive_sentiment(0.8, txns)
        assert sentiment == InsiderSentiment.BEARISH

    def test_bullish_on_buys(self):
        txns = [_make_tx(code=TransactionCode.PURCHASE) for _ in range(5)]
        sentiment = InsiderAnalyzer._derive_sentiment(0.3, txns)
        assert sentiment == InsiderSentiment.BULLISH

    def test_neutral_default(self):
        txns = []
        sentiment = InsiderAnalyzer._derive_sentiment(0.3, txns)
        assert sentiment == InsiderSentiment.NEUTRAL


class TestAnalyzeTicker:
    @pytest.mark.asyncio
    async def test_empty_transactions(self, analyzer, store):
        store.get_transactions = AsyncMock(return_value=[])
        signal = await analyzer.analyze_ticker("AAPL")
        assert signal.ticker == "AAPL"
        assert signal.anomaly_score == 0.0

    @pytest.mark.asyncio
    async def test_full_analysis(self, analyzer, store):
        txns = [_make_tx(days_ago=i * 15, shares=1000 + i * 100) for i in range(10)]
        store.get_transactions = AsyncMock(return_value=txns)
        store.get_profile = AsyncMock(return_value=_make_profile())
        store.get_recent_sellers = AsyncMock(return_value=["A"])

        signal = await analyzer.analyze_ticker("AAPL")
        assert signal.ticker == "AAPL"
        assert 0.0 <= signal.anomaly_score <= 1.0
