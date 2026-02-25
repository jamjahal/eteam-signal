import pytest
from unittest.mock import Mock, AsyncMock

from src.agents.composite_signal import CompositeSignalEngine
from src.models.schema import AlphaSignal
from src.models.insider_schema import (
    InsiderSignal,
    InsiderSentiment,
    InsiderAnomaly,
    AnomalyType,
)


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.generate = AsyncMock(return_value="Strong sell signal. Insiders dumping shares while filings show risk.")
    return llm


@pytest.fixture
def engine(mock_llm):
    return CompositeSignalEngine(llm_client=mock_llm)


class TestBlendScores:
    def test_even_blend(self, engine):
        result = engine._blend_scores(0.8, 0.6)
        # Both > 0.5 triggers convergent boost: 0.7 * 1.2 = 0.84
        assert result == pytest.approx(0.84, abs=0.01)

    def test_convergent_boost(self, engine):
        result = engine._blend_scores(0.8, 0.8)
        # 0.5*0.8 + 0.5*0.8 = 0.8, boosted by 1.2 = 0.96
        assert result > 0.8

    def test_capped_at_one(self, engine):
        result = engine._blend_scores(1.0, 1.0)
        assert result <= 1.0

    def test_low_scores_no_boost(self, engine):
        result = engine._blend_scores(0.2, 0.3)
        assert result == pytest.approx(0.25, abs=0.01)


class TestCompose:
    @pytest.mark.asyncio
    async def test_with_both_signals(self, engine):
        filing = AlphaSignal(
            ticker="AAPL",
            signal_score=0.7,
            confidence=0.8,
            summary="Negative sentiment drift",
        )
        insider = InsiderSignal(
            ticker="AAPL",
            anomaly_score=0.8,
            insider_sentiment=InsiderSentiment.BEARISH,
            anomalies=[
                InsiderAnomaly(
                    ticker="AAPL",
                    insider_name="X",
                    anomaly_type=AnomalyType.CLUSTER,
                    severity_score=0.9,
                )
            ],
        )
        result = await engine.compose("AAPL", filing, insider)
        assert result.composite_alpha_score is not None
        assert result.composite_alpha_score > 0.5
        assert len(result.recommendation) > 0

    @pytest.mark.asyncio
    async def test_insider_only(self, engine):
        insider = InsiderSignal(ticker="AAPL", anomaly_score=0.6)
        result = await engine.compose("AAPL", None, insider)
        assert result.composite_alpha_score is not None
        assert result.composite_alpha_score == pytest.approx(0.3, abs=0.01)

    @pytest.mark.asyncio
    async def test_no_signals(self, engine):
        result = await engine.compose("AAPL", None, None)
        assert result.composite_alpha_score == 0.0


class TestFallbackRecommendation:
    def test_strong_sell(self):
        sig = InsiderSignal(
            ticker="AAPL",
            anomaly_score=0.9,
            insider_sentiment=InsiderSentiment.BEARISH,
            anomalies=[],
        )
        text = CompositeSignalEngine._fallback_recommendation("AAPL", sig, 0.8)
        assert "Strong sell" in text

    def test_no_action(self):
        sig = InsiderSignal(ticker="AAPL")
        text = CompositeSignalEngine._fallback_recommendation("AAPL", sig, 0.2)
        assert "No immediate action" in text
