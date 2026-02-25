import pytest
from unittest.mock import AsyncMock

from src.services.alert_service import AlertService
from src.models.insider_schema import InsiderSignal, InsiderSentiment


@pytest.fixture
def store():
    s = AsyncMock()
    s.save_alert = AsyncMock(return_value=1)
    return s


@pytest.fixture
def service(store):
    return AlertService(store)


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_filters_below_threshold(self, service, store):
        signals = [
            InsiderSignal(ticker="AAPL", anomaly_score=0.3),
            InsiderSignal(ticker="GOOG", anomaly_score=0.8, insider_sentiment=InsiderSentiment.BEARISH),
        ]
        actionable = await service.evaluate(signals)
        assert len(actionable) == 1
        assert actionable[0].ticker == "GOOG"
        store.save_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_below_threshold(self, service, store):
        signals = [InsiderSignal(ticker="AAPL", anomaly_score=0.1)]
        actionable = await service.evaluate(signals)
        assert len(actionable) == 0
        store.save_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_signals(self, service, store):
        actionable = await service.evaluate([])
        assert actionable == []


class TestGetActiveAlerts:
    @pytest.mark.asyncio
    async def test_delegates_to_store(self, service, store):
        store.get_alerts = AsyncMock(return_value=[])
        result = await service.get_active_alerts()
        store.get_alerts.assert_called_once_with(delivered=False, limit=50)
        assert result == []
