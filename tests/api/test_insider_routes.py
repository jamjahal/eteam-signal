import pytest
from unittest.mock import AsyncMock, patch, Mock
from fastapi.testclient import TestClient

from src.main import app
from src.api.insider_routes import set_store
from src.models.insider_schema import AnomalyType


@pytest.fixture
def mock_store():
    store = AsyncMock()
    store.get_anomalies = AsyncMock(return_value=[])
    store.get_alerts = AsyncMock(return_value=[])
    store.get_profile = AsyncMock(return_value=None)
    set_store(store)
    yield store
    set_store(None)


@pytest.fixture
def client(mock_store):
    return TestClient(app, raise_server_exceptions=False)


class TestInsiderRoutes:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_get_anomalies_empty(self, client, mock_store):
        resp = client.get("/api/v1/insider/anomalies/AAPL")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_all_anomalies(self, client, mock_store):
        resp = client.get("/api/v1/insider/anomalies?min_score=0.5")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_profile_not_found(self, client, mock_store):
        resp = client.get("/api/v1/insider/profile/AAPL/John%20Doe")
        assert resp.status_code == 404

    def test_get_alerts_empty(self, client, mock_store):
        mock_store.get_alerts = AsyncMock(return_value=[])
        resp = client.get("/api/v1/insider/alerts")
        assert resp.status_code == 200

    def test_store_not_initialized(self):
        set_store(None)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/v1/insider/anomalies/AAPL")
        assert resp.status_code == 503
