"""
InsiderStore unit tests with mocked asyncpg pool.

Integration tests against a real TimescaleDB instance should use testcontainers
and live in a separate test_insider_store_integration.py.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date

from src.services.insider_store import InsiderStore
from src.models.insider_schema import (
    InsiderTransaction,
    InsiderAnomaly,
    AnomalyType,
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


@pytest.fixture
def store():
    s = InsiderStore()
    s._pool = AsyncMock()
    return s


class TestUpsertTransaction:
    @pytest.mark.asyncio
    async def test_insert_returns_true(self, store):
        store._pool.fetchrow = AsyncMock(return_value={"id": 1})
        result = await store.upsert_transaction(_make_tx())
        assert result is True

    @pytest.mark.asyncio
    async def test_duplicate_returns_false(self, store):
        store._pool.fetchrow = AsyncMock(return_value=None)
        result = await store.upsert_transaction(_make_tx())
        assert result is False


class TestUpsertTransactions:
    @pytest.mark.asyncio
    async def test_counts_insertions(self, store):
        store._pool.fetchrow = AsyncMock(side_effect=[{"id": 1}, None, {"id": 2}])
        txns = [_make_tx(), _make_tx(), _make_tx()]
        inserted = await store.upsert_transactions(txns)
        assert inserted == 2


class TestGetTransactions:
    @pytest.mark.asyncio
    async def test_returns_parsed_models(self, store):
        mock_row = {
            "ticker": "AAPL",
            "insider_name": "Tim Cook",
            "insider_title": "CEO",
            "is_officer": True,
            "is_director": False,
            "transaction_date": date(2025, 6, 1),
            "transaction_code": "S",
            "shares": 10000.0,
            "price_per_share": 190.0,
            "total_value": 1_900_000.0,
            "shares_owned_after": 50000.0,
            "is_10b5_1": False,
            "filing_date": date(2025, 6, 3),
        }
        store._pool.fetch = AsyncMock(return_value=[mock_row])
        txns = await store.get_transactions("AAPL")
        assert len(txns) == 1
        assert txns[0].ticker == "AAPL"
        assert txns[0].transaction_code == TransactionCode.SALE


class TestWatermark:
    @pytest.mark.asyncio
    async def test_get_watermark_none(self, store):
        store._pool.fetchrow = AsyncMock(return_value=None)
        assert await store.get_watermark("form4_atom") is None

    @pytest.mark.asyncio
    async def test_get_watermark_exists(self, store):
        store._pool.fetchrow = AsyncMock(
            return_value={"last_seen_accession": "0001-23-456"}
        )
        assert await store.get_watermark("form4_atom") == "0001-23-456"

    @pytest.mark.asyncio
    async def test_set_watermark(self, store):
        store._pool.execute = AsyncMock()
        await store.set_watermark("form4_atom", "0001-23-456")
        store._pool.execute.assert_called_once()


class TestSaveAnomaly:
    @pytest.mark.asyncio
    async def test_saves_and_returns_id(self, store):
        store._pool.fetchval = AsyncMock(return_value=42)
        anomaly = InsiderAnomaly(
            ticker="AAPL",
            insider_name="Tim Cook",
            anomaly_type=AnomalyType.VOLUME,
            severity_score=0.85,
            z_score=2.5,
            description="test",
        )
        result = await store.save_anomaly(anomaly)
        assert result == 42
