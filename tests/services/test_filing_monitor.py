import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import date

from src.services.filing_monitor import FilingMonitor


@pytest.fixture
def monitor():
    store = AsyncMock()
    client = Mock()
    universe = ["AAPL", "MSFT", "GOOG"]
    return FilingMonitor(store, client, universe)


class TestParseFeedEntries:
    def test_extracts_accessions(self, monitor):
        xml = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry><id>0001234567-25-000001</id></entry>
            <entry><id>0001234567-25-000002</id></entry>
        </feed>"""
        result = monitor._parse_feed_entries(xml, None)
        assert len(result) == 2
        assert result[0] == "0001234567-25-000001"

    def test_stops_at_watermark(self, monitor):
        xml = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry><id>0001234567-25-000003</id></entry>
            <entry><id>0001234567-25-000002</id></entry>
            <entry><id>0001234567-25-000001</id></entry>
        </feed>"""
        result = monitor._parse_feed_entries(xml, "0001234567-25-000002")
        assert len(result) == 1
        assert result[0] == "0001234567-25-000003"

    def test_empty_feed(self, monitor):
        xml = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
        result = monitor._parse_feed_entries(xml, None)
        assert result == []

    def test_malformed_xml(self, monitor):
        result = monitor._parse_feed_entries("not xml", None)
        assert result == []


class TestUniverse:
    def test_universe_uppercased(self, monitor):
        assert "AAPL" in monitor.universe
        assert "aapl" not in monitor.universe


class TestPollInterval:
    @patch("src.services.filing_monitor.datetime")
    def test_market_hours(self, mock_dt, monitor):
        from datetime import time as dt_time

        mock_now = Mock()
        mock_now.time.return_value = dt_time(12, 0)
        mock_dt.now.return_value = mock_now
        assert monitor._current_poll_interval() == 300

    @patch("src.services.filing_monitor.datetime")
    def test_off_hours(self, mock_dt, monitor):
        from datetime import time as dt_time

        mock_now = Mock()
        mock_now.time.return_value = dt_time(22, 0)
        mock_dt.now.return_value = mock_now
        assert monitor._current_poll_interval() == 1800
