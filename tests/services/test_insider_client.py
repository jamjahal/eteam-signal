import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import date

from src.services.insider_client import InsiderClient
from src.models.insider_schema import TransactionCode


@pytest.fixture
def client():
    with patch("src.services.insider_client.set_identity"):
        return InsiderClient()


class TestFetchForm4Filings:
    @patch("src.services.insider_client.Company")
    def test_returns_list(self, mock_company_cls, client):
        mock_filings = MagicMock()
        mock_filings.latest.return_value = [Mock(), Mock()]
        mock_company_cls.return_value.get_filings.return_value = mock_filings

        result = client.fetch_form4_filings("AAPL", limit=2)
        assert len(result) == 2

    @patch("src.services.insider_client.Company")
    def test_single_filing_normalised(self, mock_company_cls, client):
        single = Mock()
        mock_filings = MagicMock()
        mock_filings.latest.return_value = single
        mock_company_cls.return_value.get_filings.return_value = mock_filings

        result = client.fetch_form4_filings("AAPL", limit=1)
        assert len(result) == 1
        assert result[0] is single

    @patch("src.services.insider_client.Company")
    def test_handles_exception(self, mock_company_cls, client):
        mock_company_cls.side_effect = Exception("API error")
        result = client.fetch_form4_filings("BAD")
        assert result == []


class TestParseForm4:
    def _mock_filing(self, tx_code="S", shares=1000, price=150.0, owned_after=5000):
        filing = Mock()
        filing.filing_date = "2025-06-01"

        form4 = Mock()
        owner = Mock()
        owner.name = "Jane Doe"
        owner.is_officer = True
        owner.is_director = False
        owner.officer_title = "CFO"
        form4.reporting_owner = owner

        tx = Mock()
        tx.transaction_date = "2025-05-30"
        tx.transaction_code = tx_code
        tx.shares = shares
        tx.price_per_share = price
        tx.shares_owned_following_transaction = owned_after
        tx.is_10b5_1 = False
        form4.transactions = [tx]

        filing.obj.return_value = form4
        return filing

    def test_parses_sale(self, client):
        filing = self._mock_filing()
        txns = client.parse_form4(filing, "AAPL")
        assert len(txns) == 1
        assert txns[0].transaction_code == TransactionCode.SALE
        assert txns[0].insider_name == "Jane Doe"
        assert txns[0].shares == 1000

    def test_parses_purchase(self, client):
        filing = self._mock_filing(tx_code="P")
        txns = client.parse_form4(filing, "AAPL")
        assert txns[0].transaction_code == TransactionCode.PURCHASE

    def test_handles_missing_price(self, client):
        filing = self._mock_filing(price=None)
        txns = client.parse_form4(filing, "AAPL")
        assert txns[0].price_per_share is None
        assert txns[0].total_value is None

    def test_handles_parse_failure(self, client):
        filing = Mock()
        filing.obj.side_effect = Exception("parse error")
        filing.filing_date = "2025-06-01"
        txns = client.parse_form4(filing, "AAPL")
        assert txns == []
