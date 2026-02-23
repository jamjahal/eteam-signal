import pytest
from unittest.mock import Mock, patch
from src.services.sec_client import SECClient
from src.core.config import settings

# Mock the settings to ensure USER_AGENT is set
settings.SEC_USER_AGENT = "Test Agent test@example.com"

class TestSECClient:
    
    @patch("src.services.sec_client.Company")
    def test_get_latest_filings_single(self, mock_company_cls):
        # Setup
        mock_company_instance = Mock()
        mock_filings = Mock()
        
        # Configure the chain: Company(ticker).get_filings().latest(limit)
        mock_company_cls.return_value = mock_company_instance
        mock_company_instance.get_filings.return_value = mock_filings
        
        # return a single object when limit=1 (simulating edgar-python behavior mostly)
        mock_single_filing = Mock()
        mock_filings.latest.return_value = mock_single_filing
        
        client = SECClient()
        result = client.get_latest_filings("AAPL", limit=1)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == mock_single_filing
        
        mock_company_cls.assert_called_with("AAPL")

    @patch("src.services.sec_client.Company")
    def test_get_latest_filings_empty(self, mock_company_cls):
        mock_company_instance = Mock()
        mock_filings = Mock()
        mock_company_cls.return_value = mock_company_instance
        mock_company_instance.get_filings.return_value = mock_filings
        mock_filings.latest.return_value = None
        
        client = SECClient()
        result = client.get_latest_filings("AAPL", limit=1)
        
        assert result == []

    def test_download_html(self):
        client = SECClient()
        mock_filing = Mock()
        mock_filing.html.return_value = "<html><body>Content</body></html>"
        
        html = client.download_html(mock_filing)
        assert html == "<html><body>Content</body></html>"
        mock_filing.html.assert_called_once()
