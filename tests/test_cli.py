import pytest
from typer.testing import CliRunner
from unittest.mock import patch, Mock
from src.cli import app

runner = CliRunner()

import sys
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from src.cli import app

runner = CliRunner()

def test_ingest_command():
    # To properly mock the lazy imports, we can mock the imports using sys.modules patching.
    # However, since we are patching modules that might not be loaded yet, we need to be careful.
    
    mock_sec_module = MagicMock()
    mock_processor_module = MagicMock()
    mock_retriever_module = MagicMock()
    
    # We need to structure the mocks so that `from src.services.sec_client import SECClient` works.
    # This means mock_sec_module.SECClient must be the class.
    
    mock_client_cls = mock_sec_module.SECClient
    mock_client_instance = mock_client_cls.return_value
    
    mock_filing = MagicMock()
    mock_filing.filing_date = "2023-01-01"
    mock_filing.accession_no = "123"
    mock_client_instance.get_latest_filings.return_value = [mock_filing]
    mock_client_instance.download_html.return_value = "<html></html>"
    
    mock_processor_cls = mock_processor_module.FilingProcessor
    mock_processor_instance = mock_processor_cls.return_value
    mock_processor_instance.process_html.return_value = []
    
    # Patch the dictionary of modules. Note that we need to patch the EXACT module path used in import.
    with patch.dict(sys.modules, {
        "src.services.sec_client": mock_sec_module,
        "src.services.processor": mock_processor_module,
        "src.services.retriever": mock_retriever_module
    }):
        result = runner.invoke(app, ["ingest", "--ticker", "AAPL"])
        
        if result.exit_code != 0:
            print(result.stdout)
            print(result.exception)
            
        assert result.exit_code == 0
        assert "Starting ingestion" in result.stdout

def test_search_command():
    mock_retriever_module = MagicMock()
    mock_retriever_cls = mock_retriever_module.Retriever
    mock_retriever_instance = mock_retriever_cls.return_value
    
    mock_chunk = MagicMock()
    mock_chunk.section_name = "Risk"
    mock_chunk.filing_date = "2023-01-01"
    mock_chunk.content = "Dangerous"
    
    mock_result = MagicMock()
    mock_result.score = 0.99
    mock_result.chunk = mock_chunk
    
    mock_retriever_instance.search.return_value = [mock_result]
    
    with patch.dict(sys.modules, {
        "src.services.retriever": mock_retriever_module
    }):
        result = runner.invoke(app, ["search", "--query", "risk"])
        
        assert result.exit_code == 0
        assert "[0.9900] Risk" in result.stdout
