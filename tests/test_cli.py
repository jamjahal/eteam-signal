import sys
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()


def test_ingest_command():
    mock_filing = MagicMock()
    mock_filing.filing_date = "2023-01-01"
    mock_filing.accession_no = "123"
    mock_filing.cik = "320193"

    mock_sec_cls = MagicMock()
    mock_sec_cls.return_value.get_latest_filings.return_value = [mock_filing]
    mock_sec_cls.return_value.download_html.return_value = "<html></html>"

    mock_processor_cls = MagicMock()
    mock_processor_cls.return_value.process_html.return_value = []

    mock_retriever_cls = MagicMock()

    with patch("src.services.sec_client.SECClient", mock_sec_cls), \
         patch("src.services.processor.FilingProcessor", mock_processor_cls), \
         patch("src.services.retriever.Retriever", mock_retriever_cls):
        result = runner.invoke(app, ["ingest", "AAPL"])

        if result.exit_code != 0:
            print(result.stdout)
            if result.exception:
                raise result.exception

        assert result.exit_code == 0


def test_search_command():
    mock_chunk = MagicMock()
    mock_chunk.section_name = "Risk"
    mock_chunk.filing_date = "2023-01-01"
    mock_chunk.content = "Dangerous"

    mock_result = MagicMock()
    mock_result.score = 0.99
    mock_result.chunk = mock_chunk

    mock_retriever_cls = MagicMock()
    mock_retriever_cls.return_value.search.return_value = [mock_result]

    with patch("src.services.retriever.Retriever", mock_retriever_cls):
        result = runner.invoke(app, ["search", "risk"])

        if result.exit_code != 0:
            print(result.stdout)
            if result.exception:
                raise result.exception

        assert result.exit_code == 0
        assert "[0.9900] Risk" in result.stdout
