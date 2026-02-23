import asyncio
import typer
from typing import Optional
from datetime import date
from src.services.sec_client import SECClient
from src.services.processor import FilingProcessor
from src.core.logger import configure_logger, get_logger

app = typer.Typer()
log = get_logger(__name__)

from src.services.retriever import Retriever

@app.command()
def ingest(ticker: str, form_type: str = "10-K", limit: int = 1):
    """
    Download, process, AND INDEX SEC filings for a given ticker.
    """
    configure_logger()
    log.info("Starting ingestion", ticker=ticker)
    
    # Lazily initialize to avoid heavy imports (like torch) unless needed
    try:
        from src.services.sec_client import SECClient
        from src.services.processor import FilingProcessor
        from src.services.retriever import Retriever
    except ImportError as e:
        log.error("Failed to import services", error=str(e))
        raise typer.Exit(code=1)

    client = SECClient()
    processor = FilingProcessor()
    retriever = Retriever() # Initialize retriever (loads model)
    
    try:
        filings = client.get_latest_filings(ticker, form_type, limit)
        if not filings:
            log.warning("No filings found", ticker=ticker)
            return

        for filing in filings:
            log.info("Downloading filing", date=filing.filing_date, accession=filing.accession_no)
            html = client.download_html(filing)
            
            chunks = processor.process_html(
                html_content=html,
                ticker=ticker,
                cik=str(filing.cik),
                form_type=form_type,
                period_end_date=date.today(), 
                filing_date=date.fromisoformat(filing.filing_date) if isinstance(filing.filing_date, str) else filing.filing_date
            )
            
            log.info("Indexing chunks", count=len(chunks))
            retriever.index_filing(chunks)
            log.info("Finished indexing filing")

    except Exception as e:
        log.error("Ingestion failed", error=str(e))
        raise typer.Exit(code=1)

@app.command()
def search(query: str, limit: int = 5):
    """
    Search the indexed filings.
    """
    configure_logger()
    try:
        from src.services.retriever import Retriever
    except ImportError as e:
        log.error("Failed to import services", error=str(e))
        raise typer.Exit(code=1)

    retriever = Retriever()
    results = retriever.search(query, limit)
    
    for res in results:
        print(f"[{res.score:.4f}] {res.chunk.section_name} ({res.chunk.filing_date})")
        print(f"Snippet: {res.chunk.content[:200]}...\n")


if __name__ == "__main__":
    app()
