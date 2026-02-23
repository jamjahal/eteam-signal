import os
from typing import List, Optional, Any
from datetime import date
from edgar import Company, set_identity
from structlog import get_logger
from src.core.config import settings

log = get_logger(__name__)

class SECClient:
    """
    Client for interacting with the SEC EDGAR database via edgar-python.
    """
    def __init__(self):
        if not settings.SEC_USER_AGENT:
            raise ValueError("SEC_USER_AGENT must be set in settings.")
        set_identity(settings.SEC_USER_AGENT)
        log.info("Initialized SEC Client", user_agent=settings.SEC_USER_AGENT)

    def get_latest_filings(self, ticker: str, form_type: str = "10-K", limit: int = 1) -> List[Any]: # using Any for edgar objects
        """
        Fetch the latest N filings of a specific type for a ticker.
        """
        log.info("Fetching filings", ticker=ticker, form_type=form_type, limit=limit)
        try:
            company = Company(ticker)
            filings = company.get_filings(form=form_type).latest(limit)
            
            # If limit is 1, it returns a single object, not a list. Normalize.
            if limit == 1 and filings:
                 return [filings]
            elif not filings:
                return []
            
            return filings
        except Exception as e:
            log.error("Failed to fetch filings", ticker=ticker, error=str(e))
            raise

    def download_html(self, filing) -> str:
        """
        Download the HTML content of a filing.
        """
        try:
            return filing.html()
        except Exception as e:
            log.error("Failed to download HTML", accession_number=filing.accession_no, error=str(e))
            raise
