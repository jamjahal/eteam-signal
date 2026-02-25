import asyncio
from typing import List, Optional
from datetime import date, timedelta

from edgar import Company, set_identity

from src.core.config import settings
from src.core.logger import get_logger
from src.models.insider_schema import InsiderTransaction, TransactionCode

log = get_logger(__name__)

_CODE_MAP = {v.value: v for v in TransactionCode}


class InsiderClient:
    """Fetches and parses SEC Form 4 filings via edgartools."""

    def __init__(self) -> None:
        set_identity(settings.SEC_USER_AGENT)
        self._rate_delay = 1.0 / settings.INSIDER_INGEST_RATE_LIMIT

    def fetch_form4_filings(self, ticker: str, limit: int = 20) -> list:
        """Return raw Form 4 filing objects from edgartools."""
        try:
            company = Company(ticker)
            filings = company.get_filings(form="4").latest(limit)
            if limit == 1 and filings:
                return [filings]
            return list(filings) if filings else []
        except Exception as e:
            log.error("Failed to fetch Form 4 filings", ticker=ticker, error=str(e))
            return []

    def parse_form4(self, filing, ticker: str) -> List[InsiderTransaction]:
        """Parse a single Form 4 filing into InsiderTransaction models."""
        try:
            form4 = filing.obj()
        except Exception as e:
            log.warning("Could not parse Form 4 object", error=str(e))
            return []

        transactions: List[InsiderTransaction] = []

        owner_name = ""
        is_officer = False
        is_director = False
        title = ""
        try:
            owner = form4.reporting_owner if hasattr(form4, "reporting_owner") else None
            if owner:
                owner_name = getattr(owner, "name", "") or ""
                is_officer = bool(getattr(owner, "is_officer", False))
                is_director = bool(getattr(owner, "is_director", False))
                title = getattr(owner, "officer_title", "") or ""
        except Exception:
            pass

        filing_date_val = self._coerce_date(getattr(filing, "filing_date", None)) or date.today()

        raw_txns = []
        if hasattr(form4, "transactions") and form4.transactions:
            raw_txns = form4.transactions if isinstance(form4.transactions, list) else [form4.transactions]

        for raw in raw_txns:
            try:
                tx_date = self._coerce_date(getattr(raw, "transaction_date", None)) or filing_date_val
                raw_code = str(getattr(raw, "transaction_code", "O")).strip().upper()
                tx_code = _CODE_MAP.get(raw_code, TransactionCode.OTHER)

                shares = float(getattr(raw, "shares", 0) or 0)
                price = self._safe_float(getattr(raw, "price_per_share", None))
                total_value = shares * price if price else None
                owned_after = self._safe_float(getattr(raw, "shares_owned_following_transaction", None))
                is_planned = bool(getattr(raw, "is_10b5_1", False))

                transactions.append(
                    InsiderTransaction(
                        ticker=ticker,
                        insider_name=owner_name,
                        insider_title=title,
                        is_officer=is_officer,
                        is_director=is_director,
                        transaction_date=tx_date,
                        transaction_code=tx_code,
                        shares=shares,
                        price_per_share=price,
                        total_value=total_value,
                        shares_owned_after=owned_after,
                        is_10b5_1=is_planned,
                        filing_date=filing_date_val,
                    )
                )
            except Exception as e:
                log.warning("Skipping malformed transaction", error=str(e))

        return transactions

    async def batch_fetch(
        self,
        tickers: List[str],
        days_back: int = 90,
        filings_per_ticker: int = 20,
    ) -> List[InsiderTransaction]:
        """Iterate over tickers with rate limiting, returning all parsed transactions."""
        all_txns: List[InsiderTransaction] = []
        cutoff = date.today() - timedelta(days=days_back)

        for ticker in tickers:
            filings = self.fetch_form4_filings(ticker, limit=filings_per_ticker)
            for f in filings:
                txns = self.parse_form4(f, ticker)
                for tx in txns:
                    if tx.transaction_date >= cutoff:
                        all_txns.append(tx)
            await asyncio.sleep(self._rate_delay)

        log.info("Batch fetch complete", tickers=len(tickers), transactions=len(all_txns))
        return all_txns

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_date(val) -> Optional[date]:
        if val is None:
            return None
        if isinstance(val, date):
            return val
        try:
            return date.fromisoformat(str(val)[:10])
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
