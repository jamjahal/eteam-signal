import asyncio
from typing import List, Optional, Set
from datetime import date, datetime, time

import httpx
from lxml import etree

from src.core.config import settings
from src.core.logger import get_logger
from src.services.insider_client import InsiderClient
from src.services.insider_store import InsiderStore

log = get_logger(__name__)

ATOM_FEED_URL = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q=%224%22&dateRange=custom&startdt={start}&enddt={end}&forms=4"
)

FEED_NAME = "form4_atom"


class FilingMonitor:
    """
    Dual-path ingestion monitor for Form 4 filings.

    Path A: polls the SEC EDGAR ATOM feed for near-real-time detection.
    Path B: scheduled batch sweep as a safety net.
    """

    def __init__(
        self,
        store: InsiderStore,
        client: InsiderClient,
        universe: List[str],
        on_new_filings: Optional[callable] = None,
    ) -> None:
        self.store = store
        self.client = client
        self.universe: Set[str] = set(t.upper() for t in universe)
        self.on_new_filings = on_new_filings
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Launch both the ATOM poller and the batch scheduler."""
        self._running = True
        self._tasks = [
            asyncio.create_task(self._atom_poll_loop()),
            asyncio.create_task(self._batch_sweep_loop()),
        ]
        log.info(
            "FilingMonitor started",
            universe_size=len(self.universe),
            atom_interval_market=settings.INSIDER_ATOM_POLL_INTERVAL_MARKET,
            batch_interval_min=settings.INSIDER_BATCH_INTERVAL_MINUTES,
        )

    async def stop(self) -> None:
        """Gracefully shut down both background loops."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        log.info("FilingMonitor stopped")

    # ------------------------------------------------------------------
    # Path A: ATOM Feed Poller
    # ------------------------------------------------------------------

    async def _atom_poll_loop(self) -> None:
        while self._running:
            try:
                await self._poll_atom_feed()
            except Exception as e:
                log.error("ATOM poll error", error=str(e))

            interval = self._current_poll_interval()
            await asyncio.sleep(interval)

    async def _poll_atom_feed(self) -> None:
        today = date.today().isoformat()
        url = ATOM_FEED_URL.format(start=today, end=today)

        async with httpx.AsyncClient(
            headers={"User-Agent": settings.SEC_USER_AGENT},
            timeout=30.0,
        ) as http:
            resp = await http.get(url)
            resp.raise_for_status()

        watermark = await self.store.get_watermark(FEED_NAME)
        new_accessions = self._parse_feed_entries(resp.text, watermark)

        if not new_accessions:
            return

        log.info("ATOM feed: new Form 4 entries", count=len(new_accessions))

        latest_accession = new_accessions[0]
        for accession in new_accessions:
            await self._process_accession(accession)

        await self.store.set_watermark(FEED_NAME, latest_accession)

    def _parse_feed_entries(self, xml_text: str, watermark: Optional[str]) -> List[str]:
        """Extract accession numbers from ATOM feed XML, stopping at the watermark."""
        accessions: List[str] = []
        try:
            root = etree.fromstring(xml_text.encode())
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall(".//atom:entry", ns):
                acc_el = entry.find("atom:id", ns)
                if acc_el is None or acc_el.text is None:
                    continue
                acc = acc_el.text.strip()
                if acc == watermark:
                    break
                accessions.append(acc)
        except Exception as e:
            log.warning("Failed to parse ATOM feed XML", error=str(e))
        return accessions

    async def _process_accession(self, accession: str) -> None:
        """Fetch and store a single filing by accession number (placeholder path)."""
        # The ATOM feed provides accession numbers; we resolve tickers via the filing.
        # For the initial implementation we rely on the batch sweep for full coverage
        # and use this path for tickers we can match from the feed metadata.
        pass

    def _current_poll_interval(self) -> int:
        now = datetime.now().time()
        market_open = time(9, 30)
        market_close = time(16, 0)
        if market_open <= now <= market_close:
            return settings.INSIDER_ATOM_POLL_INTERVAL_MARKET
        return settings.INSIDER_ATOM_POLL_INTERVAL_OFF

    # ------------------------------------------------------------------
    # Path B: Scheduled Batch Sweep
    # ------------------------------------------------------------------

    async def _batch_sweep_loop(self) -> None:
        interval_sec = settings.INSIDER_BATCH_INTERVAL_MINUTES * 60
        while self._running:
            try:
                await self._run_batch_sweep()
            except Exception as e:
                log.error("Batch sweep error", error=str(e))
            await asyncio.sleep(interval_sec)

    async def _run_batch_sweep(self) -> None:
        """Fetch Form 4 filings for the full universe with overlapping window."""
        overlap_days = max(1, settings.INSIDER_BATCH_OVERLAP_HOURS // 24 + 1)
        tickers = sorted(self.universe)
        log.info("Starting batch sweep", tickers=len(tickers), overlap_days=overlap_days)

        txns = await self.client.batch_fetch(tickers, days_back=overlap_days)
        inserted = await self.store.upsert_transactions(txns)
        log.info("Batch sweep complete", new_transactions=inserted, total_fetched=len(txns))

        if inserted > 0 and self.on_new_filings:
            await self.on_new_filings(txns[:inserted])
