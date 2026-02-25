import asyncio
from typing import Optional

import typer
from datetime import date

from src.core.logger import configure_logger, get_logger

app = typer.Typer()
insider_app = typer.Typer(help="Insider trading anomaly detection commands.")
app.add_typer(insider_app, name="insider")

log = get_logger(__name__)


# ======================================================================
# Existing filing commands
# ======================================================================

@app.command()
def ingest(ticker: str, form_type: str = "10-K", limit: int = 1):
    """Download, process, AND INDEX SEC filings for a given ticker."""
    configure_logger()
    log.info("Starting ingestion", ticker=ticker)

    try:
        from src.services.sec_client import SECClient
        from src.services.processor import FilingProcessor
        from src.services.retriever import Retriever
    except ImportError as e:
        log.error("Failed to import services", error=str(e))
        raise typer.Exit(code=1)

    client = SECClient()
    processor = FilingProcessor()
    retriever = Retriever()

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
                filing_date=(
                    date.fromisoformat(filing.filing_date)
                    if isinstance(filing.filing_date, str)
                    else filing.filing_date
                ),
            )

            log.info("Indexing chunks", count=len(chunks))
            retriever.index_filing(chunks)
            log.info("Finished indexing filing")

    except Exception as e:
        log.error("Ingestion failed", error=str(e))
        raise typer.Exit(code=1)


@app.command()
def search(query: str, limit: int = 5):
    """Search the indexed filings."""
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


# ======================================================================
# Insider trading commands
# ======================================================================

def _run(coro):
    """Helper to run an async coroutine from the sync Typer context."""
    asyncio.get_event_loop().run_until_complete(coro)


async def _get_store():
    from src.services.insider_store import InsiderStore

    store = InsiderStore()
    await store.connect()
    return store


@insider_app.command("ingest")
def insider_ingest(days_back: int = typer.Option(90, help="Days of history to fetch")):
    """One-shot batch ingest of Form 4 filings for the universe."""
    configure_logger()

    async def _run_ingest():
        from src.services.insider_client import InsiderClient
        from src.services.universe import load_universe

        store = await _get_store()
        try:
            tickers = load_universe()
            if not tickers:
                log.error("No tickers in universe. Run 'insider universe-refresh' first.")
                raise typer.Exit(code=1)

            client = InsiderClient()
            txns = await client.batch_fetch(tickers, days_back=days_back)
            inserted = await store.upsert_transactions(txns)
            log.info("Ingestion complete", fetched=len(txns), new=inserted)
        finally:
            await store.close()

    _run(_run_ingest())


@insider_app.command("analyze")
def insider_analyze(ticker: str = typer.Option(..., help="Ticker to analyze")):
    """Run anomaly detection for a single ticker."""
    configure_logger()

    async def _run_analyze():
        from src.agents.insider_analyzer import InsiderAnalyzer

        store = await _get_store()
        try:
            analyzer = InsiderAnalyzer(store)
            signal = await analyzer.analyze_ticker(ticker.upper())
            print(f"\n{'='*60}")
            print(f"  Insider Signal: {signal.ticker}")
            print(f"  Anomaly Score:  {signal.anomaly_score:.2f}")
            print(f"  Sentiment:      {signal.insider_sentiment.value}")
            print(f"  Anomalies:      {len(signal.anomalies)}")
            for a in signal.anomalies:
                print(f"    [{a.anomaly_type.value}] {a.description} (severity {a.severity_score:.2f})")
            print(f"{'='*60}\n")
        finally:
            await store.close()

    _run(_run_analyze())


@insider_app.command("scan")
def insider_scan(days_back: int = typer.Option(90, help="Days of history to fetch")):
    """Full pipeline: ingest + analyze + alert for entire universe."""
    configure_logger()

    async def _run_scan():
        from src.services.insider_client import InsiderClient
        from src.services.universe import load_universe
        from src.agents.insider_analyzer import InsiderAnalyzer
        from src.services.alert_service import AlertService

        store = await _get_store()
        try:
            tickers = load_universe()
            if not tickers:
                log.error("No tickers in universe.")
                raise typer.Exit(code=1)

            # Ingest
            client = InsiderClient()
            txns = await client.batch_fetch(tickers, days_back=days_back)
            inserted = await store.upsert_transactions(txns)
            log.info("Ingestion phase done", new=inserted)

            # Analyze
            analyzer = InsiderAnalyzer(store)
            signals = []
            for t in tickers:
                sig = await analyzer.analyze_ticker(t)
                if sig.anomaly_score > 0:
                    signals.append(sig)

            # Alert
            alert_svc = AlertService(store)
            actionable = await alert_svc.evaluate(signals)
            for sig in actionable:
                print(f"[ALERT] {sig.ticker}: score={sig.anomaly_score:.2f} sentiment={sig.insider_sentiment.value}")
            if not actionable:
                print("No actionable alerts.")
        finally:
            await store.close()

    _run(_run_scan())


@insider_app.command("monitor")
def insider_monitor():
    """Start the FilingMonitor in foreground (ATOM poller + batch scheduler)."""
    configure_logger()

    async def _run_monitor():
        from src.services.insider_client import InsiderClient
        from src.services.filing_monitor import FilingMonitor
        from src.services.universe import load_universe

        store = await _get_store()
        client = InsiderClient()
        universe = load_universe()
        monitor = FilingMonitor(store, client, universe)

        try:
            await monitor.start()
            log.info("Monitor running. Press Ctrl+C to stop.")
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await monitor.stop()
            await store.close()

    _run(_run_monitor())


@insider_app.command("alerts")
def insider_alerts(limit: int = typer.Option(50, help="Max alerts to display")):
    """Display current active alerts."""
    configure_logger()

    async def _run_alerts():
        store = await _get_store()
        try:
            rows = await store.get_alerts(delivered=False, limit=limit)
            if not rows:
                print("No active alerts.")
                return
            for r in rows:
                print(
                    f"[{r['created_at']}] {r['ticker']}: "
                    f"score={r['anomaly_score']:.2f} "
                    f"sentiment={r['insider_sentiment']} "
                    f"-- {r['recommendation'][:120]}"
                )
        finally:
            await store.close()

    _run(_run_alerts())


@insider_app.command("universe-refresh")
def insider_universe_refresh():
    """Refresh the S&P 500 ticker universe from Wikipedia."""
    configure_logger()

    try:
        import pandas as pd
    except ImportError:
        log.error("pandas is required for universe refresh")
        raise typer.Exit(code=1)

    from src.services.universe import save_universe

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    log.info("Fetching S&P 500 list from Wikipedia")
    tables = pd.read_html(url)
    df = tables[0]

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "ticker": str(row.get("Symbol", "")).strip(),
            "company_name": str(row.get("Security", "")).strip(),
            "sector": str(row.get("GICS Sector", "")).strip(),
            "sub_industry": str(row.get("GICS Sub-Industry", "")).strip(),
        })

    count = save_universe(rows)
    print(f"Saved {count} tickers to universe.")


if __name__ == "__main__":
    app()
