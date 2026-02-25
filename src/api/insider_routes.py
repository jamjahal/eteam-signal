from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.core.logger import get_logger
from src.models.insider_schema import InsiderAnomaly, InsiderSignal
from src.services.insider_store import InsiderStore
from src.services.insider_client import InsiderClient
from src.agents.insider_analyzer import InsiderAnalyzer
from src.agents.composite_signal import CompositeSignalEngine

log = get_logger(__name__)

insider_router = APIRouter(prefix="/insider", tags=["insider"])

_store: Optional[InsiderStore] = None


def get_store() -> InsiderStore:
    if _store is None:
        raise HTTPException(status_code=503, detail="InsiderStore not initialized")
    return _store


def set_store(store: InsiderStore) -> None:
    global _store
    _store = store


@insider_router.post("/ingest")
async def ingest_form4(
    tickers: Optional[List[str]] = None,
    days_back: int = Query(default=90, ge=1, le=365),
):
    """Trigger Form 4 ingestion for specified tickers or the full universe."""
    store = get_store()
    client = InsiderClient()

    if not tickers:
        from src.services.universe import load_universe
        tickers = load_universe()

    txns = await client.batch_fetch(tickers, days_back=days_back)
    inserted = await store.upsert_transactions(txns)
    return {"tickers": len(tickers), "fetched": len(txns), "new": inserted}


@insider_router.get("/anomalies/{ticker}", response_model=List[InsiderAnomaly])
async def get_ticker_anomalies(ticker: str):
    store = get_store()
    return await store.get_anomalies(ticker=ticker.upper())


@insider_router.get("/anomalies", response_model=List[InsiderAnomaly])
async def get_all_anomalies(
    min_score: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=100, ge=1, le=1000),
):
    store = get_store()
    return await store.get_anomalies(min_score=min_score, limit=limit)


@insider_router.get("/profile/{ticker}/{insider_name}")
async def get_insider_profile(ticker: str, insider_name: str):
    store = get_store()
    profile = await store.get_profile(ticker.upper(), insider_name)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@insider_router.get("/signal/{ticker}", response_model=InsiderSignal)
async def get_ticker_signal(ticker: str):
    """Run anomaly analysis and return the composite alpha signal."""
    store = get_store()
    analyzer = InsiderAnalyzer(store)
    insider_signal = await analyzer.analyze_ticker(ticker.upper())

    engine = CompositeSignalEngine()
    # Attempt to retrieve filing signal if available
    filing_signal = None
    try:
        from src.services.retriever import Retriever
        from src.agents.workflow import AgentWorkflow

        retriever = Retriever()
        results = retriever.search(f"{ticker} Risk Factors", limit=5)
        if results:
            workflow = AgentWorkflow()
            filing_signal = await workflow.run_analysis(
                ticker.upper(), [r.chunk for r in results]
            )
    except Exception as e:
        log.warning("Filing signal unavailable, using insider-only", error=str(e))

    return await engine.compose(ticker.upper(), filing_signal, insider_signal)


@insider_router.get("/alerts")
async def get_alerts(limit: int = Query(default=50, ge=1, le=500)):
    store = get_store()
    rows = await store.get_alerts(delivered=False, limit=limit)
    return [dict(r) for r in rows]
