from fastapi import APIRouter, HTTPException, BackgroundTasks
from src.models.schema import AlphaSignal
from src.agents.workflow import AgentWorkflow
from src.services.retriever import Retriever
from src.services.sec_client import SECClient
from src.services.processor import FilingProcessor
from datetime import date

router = APIRouter()

@router.post("/analyze/{ticker}", response_model=AlphaSignal)
async def analyze_ticker(ticker: str):
    """
    Full pipeline: Retrieve (or Ingest) -> Analyze -> Critique.
    Note: In a real prod system, ingestion would be async/backgrounded.
    Here we assume data might already be indexed or we do a quick lookup.
    """
    # 1. Retrieve relevant chunks
    # For this demo, we assume we search for "Risk Factors" and "Management Discussion"
    retriever = Retriever()
    
    # We try to search first. If no results, we might need to trigger ingestion (not implemented in this sync route for speed)
    # Let's search for "Risk Factors"
    results = retriever.search(f"{ticker} Risk Factors", limit=5)
    
    if not results:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Please ingest first via CLI.")
        
    chunks = [r.chunk for r in results]
    
    # 2. Run Agents
    workflow = AgentWorkflow()
    signal = await workflow.run_analysis(ticker, chunks)
    
    return signal
