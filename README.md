# SEC Alpha-Sentinel

A production-grade hybrid RAG & agentic pipeline for extracting "Alpha Signals" from SEC 10-K/10-Q filings.

## Architecture

1.  **Ingestion**: `edgar-python` downloads filings; `FilingProcessor` splits them by section (Item 1A, Item 7, etc.).
2.  **Storage**: `Qdrant` stores dense vectors (embedded via `sentence-transformers`).
3.  **Agents**:
    *   **Analyst**: Synthesizes retrieved chunks into a signal (Risk/Sentiment).
    *   **Critic**: Reviews the Analyst's output for hallucinations.
4.  **Serving**: FastAPI exposes the analysis engine.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -e .
    ```

2.  **Environment Variables**:
    Create a `.env` file:
    ```bash
    ANTHROPIC_API_KEY=sk-ant-...
    QDRANT_HOST=localhost
    QDRANT_PORT=6333
    SEC_USER_AGENT="Your Name your@email.com"
    ```

3.  **Run Qdrant** (if using local docker):
    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```

## Usage

### 1. Ingest Data (CLI)
Download and index the latest 10-K for Apple (AAPL).
```bash
python src/cli.py ingest --ticker AAPL --limit 1
```

### 2. Search (CLI)
Test the retrieval engine.
```bash
python src/cli.py search --query "risk factors china"
```

### 3. Run API
Start the server.
```bash
uvicorn src.main:app --reload
```
Then POST to `http://localhost:8000/api/v1/analyze/AAPL`

## Project Structure
- `src/core`: Configuration & Logging
- `src/services`: External integrations (SEC, Qdrant, LLM)
- `src/agents`: Reasoning logic (Analyst/Critic)
- `src/models`: Pydantic schemas
