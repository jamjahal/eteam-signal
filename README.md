# ETeam-Signal

A production-grade hybrid RAG & agentic pipeline for extracting "Alpha Signals" from SEC 10-K/10-Q filings and insider trading data.

## Problem

SEC 10-K and 10-Q filings routinely exceed 100 pages of dense legal and financial prose. Identifying material changes in risk language, management sentiment, or regulatory exposure across filings requires hours of manual review per company -- and the sheer volume makes comprehensive coverage impractical for most teams.

Separately, SEC Form 4 insider-trading disclosures are publicly available but noisy. Executives sell shares for many routine reasons (tax planning, diversification, pre-scheduled 10b5-1 plans), making it difficult to distinguish meaningful selling pressure from background activity.

ETeam-Signal addresses both problems in a single pipeline:

1. **Filing Analysis** -- Retrieval-Augmented Generation (RAG) over section-aware filing chunks, followed by an agentic Analyst/Critic reasoning loop that scores risk and sentiment drift.
2. **Insider Anomaly Detection** -- A two-tier statistical and ML engine that flags abnormal trading patterns (volume spikes, cluster selling, holdings liquidation).
3. **Composite Signal** -- An LLM-powered synthesis that merges filing sentiment with insider anomaly data into a unified Alpha Signal score and natural-language recommendation.

The result is an automated system that surfaces actionable intelligence a human analyst would otherwise spend hours assembling manually.

## Architecture

```
SEC EDGAR ──► FilingProcessor ──► Qdrant (dense vectors)
                                        │
                                        ▼
                                   Retriever
                                        │
                                        ▼
                              ┌─────────────────┐
                              │  AnalystAgent    │ ◄── Claude Sonnet 4
                              │  (signal draft)  │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  CriticAgent     │ ◄── Claude Sonnet 4
                              │  (verification)  │
                              └────────┬────────┘
                                       │
SEC EDGAR Form 4 ──► InsiderAnalyzer ──┤
   (ATOM feed +       (Tier 1 stats +  │
    batch sweep)       Tier 2 ML)      │
                                       ▼
                            CompositeSignalEngine
                                       │
                                       ▼
                                 FastAPI / CLI
```

1. **Ingestion**: `edgartools` downloads filings; `FilingProcessor` splits them by section (Item 1A, Item 7, etc.) using regex-based header detection.
2. **Storage**: Qdrant stores dense vectors embedded via `sentence-transformers` (`all-MiniLM-L6-v2`).
3. **Retrieval**: The `Retriever` encodes queries and performs cosine-similarity search against indexed filing chunks.
4. **Agents**:
   - **Analyst**: Synthesizes retrieved chunks into a scored signal (risk/sentiment drift).
   - **Critic**: Reviews the Analyst's output for hallucinations and weak evidence.
5. **Insider Analysis**: `InsiderAnalyzer` runs statistical anomaly rules and an Isolation Forest model over Form 4 transaction histories.
6. **Composite Engine**: Merges filing and insider signals into a unified score and LLM-generated recommendation.
7. **Serving**: FastAPI exposes the analysis engine; a Typer CLI provides ingestion and search commands.

## LLM(s) Used and Why

### Claude Sonnet 4 (Anthropic) -- Reasoning

All generative reasoning tasks use **Claude Sonnet 4** (`claude-sonnet-4-20250514`), accessed through the Anthropic Python SDK via `src/services/llm_client.py`. The model is configurable by setting `LLM_MODEL` in your `.env` file.

Why Claude Sonnet:

- **Structured-output compliance**: The pipeline relies on the LLM returning valid JSON matching specific schemas. Claude Sonnet demonstrates strong adherence to JSON output instructions, reducing parse failures.
- **Long context window**: SEC filing excerpts can be lengthy. Claude's context window accommodates multiple filing sections in a single prompt without truncation.
- **Low hallucination rate**: The Critic agent's entire purpose is catching fabricated quotes and unsupported claims. Using a model with lower baseline hallucination rates makes the Critic's job easier and the overall pipeline more reliable.
- **Temperature control**: The Analyst runs at temperature 0.1 (creative but grounded), while the Critic runs at 0.0 (maximum determinism for verification). The composite recommendation uses 0.1.

### all-MiniLM-L6-v2 (sentence-transformers) -- Embeddings

Dense vector embeddings for the retrieval layer use **`all-MiniLM-L6-v2`**, loaded locally via the `sentence-transformers` library in `src/services/retriever.py`.

Why this model:

- **Lightweight and fast**: Runs on CPU without GPU requirements, keeping the deployment footprint small.
- **Zero API cost**: Embeddings are computed locally, avoiding per-token charges on every ingestion and query.
- **Good general-purpose quality**: Produces 384-dimensional vectors that perform well on semantic similarity benchmarks for the retrieval use case.

## Prompting Strategy

The pipeline uses three distinct LLM prompts, each with a specific role and output contract.

### Analyst Prompt

The Analyst (`src/agents/analyst.py`) is instructed to role-play as a **"Senior Quantitative Analyst at a top-tier hedge fund"** with explicit focus areas:

1. Material changes in Risk Factors (Item 1A)
2. Subtle shifts in Management sentiment (MD&A)
3. Specific legal or regulatory threats

The prompt requests structured JSON output matching a defined schema: `signal_score` (0.0--1.0), `confidence` (0.0--1.0), `summary`, `risk_factors`, and `key_quotes`. Filing chunks are formatted with section name and filing date metadata to give the model temporal and structural context. Temperature is set to **0.1** to allow nuanced analysis while keeping output grounded.

### Critic Prompt

The Critic (`src/agents/critic.py`) is instructed to role-play as a **"Compliance Officer and Risk Manager"** with three hard verification rules:

1. If the Analyst cites a quote that does not exist in the source text, **reject**.
2. If the signal score is high (>0.7) but evidence is weak, **reject**.
3. If the analysis is sound, **approve**.

Output is a JSON object with `approved` (boolean) and `critique` (string explanation). Temperature is set to **0.0** for maximum determinism -- the Critic should be as consistent and literal as possible.

### Composite Recommendation Prompt

The `CompositeSignalEngine` (`src/agents/composite_signal.py`) issues a third LLM call to generate a natural-language recommendation that merges both data streams. The prompt asks for five specific elements: what insiders did, what filings say, suggested position, confidence and time horizon, and key risk caveats. Output is plain text (3--5 sentences) at temperature 0.1.

## Evaluation and Iteration

### Analyst-Critic Loop

The `AgentWorkflow` (`src/agents/workflow.py`) orchestrates a sequential Analyst → Critic pipeline:

1. The **Analyst** receives retrieved filing chunks and produces a scored `AlphaSignal`.
2. The **Critic** receives the Analyst's signal *and* the original source chunks, then verifies claims against the text.
3. If the Critic **rejects** the signal, the confidence score is halved (multiplied by 0.5) and the critique notes are attached to the output for transparency.

The workflow accepts a `max_retries` parameter, providing the architectural hook for a full iterative loop where rejected signals would be re-analyzed incorporating the Critic's feedback. The current implementation runs a single pass.

### Insider Anomaly Detection (Two-Tier)

The `InsiderAnalyzer` (`src/agents/insider_analyzer.py`) applies a layered evaluation strategy:

**Tier 1 -- Statistical Rules:**
- **Volume anomaly**: Z-score of the latest transaction size vs. historical mean (threshold: z > 2.0).
- **Frequency anomaly**: Ratio of days since last trade vs. average trading frequency (threshold: ratio < 0.25).
- **Holdings percentage**: Fraction of total holdings sold in a single transaction (threshold: > 20%).
- **Cluster selling**: Number of distinct insiders selling within a configurable window (threshold: >= 3 sellers in 14 days).

**Tier 2 -- Isolation Forest ML:**
- Builds a feature matrix (transaction size, inter-trade days, percent sold, C-suite flag) from historical data.
- Trains an Isolation Forest (100 estimators, 10% contamination) and scores the latest transaction.
- Requires a minimum of 10 historical transactions to activate.

**Composite Scoring:**
- Blends Tier 1 max severity (60% weight) with Tier 2 ML score (40% weight), plus a co-occurrence boost when multiple anomaly types fire simultaneously.
- Applies role weighting (CEO/CFO transactions score 1.5x) and a discount for pre-planned 10b5-1 trades.

### Filing Monitor

The `FilingMonitor` (`src/services/filing_monitor.py`) uses a dual-path ingestion strategy:
- **Path A**: Polls the SEC EDGAR ATOM feed at adaptive intervals (5 min during market hours, 30 min off-hours) for near-real-time Form 4 detection.
- **Path B**: A scheduled batch sweep over the full ticker universe as a safety net, with configurable overlap windows to avoid gaps.

## Tradeoffs and Limitations

| Area | Current State | Implication |
|------|--------------|-------------|
| **Analyst-Critic loop** | Single pass; Critic rejection halves confidence but does not re-prompt the Analyst | A full iterative loop would improve signal quality at the cost of added latency and API spend |
| **LLM output parsing** | Strips markdown fences and calls `json.loads` | No Pydantic output parser or Anthropic tool-use/structured-output mode; malformed JSON from the LLM falls back to a zero-score signal |
| **Retrieval strategy** | Dense-only cosine similarity via Qdrant | The project blueprint calls for hybrid BM25 + dense with Reciprocal Rank Fusion (RRF), which is not yet implemented; keyword-heavy queries may underperform |
| **Section splitting** | Regex-based header detection in `FilingProcessor` | SEC filings have notoriously inconsistent HTML formatting; the parser may miss or misalign sections in edge cases |
| **Cross-period drift** | Analyzes individual filings in isolation | Does not yet compare language across filing periods to detect semantic drift over time -- the core "alpha" thesis |
| **ATOM feed processing** | `FilingMonitor._process_accession` is a stub | Near-real-time Form 4 ingestion relies entirely on the batch sweep path for now |
| **Embedding model** | `all-MiniLM-L6-v2` (384-dim) | Trades retrieval accuracy for speed and simplicity; a larger model (e.g., `e5-large`) would improve recall at higher compute cost |
| **No reranking** | Retrieved chunks are passed directly to the Analyst | A cross-encoder reranking step (as described in the blueprint) would improve the relevance of chunks fed to the LLM |

## Setup

### 1. Install Dependencies

```bash
pip install -e .
```

### 2. Environment Variables

Create a `.env` file:

```bash
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-sonnet-4-20250514
SEC_USER_AGENT="Your Name your@email.com"
QDRANT_HOST=localhost
QDRANT_PORT=6333
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=sec_alpha
POSTGRES_USER=sec_alpha
POSTGRES_PASSWORD=your_password
```

`LLM_MODEL` defaults to `claude-sonnet-4-20250514` if omitted. Set it to any Anthropic model ID to swap models without changing code. See `.env.example` for a template.

### 3. Run Infrastructure (Docker Compose)

```bash
docker compose up -d
```

This starts Qdrant (vector DB) and TimescaleDB (insider transaction storage).

## How It's Intended to Be Used

ETeam-Signal has **no GUI**. It exposes two interfaces:

- **CLI** (`src/cli.py`) -- for data ingestion, ad-hoc searches, and one-off analysis runs from the terminal. Best for setup, debugging, and manual exploration.
- **REST API** (`src/main.py`) -- for programmatic access. Designed to be called by dashboards, trading systems, scheduled jobs, or any HTTP client. Returns structured JSON suitable for downstream consumption.

A typical workflow:

1. **Ingest** filings for your universe of tickers via the CLI (one-time or scheduled).
2. **Query** the REST API from your application to get on-demand Alpha Signal scores.
3. **Monitor** insider trading continuously via the CLI's `insider monitor` command or schedule `insider scan` as a cron job.

## Usage

### CLI Commands

**Ingest a filing into the vector store:**

```bash
python src/cli.py ingest AAPL --limit 1
```

**Search indexed filings:**

```bash
python src/cli.py search "risk factors china"
```

**Insider trading commands:**

```bash
python src/cli.py insider universe-refresh       # fetch S&P 500 ticker list
python src/cli.py insider ingest --days-back 90  # ingest Form 4 data
python src/cli.py insider analyze --ticker AAPL  # analyze one ticker
python src/cli.py insider scan                   # full pipeline: ingest + analyze + alert
python src/cli.py insider monitor                # continuous ATOM feed + batch polling
python src/cli.py insider alerts                 # view active alerts
```

### REST API

Start the server:

```bash
uvicorn src.main:app --reload
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/analyze/{ticker}` | Full filing analysis pipeline (retrieve → analyze → critique) |
| `POST` | `/api/v1/insider/ingest` | Trigger Form 4 ingestion |
| `GET` | `/api/v1/insider/signal/{ticker}` | Composite alpha signal (filing + insider) |
| `GET` | `/api/v1/insider/anomalies/{ticker}` | Anomalies for a ticker |
| `GET` | `/api/v1/insider/anomalies` | All anomalies (filterable by score) |
| `GET` | `/api/v1/insider/profile/{ticker}/{insider_name}` | Insider trading profile |
| `GET` | `/api/v1/insider/alerts` | Active alerts |
| `GET` | `/health` | Health check |

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/analyze/AAPL
```

Returns a JSON `AlphaSignal` with `signal_score`, `confidence`, `summary`, `risk_factors`, `key_quotes`, and `critic_notes`.

## Project Structure

```
src/
├── api/              # FastAPI route handlers
│   ├── routes.py     # Filing analysis endpoints
│   └── insider_routes.py  # Insider trading endpoints
├── agents/           # Reasoning logic
│   ├── analyst.py    # Analyst agent (signal generation)
│   ├── critic.py     # Critic agent (hallucination check)
│   ├── workflow.py   # Analyst→Critic orchestration
│   ├── composite_signal.py  # Filing + insider signal merger
│   └── insider_analyzer.py  # Two-tier anomaly detection
├── core/             # Configuration & logging
│   ├── config.py     # Pydantic settings (env vars)
│   └── logger.py     # Structured logging (structlog)
├── models/           # Pydantic schemas
│   ├── schema.py     # FilingChunk, RetrievalResult, AlphaSignal
│   └── insider_schema.py  # InsiderTransaction, InsiderSignal, etc.
├── services/         # External integrations
│   ├── llm_client.py      # Anthropic Claude wrapper
│   ├── sec_client.py      # SEC EDGAR downloader
│   ├── processor.py       # Section-aware filing splitter
│   ├── retriever.py       # Embedding + vector search
│   ├── vector_store.py    # Qdrant client
│   ├── insider_client.py  # Form 4 data fetcher
│   ├── insider_store.py   # TimescaleDB persistence
│   ├── filing_monitor.py  # ATOM feed + batch sweep
│   ├── alert_service.py   # Actionable alert generation
│   └── universe.py        # Ticker universe management
├── utils/
│   └── reporter.py   # Report generation utilities
├── cli.py            # Typer CLI (ingest, search)
└── main.py           # FastAPI application factory
tests/                # pytest suite mirroring src/ structure
```

## Tech Stack

- **Language**: Python 3.10+
- **LLM**: Anthropic Claude Sonnet 4
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector DB**: Qdrant
- **Time-series DB**: TimescaleDB (PostgreSQL)
- **API Framework**: FastAPI + Uvicorn
- **CLI**: Typer
- **ML**: scikit-learn (Isolation Forest), scipy, numpy
- **Observability**: structlog (JSON structured logging)
- **Validation**: Pydantic v2
