## Project Status
**Current Phase**: Feature Complete (v0.1.0)
**Build Status**: Passing
**Environment**: `conda activate sec_alpha_env` (Python 3.10)
## A Production-Grade Hybrid RAG & Agentic Pipeline for Financial Alpha Extraction.

# 1. Project Goals & Spirit
Goal: Build a systematic pipeline to ingest SEC 10-K/10-Q filings, detect semantic "drifts" in risk/sentiment between periods, and output an "Alpha Signal" score.

Spirit: The code must reflect Principal Engineer standards: scalability, observability, and defensive programming. It is not a script; it is a system.

Everlaw Alignment: Showcase "needle in a haystack" retrieval, hybrid search (keyword + semantic), and agentic self-correction (supervision).

# 2. Standards (Principal Hygiene)
Strict Typing: Use Python Type Hints and Pydantic models for all data structures (especially LLM outputs).

Observability: Implement structured logging (JSON format) and "Trace IDs" to follow a query through retrieval and generation stages.

Asynchrony: Use asyncio for I/O bound tasks (API calls to LLMs and Vector DBs) to demonstrate high-throughput capability.

Modularity: Separate the Ingestion Engine, the Retriever, and the Agentic Reasoner into distinct modules with clear interfaces.

Test-Driven Development (TDD): Adhere to a strict "Red-Green-Refactor" cycle.
    - Write the test case first (defining the expected interface and behavior).
    - Run the test (ensure it fails).
    - Write the minimal code to pass the test.
    - Refactor for cleanliness and performance.
    - This ensures all code is testable by design and prevents "testing as an afterthought."
    - Verification of Source: Tests must import and execute the actual code from the `src/` directory. Do not redefine logic or classes within the test file itself to "fake" functionality; test the real implementation.  

# 3. Technical Requirements
Source Data: SEC 10-K/10-Q filings (via edgar-python or Hugging Face financial-reports-sec).

Vector DB: Pinecone, Qdrant, or Weaviate (Local or Cloud) to store document embeddings.

Hybrid Search: Implementation of BM25 (keyword) and Cosine Similarity (vector) fused via Reciprocal Rank Fusion (RRF).

LLM Stack: GPT-4o or Claude 3.5 Sonnet for reasoning; FastAPI for the service layer.

# 4. Architectural Strategy (The "Agent's Instruction")
Phase 1: Context-Aware Ingestion
Instruction: Don't just chunk by character count. Implement a "Section-Aware" splitter that identifies headers (e.g., "Item 1A: Risk Factors") to maintain metadata context.

Metadata: Store ticker, report_type, period_end_date, and section_name with every vector.

## Phase 2: Hybrid Retrieval Engine
Instruction: Implement a dual-retrieval path. Path A is semantic (Dense Vectors); Path B is lexical (BM25).

Reranking: After retrieval, use a "Cross-Encoder" or a lightweight LLM call to rerank the top 10 results for relevance to the "Alpha" query.

## Phase 3: Agentic Reasoning Loop (The "Critic")
Instruction: Build a "Self-Correction" agent.

Process: Agent 1 (Analyst) proposes an Alpha Signal. Agent 2 (Critic) attempts to find contradictions in the retrieved text. If a contradiction is found, the Analyst must refine the score.

# 5. Implementation Roadmap for the Coding Agent
Skeleton: Create the project structure with pyproject.toml, src/, and tests/.

Schema: Define Pydantic models for FilingChunk, RetrievalResult, and AlphaSignal.

Ingestion: Build the SEC data downloader and the Section-Aware splitter.

Retrieval: Set up the Vector DB and the Hybrid Search function.

Agent Logic: Implement the Analyst/Critic loop using LangGraph or a simple custom state machine.

CLI/API: Build a main.py that takes a Ticker and outputs a PDF "Alpha Report."