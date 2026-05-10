# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IaC AI Agent is a FastAPI-based Retrieval-Augmented Generation (RAG) system that answers questions about Infrastructure-as-Code. It combines hybrid semantic search (dense + sparse vectors) with LLM-powered answer generation, integrated with observability via OpenTelemetry and Langfuse.

## Core Architecture

The system has three main components:

**1. Hybrid Retriever (`app/retriever.py`)**
- `HybridContextRetriever`: Executes dual-path search on Qdrant vector database
  - **Dense search**: Uses BGE-M3 embeddings via vLLM to find semantically similar documents
  - **Sparse search**: Computes token-based sparse vectors (MD5 hashing into 2^20 buckets) for keyword matching
  - **Ranking**: Applies Reciprocal Rank Fusion (RRF) to merge dense and sparse results into a unified top-k list
  - Retrieves from two Qdrant collections: `iac_code_dense` and `iac_code_sparse`

**2. RAG Pipeline (`app/rag.py`)**
- **query_rewriter**: LangChain chain that rewrites follow-up questions into standalone search queries using conversation history
- **answer_generator**: LangChain chain that generates answers using retrieved context and conversation history
- Both use ChatOpenAI (pointing to vLLM) with configurable parameters (temperature=0.1, max_tokens=1024)
- Supports multi-turn conversation via `to_lc_messages()` helper that converts Message objects to LangChain HumanMessage/AIMessage

**3. API Layer (`app/main.py`)**
- **FastAPI app** with two main endpoints:
  - `/query` (POST): Custom query interface accepting QueryRequest with query, top_k (1-20), and optional conversation history
  - `/v1/chat/completions` (POST): OpenAI-compatible endpoint for Open WebUI integration
  - `/health` (GET): Health check endpoint
  - `/v1/models` (GET): Lists available models (returns "ai-agent-iac")
- Internal `_execute_pipeline()` orchestrates the retriever → rewriter → generator flow
- Integrates Langfuse v4 for tracing (requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables)
- Uses OpenTelemetry for explicit span management (tracer instantiated at module load)

## Configuration

Settings are loaded from `app/config.py` using Pydantic BaseSettings (reads from `.env` file):

**Required for local/remote services:**
- `qdrant_url`: Qdrant vector database endpoint (default: Kubernetes cluster DNS)
- `dense_collection`, `sparse_collection`: Collection names in Qdrant
- `vllm_base_url`, `vllm_api_key`: vLLM router for embeddings and LLM inference
- `embedding_model`: BAAI/bge-m3 (BGE multilingual)
- `llm_model`: Qwen/Qwen2.5-3B-Instruct

**Tunable parameters:**
- `search_top_k`: Default number of documents to retrieve (default: 5, endpoint can override with 1-20)
- `llm_max_tokens`: Max generation length (default: 1024)
- `llm_temperature`: LLM randomness (default: 0.1 — low temperature for consistency)

**Observability (optional):**
- `langfuse_public_key`, `langfuse_secret_key`: For tracing (empty string disables Langfuse)
- `langfuse_host`: Langfuse backend URL

## Build & Deployment

**Docker:**
```bash
docker build -t iac-agent:latest .
docker run -p 8000:8000 -e VLLM_BASE_URL=... -e QDRANT_URL=... iac-agent:latest
```

**Local Development:**
```bash
pip install -r requirements.txt
# Set environment variables or create .env file
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Test the server:**
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "How do I configure X?"}'
curl http://localhost:8000/v1/models
```

## Key Design Patterns

1. **Async-first**: All I/O operations use `async`/`await` (asyncio, httpx, Qdrant AsyncQdrantClient)
2. **LangChain chains**: Composable prompt → LLM → parser pipelines with built-in observability (`run_name` config)
3. **Span nesting**: OpenTelemetry spans explicitly nest within each other (e.g., `embed_question` and search spans nest under `HybridContextRetriever`)
4. **Payload-driven metadata**: Vector search results carry metadata (file_path, chunk_index, score) in Qdrant payloads
5. **Conversation awareness**: History is passed through the pipeline to both query rewriter and answer generator

## Important Implementation Details

- **Sparse vector hashing**: Uses MD5(token).hexdigest() % 2^20 to map tokens to indices. **SPARSE_BUCKETS constant must match the indexing pipeline.**
- **RRF formula**: `score[docid] += 1 / (k + rank + 1)` for each search path (k=60 by default). Prevents sparse search from dominating dense search.
- **Langfuse v4**: Must initialize `Langfuse()` client at module load before passing CallbackHandler to chains. Client creation is skipped if secrets are empty.
- **OpenAI compatibility**: The `/v1/chat/completions` endpoint wraps the RAG pipeline to match OpenAI API shape. Last message must be from "user" role.
- **Error handling**: Missing relevant documents returns 404 on `/query` but gracefully falls back to "couldn't find information" response on `/v1/chat/completions`.
- **Embedding context length**: `check_embedding_ctx_length=False` on OpenAIEmbeddings to allow long documents without truncation warnings.

## Dependencies

- **FastAPI 0.115.0**: Web framework
- **uvicorn[standard] 0.30.6**: ASGI server
- **LangChain** (langchain, langchain-core, langchain-openai): RAG orchestration and prompts
- **Qdrant (qdrant-client 1.12.0)**: Vector database client with async support
- **Langfuse 3.0+**: Tracing and observability backend
- **OpenTelemetry (trace)**: Structured span management
- **Pydantic 2.5.2**: Settings and data validation
- **httpx < 0.28.0**: Pinned for OpenAI SDK proxy compatibility

## Common Commands

There is no test suite, Makefile, or linting configuration. The codebase is designed for containerized deployment with external service dependencies (vLLM, Qdrant, optional Langfuse).

To run locally:
```bash
# Install dependencies
pip install -r requirements.txt

# Start server (requires .env or env vars: VLLM_BASE_URL, VLLM_API_KEY, QDRANT_URL)
uvicorn app.main:app --reload
```

## Recent Work

The codebase has gone through several observability refinements:
- Migrated from direct Langfuse integration to LangChain's CallbackHandler for cleaner tracing
- Added explicit OpenTelemetry span nesting to control trace hierarchy (critical for readability in Langfuse)
- Fixed Langfuse v4 API (client must be initialized before CallbackHandler use)
- Added OpenAI-compatible `/v1/chat/completions` endpoint for Open WebUI integration
- Implemented query rewriting for multi-turn conversation support
