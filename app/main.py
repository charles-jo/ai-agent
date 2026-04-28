from fastapi import FastAPI, HTTPException

from .llm import generate_response
from .models import QueryRequest, QueryResponse
from .search import hybrid_search

app = FastAPI(title="IaC AI Agent", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        results = await hybrid_search(request.query, request.top_k)
        if not results:
            raise HTTPException(status_code=404, detail="No relevant documents found.")
        answer = await generate_response(request.query, results, request.history)
        return QueryResponse(answer=answer, sources=results)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
