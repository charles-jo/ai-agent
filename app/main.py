from fastapi import FastAPI, HTTPException
from langfuse.langchain import CallbackHandler

from .chain import rag_chain, rewrite_chain, to_lc_messages
from .config import settings
from .models import QueryRequest, QueryResponse, SearchResult
from .retriever import HybridQdrantRetriever

app = FastAPI(title="IaC AI Agent", version="1.0.0")


def _callbacks() -> list:
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return []
    return [CallbackHandler(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        trace_name="iac-query",
    )]


def _format_context(docs) -> str:
    return "\n\n---\n\n".join(
        f"[{i}] {doc.metadata['file_path']}\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        config = {"callbacks": _callbacks()}
        history = to_lc_messages(request.history)
        retriever = HybridQdrantRetriever(top_k=request.top_k)

        search_query = request.query
        if history:
            search_query = await rewrite_chain.ainvoke(
                {"query": request.query, "history": history}, config=config
            )

        docs = await retriever.ainvoke(search_query, config=config)
        if not docs:
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        answer = await rag_chain.ainvoke(
            {"query": request.query, "context": _format_context(docs), "history": history},
            config=config,
        )

        return QueryResponse(
            answer=answer,
            sources=[
                SearchResult(
                    file_path=doc.metadata["file_path"],
                    chunk_index=doc.metadata["chunk_index"],
                    text=doc.page_content,
                    score=doc.metadata["score"],
                )
                for doc in docs
            ],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
