import time
import uuid

from fastapi import FastAPI, HTTPException
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from opentelemetry import trace

from .config import settings
from .models import (
    ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, Message, QueryRequest, QueryResponse, SearchResult,
)
from .rag import answer_generator, query_rewriter, to_lc_messages
from .retriever import HybridContextRetriever

app = FastAPI(title="IaC AI Agent", version="1.0.0")
_tracer = trace.get_tracer("ai-agent")

# Langfuse v4: must initialize a client before CallbackHandler can use it.
# Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from env.
_langfuse = (
    Langfuse()
    if settings.langfuse_public_key and settings.langfuse_secret_key
    else None
)


def _callbacks() -> list:
    if not _langfuse:
        return []
    return [CallbackHandler(public_key=settings.langfuse_public_key)]


def _format_context(docs) -> str:
    return "\n\n---\n\n".join(
        f"[{i}] {doc.metadata['file_path']}\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )


async def _execute_pipeline(query: str, history: list[Message], top_k: int = settings.search_top_k):
    config = {"callbacks": _callbacks()}
    lc_history = to_lc_messages(history)
    retriever = HybridContextRetriever(top_k=top_k)

    search_query = query
    if lc_history:
        search_query = await query_rewriter.ainvoke(
            {"query": query, "history": lc_history}, config=config
        )

    with _tracer.start_as_current_span("HybridContextRetriever"):
        docs = await retriever.retrieve(search_query)

    answer = await answer_generator.ainvoke(
        {"query": query, "context": _format_context(docs), "history": lc_history},
        config=config,
    ) if docs else None

    return answer, docs


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        answer, docs = await _execute_pipeline(request.query, request.history, request.top_k)
        if not docs:
            raise HTTPException(status_code=404, detail="No relevant documents found.")
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


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "ai-agent-iac", "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    try:
        *prior, last = request.messages
        if last.role != "user":
            raise HTTPException(status_code=400, detail="Last message must be from a user.")

        history = [
            Message(role=m.role, content=m.content)
            for m in prior
            if m.role in ("user", "assistant")
        ]

        answer, docs = await _execute_pipeline(last.content, history)
        if not docs:
            answer = "I couldn't find relevant information in the knowledge base."

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model or "ai-agent-iac",
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=answer),
                )
            ],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
