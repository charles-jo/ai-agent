from openai import AsyncOpenAI

from .config import settings
from .models import Message, SearchResult

_client = AsyncOpenAI(base_url=settings.vllm_base_url, api_key=settings.vllm_api_key)

_SYSTEM_PROMPT = (
    "You are an expert assistant for Infrastructure-as-Code (IaC). "
    "Answer the user's question using only the provided context. "
    "Be concise and accurate. If the context does not contain enough information, say so."
)

_REWRITE_PROMPT = (
    "You are a search query optimizer. "
    "Given a conversation history and a follow-up question, rewrite the question as a "
    "standalone, specific search query suitable for a vector database. "
    "Output only the rewritten query, nothing else."
)


async def rewrite_query(query: str, history: list[Message]) -> str:
    if not history:
        return query
    turns = "\n".join(f"{m.role}: {m.content}" for m in history)
    resp = await _client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _REWRITE_PROMPT},
            {"role": "user", "content": f"Conversation:\n{turns}\n\nFollow-up: {query}"},
        ],
        max_tokens=64,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def _build_context(results: list[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.file_path}\n{r.text}")
    return "\n\n---\n\n".join(parts)


async def generate_response(
    query: str,
    results: list[SearchResult],
    history: list[Message],
) -> str:
    context = _build_context(results)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})

    resp = await _client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )
    return resp.choices[0].message.content.strip()
