from openai import AsyncOpenAI

from .config import settings
from .models import SearchResult

_client = AsyncOpenAI(base_url=settings.vllm_base_url, api_key=settings.vllm_api_key)

_SYSTEM_PROMPT = (
    "You are an expert assistant for Infrastructure-as-Code (IaC). "
    "Answer the user's question using only the provided context. "
    "Be concise and accurate. If the context does not contain enough information, say so."
)


def _build_context(results: list[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.file_path}\n{r.text}")
    return "\n\n---\n\n".join(parts)


async def generate_response(query: str, results: list[SearchResult]) -> str:
    context = _build_context(results)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    resp = await _client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )
    return resp.choices[0].message.content.strip()
