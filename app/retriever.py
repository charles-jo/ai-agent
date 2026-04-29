import asyncio
import hashlib
import re
from collections import Counter

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import SparseVector

from .config import settings

SPARSE_BUCKETS = 2**20  # must match the indexing pipeline

_qdrant = AsyncQdrantClient(url=settings.qdrant_url)
_embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    base_url=settings.vllm_base_url,
    api_key=settings.vllm_api_key,
    check_embedding_ctx_length=False,
)


def _bm25_sparse(text: str) -> tuple[list[int], list[float]]:
    tf = Counter(re.findall(r"[a-z0-9_\-\.]+", text.lower()))
    indices, values, seen = [], [], set()
    for token, freq in tf.items():
        idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % SPARSE_BUCKETS
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
            values.append(float(freq))
    return indices, values


def _rrf_merge(dense_hits: list, sparse_hits: list, top_k: int, k: int = 60) -> list[tuple[str, float, dict]]:
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    for rank, hit in enumerate(dense_hits):
        pid = str(hit.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        payloads[pid] = hit.payload or {}
    for rank, hit in enumerate(sparse_hits):
        pid = str(hit.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        payloads.setdefault(pid, hit.payload or {})
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(pid, score, payloads[pid]) for pid, score in ranked[:top_k]]


class HybridQdrantRetriever(BaseRetriever):
    top_k: int = settings.search_top_k

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        raise NotImplementedError("Use ainvoke for async retrieval")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        dense_vec = await _embeddings.aembed_query(query)
        sparse_indices, sparse_values = _bm25_sparse(query)

        dense_result, sparse_result = await asyncio.gather(
            _qdrant.query_points(
                collection_name=settings.dense_collection,
                query=dense_vec,
                limit=self.top_k * 2,
                with_payload=True,
            ),
            _qdrant.query_points(
                collection_name=settings.sparse_collection,
                query=SparseVector(indices=sparse_indices, values=sparse_values),
                limit=self.top_k * 2,
                with_payload=True,
                using="sparse",
            ),
        )

        return [
            Document(
                page_content=payload.get("text", ""),
                metadata={
                    "file_path": payload.get("file_path", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "score": round(score, 4),
                },
            )
            for _, score, payload in _rrf_merge(dense_result.points, sparse_result.points, self.top_k)
        ]
