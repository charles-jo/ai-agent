import hashlib
import re
from collections import Counter

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import SparseVector

from .config import settings
from .models import SearchResult

_qdrant = AsyncQdrantClient(url=settings.qdrant_url)
_embedder = AsyncOpenAI(base_url=settings.vllm_base_url, api_key=settings.vllm_api_key)

SPARSE_INDEX_SIZE = 2**20  # 1M buckets — must match indexing pipeline


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_\-\.]+", text.lower())


def _sparse_vector(text: str) -> tuple[list[int], list[float]]:
    """BM25-style sparse encoding using consistent token hashing."""
    tf = Counter(_tokenize(text))
    indices, values = [], []
    seen: set[int] = set()
    for token, freq in tf.items():
        idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % SPARSE_INDEX_SIZE
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
            values.append(float(freq))
    return indices, values


async def _embed(text: str) -> list[float]:
    resp = await _embedder.embeddings.create(
        model=settings.embedding_model,
        input=text,
        encoding_format="float",
    )
    return resp.data[0].embedding


def _rrf(
    dense_hits: list,
    sparse_hits: list,
    k: int = 60,
    top_k: int = 5,
) -> list[tuple[str, float, dict]]:
    """Reciprocal Rank Fusion — merges dense and sparse result lists."""
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
    return [(pid, score, payloads[pid]) for pid, score, _ in zip(ranked, [s for _, s in ranked], range(top_k))][:top_k]


async def hybrid_search(query: str, top_k: int) -> list[SearchResult]:
    dense_vec = await _embed(query)
    sparse_indices, sparse_values = _sparse_vector(query)

    dense_hits, sparse_hits = await _qdrant.query_points(
        collection_name=settings.dense_collection,
        query=dense_vec,
        limit=top_k * 2,
        with_payload=True,
    ), await _qdrant.query_points(
        collection_name=settings.sparse_collection,
        query=SparseVector(indices=sparse_indices, values=sparse_values),
        limit=top_k * 2,
        with_payload=True,
        using="sparse",
    )

    merged = _rrf(dense_hits.points, sparse_hits.points, top_k=top_k)

    return [
        SearchResult(
            file_path=payload.get("file_path", ""),
            chunk_index=payload.get("chunk_index", 0),
            text=payload.get("text", ""),
            score=round(score, 4),
        )
        for _, score, payload in merged
    ]
