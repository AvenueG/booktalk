"""向量库 + 嵌入 + 重排（本地，Qdrant path 模式，单机单用户）。"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder, SentenceTransformer

from . import config


@dataclass
class Chunk:
    """书的一个片段。heading 用于引用出处（章节级）。"""

    text: str
    source: str
    heading: str
    page: int | None = None

    @property
    def chunk_id(self) -> str:
        raw = f"{self.source}|{self.heading}|{self.text[:200]}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


class BookStore:
    """把书块写进 Qdrant，按问题检索并重排。"""

    def __init__(self) -> None:
        self.client = QdrantClient(path=str(config.INDEX_DIR))
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        self.reranker = CrossEncoder(config.RERANK_MODEL)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if config.COLLECTION not in existing:
            self.client.create_collection(
                collection_name=config.COLLECTION,
                vectors_config=models.VectorParams(
                    size=1024, distance=models.Distance.COSINE
                ),
            )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        )
        points = [
            models.PointStruct(
                id=c.chunk_id,
                vector=emb.tolist(),
                payload={
                    "text": c.text,
                    "source": c.source,
                    "heading": c.heading,
                    "page": c.page,
                },
            )
            for c, emb in zip(chunks, embeddings)
        ]
        self.client.upsert(collection_name=config.COLLECTION, points=points)
        return len(points)

    def clear(self) -> None:
        """清空整本书的索引（换书时用）。"""
        self.client.delete_collection(collection_name=config.COLLECTION)
        self._ensure_collection()

    def search(self, question: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or config.TOP_K_FINAL
        q = self.embedder.encode(question, normalize_embeddings=True)
        hits = self.client.query_points(
            collection_name=config.COLLECTION,
            query=q.tolist(),
            limit=config.TOP_K_RETRIEVE,
            with_payload=True,
        ).points
        if not hits:
            return []
        pairs = [(question, h.payload["text"]) for h in hits]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(hits, scores), key=lambda x: -float(x[1]))
        return [
            {**hit.payload, "score": round(float(score), 4)}
            for hit, score in ranked[:top_k]
        ]
