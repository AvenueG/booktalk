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

    def add_chunks(self, chunks: list[Chunk], book: str = "") -> int:
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
                    "book": book,
                },
            )
            for c, emb in zip(chunks, embeddings)
        ]
        self.client.upsert(collection_name=config.COLLECTION, points=points)
        return len(points)

    def _scroll_all(self, with_payload: bool = True) -> list:
        """把集合里的点全滚出来。本地 path 模式对高级过滤支持不稳，用这个兜底。"""
        pts: list = []
        offset = None
        while True:
            page, offset = self.client.scroll(
                collection_name=config.COLLECTION,
                limit=1000,
                offset=offset,
                with_payload=with_payload,
                with_vectors=False,
            )
            pts.extend(page)
            if offset is None or not page:
                break
        return pts

    def indexed_books(self) -> list[str]:
        """索引里出现过的 book 值（用于迁移判断/调试）。"""
        books = set()
        for p in self._scroll_all():
            b = (p.payload or {}).get("book")
            if b:
                books.add(b)
        return sorted(books)

    def tag_legacy_points(self, book: str) -> int:
        """给缺少 book 字段的旧点补上 book（历史数据迁移），返回打标数量。"""
        ids = [p.id for p in self._scroll_all() if not (p.payload or {}).get("book")]
        if ids:
            self.client.set_payload(
                collection_name=config.COLLECTION, payload={"book": book}, points=ids
            )
        return len(ids)

    def clear(self, book: str | None = None) -> None:
        """清空索引：book=None 全清（换书用），否则只删这本书的点。"""
        if book is None:
            self.client.delete_collection(collection_name=config.COLLECTION)
            self._ensure_collection()
            return
        ids = [
            p.id
            for p in self._scroll_all(with_payload=True)
            if (p.payload or {}).get("book") == book
        ]
        if ids:
            self.client.delete(
                collection_name=config.COLLECTION,
                points_selector=models.PointIdsList(points=ids),
            )

    def search(self, question: str, book: str | None = None, top_k: int | None = None) -> list[dict]:
        top_k = top_k or config.TOP_K_FINAL
        q = self.embedder.encode(question, normalize_embeddings=True)
        # 多书时多取一些再按 book 过滤，避免本地过滤把目标书挤掉
        limit = config.TOP_K_RETRIEVE if not book else max(config.TOP_K_RETRIEVE * 8, 200)
        hits = self.client.query_points(
            collection_name=config.COLLECTION,
            query=q.tolist(),
            limit=limit,
            with_payload=True,
        ).points
        if book:
            hits = [h for h in hits if (h.payload or {}).get("book") == book]
        if not hits:
            return []
        pairs = [(question, h.payload["text"]) for h in hits]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(hits, scores), key=lambda x: -float(x[1]))
        return [
            {**hit.payload, "score": round(float(score), 4)}
            for hit, score in ranked[:top_k]
        ]
