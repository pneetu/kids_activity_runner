import os
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
QDRANT_URL = os.getenv("QDRANT_URL")  #http://localhost:6333
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
class QdrantVectorStore:
    def __init__(self, collection: str, dim: int, recreate: bool = False):
        self.collection = collection
        self.dim = dim
        # Prefer QDRANT_URL if provided
        if QDRANT_URL:
            self.client = QdrantClient(url=QDRANT_URL)
        else:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        if recreate:
            existing = [c.name for c in self.client.get_collections().collections]
            if self.collection in existing:
                self.client.delete_collection(self.collection)
        self._ensure_collection()
    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
    # Drop-in alias for your pipeline's store.add(...)
    def add(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if metadatas is None:
            metadatas = [{} for _ in texts]
        payloads = []
        for text, meta in zip(texts, metadatas):
            payload = {"text": text}
            if meta:
                payload.update(meta)
            payloads.append(payload)
        self.upsert(embeddings=embeddings, payloads=payloads)
    def upsert(self, embeddings: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        points = []
        for emb, payload in zip(embeddings, payloads):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        resp = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        hits = resp.points

        results: List[Dict[str, Any]] = []
        for h in hits:
            payload = h.payload or {}
            results.append(
                {
                    "text": payload.get("text", ""),
                    "score": float(h.score),
                    "metadata": payload,
                }
            )
        return results