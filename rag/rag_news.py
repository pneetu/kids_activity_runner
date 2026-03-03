from typing import List, Dict, Any
import os
import re
from rank_bm25 import BM25Okapi
from openai import OpenAI

from rag.chunking import chunk_text
from embeddings.embedder import embed_texts
from rag.qdrant_store import QdrantVectorStore

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

COLLECTION = os.getenv("QDRANT_COLLECTION", "ai_news")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# text-embedding-3-small is 1536 dims
EMBED_DIM = 1536


def index_news_items(news_items: List[Dict[str, str]], recreate: bool = True) -> int:
    """
    news_items: [{"title":..., "url":..., "published":..., "summary":...}, ...]
    Index chunks of title + (optional) summary.
    Returns number of chunks indexed.
    """
    store = QdrantVectorStore(collection=COLLECTION, dim=EMBED_DIM, recreate=recreate)

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for item in news_items:
        title = item.get("title", "")
        url = item.get("url", "")
        published = item.get("published", "")
        summary = item.get("summary", "") or item.get("description", "")

        base_text = (
            f"TITLE: {title}\n"
            f"DATE: {published}\n"
            f"URL: {url}\n"
            f"SUMMARY: {summary}\n"
        ).strip()

        chunks = chunk_text(base_text, chunk_size=800, overlap=100)
        for ch in chunks:
            texts.append(ch)
            metadatas.append(
                {
                    "title": title,
                    "url": url,
                    "published": published,
                }
            )

    if not texts:
        return 0

    embs = embed_texts(texts, model=EMBED_MODEL)

    # store.add will wrap metadatas into payloads + include "text"
    store.add(embeddings=embs, texts=texts, metadatas=metadatas)

    return len(texts)


def rag_answer(question: str, top_k: int = 5) -> Dict[str, Any]:
    store = QdrantVectorStore(collection=COLLECTION, dim=EMBED_DIM, recreate=False)

    # 1) Dense retrieve more candidates than we need
    initial_k = max(15, top_k * 4)
    q_emb = embed_texts([question], model=EMBED_MODEL)[0]
    hits = store.search(q_emb, top_k=initial_k)  # [{"text","score","metadata"}, ...]

    # 2) BM25 rerank (hybrid)
    def tokenize(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", (s or "").lower())

    docs = [h.get("text", "") for h in hits]
    tokenized_docs = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)

    q_tokens = tokenize(question)
    bm25_scores = bm25.get_scores(q_tokens)

    # Combine scores: keep dense relevance, boost lexical matches
    combined = []
    for h, bm in zip(hits, bm25_scores):
        dense = float(h.get("score", 0.0))
        combined.append((0.7 * dense + 0.3 * float(bm), h))

    combined.sort(key=lambda x: x[0], reverse=True)
    top_hits = [h for _, h in combined[:top_k]]

    # 3) Build context + sources from reranked hits
    context_parts = []
    sources = []
    for h in top_hits:
        meta = h.get("metadata", {}) or {}
        url = meta.get("url", "")
        title = meta.get("title", "")
        published = meta.get("published", "")

        context_parts.append(f"{h.get('text','')}\nSOURCE: {url}".strip())
        if url:
            sources.append({"url": url, "title": title, "published": published})

    context = "\n\n---\n\n".join(context_parts).strip()

    prompt = f"""Answer the question using ONLY the context.
If the context is insufficient, say you don't have enough info.

QUESTION:
{question}

CONTEXT:
{context}
""".strip()

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You answer questions grounded in provided context. When possible, cite sources by URL."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        timeout=30,
    )

    answer = (resp.choices[0].message.content or "").strip()

    # de-dup sources by url
    dedup = {}
    for s in sources:
        if s["url"] and s["url"] not in dedup:
            dedup[s["url"]] = s

    return {"answer": answer, "sources": list(dedup.values())}