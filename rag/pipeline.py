from dotenv import load_dotenv
load_dotenv()

from typing import List, Tuple
import os
import certifi
import httpx
from openai import OpenAI

from ingestion.news_fetcher import fetch_urls
from rag.chunking import chunk_text
from embeddings.embedder import embed_texts
from rag.qdrant_store import QdrantVectorStore


http_client = httpx.Client(verify=certifi.where(), timeout=30.0)
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), http_client=http_client)

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set")


async def rag_summarize(
    urls: List[str] | None,
    text: str | None,
    question: str,
    max_sentences: int = 5,
    top_k: int = 6,
) -> tuple[list[str], list[str]]:
    sources: List[str] = []
    all_texts: List[Tuple[str, str]] = []

    valid_urls = [u for u in (urls or []) if u.startswith("http")]

    if valid_urls:
        fetched = await fetch_urls(valid_urls)
        all_texts.extend(fetched)
        sources.extend([u for u, _ in fetched])

    if text and text.strip():
        all_texts.append(("user_text", text))
        sources.append("user_text")

    combined = "\n\n".join([t for _, t in all_texts]).strip()
    if not combined.strip():
     return (["No content provided to summarize."], sources)

    # 1) chunk
    chunks = chunk_text(combined)
    if not chunks:
        return (["No content found after processing."], sources)

    # 2) embed and index (Qdrant)
    chunk_embs = embed_texts(chunks)
    if not chunk_embs or not chunk_embs[0]:
        return (["Failed to generate embeddings."], sources)

    dim = len(chunk_embs[0])

    store = QdrantVectorStore(
        collection="ai_news_chunks",
        dim=dim,
        recreate=True,  # overwrite each run; change to False to keep history
    )

    store.add(
        embeddings=chunk_embs,
        texts=chunks,
        metadatas=[{"source": "combined"} for _ in chunks],
    )

    # 3) retrieve
    q_emb = embed_texts([question])[0]
    retrieved = store.search(q_emb, top_k=top_k)
    context = "\n\n---\n\n".join([r.get("text", "") for r in retrieved]).strip()

    if not context:
        return (["No relevant context found."], sources)

    # 4) generate key points
    system = "You are a helpful assistant that summarizes content into clear, factual bullet points."
    user = f"""
Question: {question}

Create exactly {max_sentences} key points (short bullet points).
Only use the provided context.

CONTEXT:
{context}
""".strip()

    resp = _client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        timeout=30,
    )

    raw = (resp.choices[0].message.content or "").strip()

    # simple bullet parsing
    key_points: list[str] = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-•").strip()
        if line:
            key_points.append(line)

    # Ensure exactly max_sentences (trim or pad)
    key_points = key_points[:max_sentences]
    while len(key_points) < max_sentences:
        key_points.append("")

    return (key_points, sources)
if __name__ == "__main__":
    import asyncio

    async def _test():
        points, sources = await rag_summarize(
            urls=None,
            text="OpenAI released new models. Qdrant stores embeddings. RAG retrieves relevant chunks to answer questions more accurately.",
            question="Why use Qdrant in RAG?",
            max_sentences=5,
            top_k=6,
        )
        print(points)
        print(sources)

    asyncio.run(_test())