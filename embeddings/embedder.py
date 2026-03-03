import os
import certifi
import httpx
from typing import List
from openai import OpenAI

_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set (env not loaded).")

        http_client = httpx.Client(verify=certifi.where(), timeout=30.0)
        _client = OpenAI(api_key=key, http_client=http_client)
    return _client

def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[list[float]]:
    resp = _get_client().embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]