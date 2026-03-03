import certifi
import httpx
from typing import List, Tuple

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

async def fetch_urls(urls: List[str]) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []

    async with httpx.AsyncClient(
        verify=certifi.where(),
        headers=HEADERS,
        follow_redirects=True,
        timeout=30.0,
    ) as client:
        for url in urls:
            try:
                r = await client.get(url)
                r.raise_for_status()
                results.append((url, r.text))
            except Exception as e:
                # optional: log / skip
                # print(f"Failed to fetch {url}: {e}")
                continue

    return results