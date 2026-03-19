from dotenv import load_dotenv
load_dotenv()

import feedparser
import requests
from openai import OpenAI

FEEDS = [
    "https://news.google.com/rss/search?q=kids+activities+bay+area&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=family+events+sunnyvale&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=children%27s+museum+events+san+jose&hl=en-US&gl=US&ceid=US:en",
]

MAX_ARTICLES = 10

client = OpenAI()

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def fetch_activity_items():
    items = []

    for url in FEEDS:
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()

            feed = feedparser.parse(response.content)
            print("Fetching feed:", url)
            print("Entries found:", len(feed.entries))

            for entry in feed.entries:
                title = getattr(entry, "title", "").strip()
                link = getattr(entry, "link", "").strip()
                published = getattr(entry, "published", "").strip()

                if title and link:
                    items.append((title, link, published))
        except Exception as e:
            print(f"Failed to fetch feed {url}: {e}")
            continue

    seen = set()
    unique = []
    for t, l, p in items:
        if l in seen:
            continue
        seen.add(l)
        unique.append((t, l, p))
    if not unique:
         return [
            ("Art classes for kids in Sunnyvale", "https://example.com/art", ""),
            ("Visit Children's Discovery Museum San Jose", "https://example.com/museum", ""),
            ("Family weekend events in Bay Area", "https://example.com/events", ""),
            ("Outdoor parks and playgrounds near Sunnyvale", "https://example.com/parks", ""),
         ]
 

    return unique[:MAX_ARTICLES]


def summarize_activity_titles(titles):
    if not titles:
        return "No recent activity articles were found."

    prompt = (
        "Summarize these kids activities and family-friendly events. "
        "Prefer activities near the user's area if a location is mentioned.\n\n"
        + "\n".join([f"- {t}" for t in titles])
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize kids activities and family-friendly local events clearly. "
                    "Highlight variety, location, and why the activity may be good for children."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def get_activity_data(limit: int = 10, include_summary: bool = True):
    activities = fetch_activity_items()[:limit]

    articles = []
    for title, link, published in activities:
        articles.append(
            {
                "title": title,
                "url": link,
                "published": published,
            }
        )

    if not activities:
        return {
            "articles": [],
            "summary": "No recent activity articles were found."
        }

    titles = [t for (t, _, _) in activities]
    summary = summarize_activity_titles(titles) if include_summary else ""

    return {"articles": articles, "summary": summary}


def chat_activity_assistant(question: str) -> str:
    if not question.strip():
        return "Please enter a question."

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You help parents and families find kids activities, local events, "
                    "and family-friendly outings. Give practical, clear suggestions and "
                    "prefer nearby options when a location is mentioned."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def main():
    activities = fetch_activity_items()

    if not activities:
        print("No activities found.")
        return

    print("\n==============================")
    print("Latest Activities near your area")
    print("==============================\n")

    for i, (title, link, published) in enumerate(activities, 1):
        when = f" ({published})" if published else ""
        print(f"{i}. {title}{when}")
        print(f"   {link}\n")

    titles = [t for (t, _, _) in activities]

    print("\n==============================")
    print("Kids Activities Summary")
    print("==============================\n")
    summary = summarize_activity_titles(titles)
    print(summary)


if __name__ == "__main__":
    main()