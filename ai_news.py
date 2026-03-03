from dotenv import load_dotenv
load_dotenv()

import feedparser
from openai import OpenAI

# ---- AI News Sources (Google News RSS search queries) ----
FEEDS = [
    "https://news.google.com/rss/search?q=artificial+intelligence&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=OpenAI&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=machine+learning&hl=en-US&gl=US&ceid=US:en",
]

MAX_ARTICLES = 10

client = OpenAI()

# ---- Fetch news headlines ----
def fetch_ai_news():
    items = []

    for url in FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            published = getattr(entry, "published", "").strip()

            if title and link:
                items.append((title, link, published))

    # remove duplicates
    seen = set()
    unique = []
    for t, l, p in items:
        if l in seen:
            continue
        seen.add(l)
        unique.append((t, l, p))

    return unique[:MAX_ARTICLES]


# ---- Summarize headlines ----
def summarize_titles(titles):
    prompt = (
        "Summarize the following AI-related news headlines into 5 short bullet points.\n\n"
        + "\n".join([f"- {t}" for t in titles])
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You summarize news trends clearly."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


# ---- Main flow ----
def main():
    news = fetch_ai_news()

    if not news:
        print("No AI news found.")
        return

    print("\n==============================")
    print("📰 Latest AI News")
    print("==============================\n")

    for i, (title, link, published) in enumerate(news, 1):
        when = f" ({published})" if published else ""
        print(f"{i}. {title}{when}")
        print(f"   {link}\n")

    titles = [t for (t, _, _) in news]

    print("\n==============================")
    print("🤖 AI News Summary")
    print("==============================\n")
    summary = summarize_titles(titles)
    print(summary)

def get_ai_news(limit: int = 10, include_summary: bool = True):
    news = fetch_ai_news()[:limit]  # list of (title, link, published)
    titles = [t for (t, _, _) in news]
    summary = summarize_titles(titles) if include_summary else ""
    articles = []
    for title, link, published in news:
        articles.append(
            {
                "title": title,
                "url": link,
                "published": published,
            }
        )
    return {"articles": articles, "summary": summary}
def chat_ai(question: str) -> str:
    if not question.strip():
        return "Please enter a question."

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()
if __name__ == "__main__":
    main()