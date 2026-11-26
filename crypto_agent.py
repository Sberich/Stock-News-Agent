import os
import json
from datetime import datetime, timezone
from google import genai
from google.genai import types
from pydantic import BaseModel

from news_utils import (
    load_sent_cache,
    save_sent_cache,
    format_article_links,
    fetch_rss_feeds,
    send_discord_embeds,
)

# ==========================================
# CONFIGURATION
# ==========================================
DISCORD_WEBHOOK_URL_CRYPTO = os.getenv("DISCORD_WEBHOOK_URL_CRYPTO")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CACHE_FILE = os.path.join(os.path.dirname(__file__), "sent_crypto_cache.json")

if not DISCORD_WEBHOOK_URL_CRYPTO or not GEMINI_API_KEY:
    print("Warning: Missing API Keys. Make sure DISCORD_WEBHOOK_URL_CRYPTO and GEMINI_API_KEY are set.")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.5-flash"


class CuratedNewsItem(BaseModel):
    title: str
    analysis: str
    sentiment: str
    original_index: int


# Crypto RSS Feeds to monitor
RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
}


def curate_and_summarize_news(news_list: list) -> list:
    """Uses Gemini to pick top crypto news and summarize them in Thai."""
    news_text = ""
    for i, news in enumerate(news_list):
        news_text += f"[{i + 1}] Source: {news['source']} | Title: {news['title']} | Summary: {news['summary']}\n"

    prompt = f"""
You are a world-class Cryptocurrency and Blockchain Analyst.
Below is a list of recent crypto news headlines and short summaries.
Your task:
1. Select 6 to 10 MOST IMPORTANT news items that have the biggest impact on the Crypto market (Bitcoin, Ethereum, major Altcoins, regulation, ETF flows, macro liquidity, tech upgrades).
2. For each selected news, TRANSLATE the title into THAI language and provide a suitable crypto-related emoji at the beginning.
3. Provide a DETAILED and COMPREHENSIVE analysis translated into THAI language. Do not make it too short.
4. Format the 'analysis' field to be engaging and easy to read using double newline characters (\\n\\n) between sections, and Markdown bold (**text**) for key metrics/tickers. Include:
   - 📖 เนื้อหาข่าว (Detailed summary of what happened - at least 3-4 sentences)
   - 🎯 ประเด็นสำคัญ (Key takeaway)
   - 💡 ผลกระทบต่อตลาดคริปโต (Impact on crypto market/prices and ETF sentiment)
5. Determine the sentiment of the news (POSITIVE, NEGATIVE, or NEUTRAL) for the 'sentiment' field.

Here is the news list:
{news_text}
"""
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[CuratedNewsItem],
                temperature=0.4,
            ),
        )
        return json.loads(response.text.strip())
    except Exception as e:
        print(f"Error in AI curation: {e}")
        return []


def build_discord_embeds(curated_news: list, original_news_list: list) -> list:
    """Builds Discord rich embeds with 1-click Thai translation links."""
    colors = {
        "POSITIVE": 5763719,   # Green
        "NEGATIVE": 15548997,  # Red
        "NEUTRAL": 3447003,    # Blue
    }

    embeds = []
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    for item in curated_news:
        orig_idx = int(item.get("original_index", 1)) - 1
        if 0 <= orig_idx < len(original_news_list):
            original_link = original_news_list[orig_idx]["link"]
            source_name = original_news_list[orig_idx]["source"]
        else:
            original_link = "https://cointelegraph.com/"
            source_name = "Unknown"

        sentiment = item.get("sentiment", "NEUTRAL").upper()
        embed_color = colors.get(sentiment, 3447003)

        analysis_text = item.get("analysis", "ไม่มีเนื้อหา").replace("\\n", "\n")
        analysis_text = analysis_text.replace("🎯", "\n🎯").replace("💡", "\n💡").replace("\n\n\n", "\n\n")
        link_line = format_article_links(source_name, original_link)

        embeds.append({
            "title": item.get("title", "ไม่มีหัวข้อ")[:250],
            "description": (analysis_text + f"\n\n{link_line}")[:4000],
            "color": embed_color,
            "footer": {
                "text": "🤖 วิเคราะห์โดย Gemini AI (Crypto Agent)"
            },
            "timestamp": now_iso,
        })

    return embeds


def main():
    if not DISCORD_WEBHOOK_URL_CRYPTO or not GEMINI_API_KEY:
        print("Skipping Crypto Agent: missing DISCORD_WEBHOOK_URL_CRYPTO or GEMINI_API_KEY.")
        return

    print("1. Loading cache & fetching Crypto news...")
    sent_cache = load_sent_cache(CACHE_FILE)
    print(f"   Sent crypto cache has {len(sent_cache)} items.")

    raw_news = fetch_rss_feeds(RSS_FEEDS, sent_cache=sent_cache, max_per_feed=25)
    print(f"   Found {len(raw_news)} fresh crypto articles.")

    if not raw_news:
        print("No new unseen crypto articles since last run.")
        return

    print("2. Sending to AI for curation and summarization...")
    curated_news = curate_and_summarize_news(raw_news)

    if not curated_news:
        print("Failed to curate crypto news.")
        return

    print("3. Sending Crypto news to Discord...")
    embeds = build_discord_embeds(curated_news, raw_news)
    header = "🪙 **อัปเดตข่าวคริปโต (Crypto News) น่าจับตามองประจำรอบ!** 🚀"
    if send_discord_embeds(DISCORD_WEBHOOK_URL_CRYPTO, header, embeds, chunk_size=2):
        for item in curated_news:
            orig_idx = int(item.get("original_index", 1)) - 1
            if 0 <= orig_idx < len(raw_news):
                sent_cache.add(raw_news[orig_idx]["link"])
        save_sent_cache(CACHE_FILE, sent_cache)
        print("4. Updated sent crypto cache successfully.")


if __name__ == "__main__":
    main()
