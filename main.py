import os
import sys
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
    create_gemini_client,
    generate_gemini_with_fallback,
)

# ==========================================
# CONFIGURATION
# ==========================================
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()

CACHE_FILE = os.path.join(os.path.dirname(__file__), "sent_news_cache.json")

if not DISCORD_WEBHOOK_URL or not GEMINI_API_KEY:
    raise ValueError("Missing API Keys! Please set DISCORD_WEBHOOK_URL and GEMINI_API_KEY as environment variables.")

# Configure Gemini Client (v1beta required for JSON schema mode)
client = create_gemini_client(GEMINI_API_KEY)


class CuratedNewsItem(BaseModel):
    title: str
    analysis: str
    sentiment: str
    original_index: int


# Verified Real-Time Stock & Macro RSS Feeds
RSS_FEEDS = {
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "CNBC Markets": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "Yahoo Finance": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,NVDA,AAPL,MSFT,GOOGL,AMZN,META,TSLA&region=US&lang=en-US",
    "Investing.com": "https://www.investing.com/rss/news_25.rss",
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_monetary.xml",
}


def curate_and_summarize_news(news_list: list) -> list:
    """Uses Gemini to pick top macro & stock-moving news and summarize them in Thai."""
    news_text = ""
    for i, news in enumerate(news_list):
        news_text += (
            f"[{i + 1}] Source: {news['source']} | Published: {news['published']} | "
            f"Title: {news['title']} | Summary: {news['summary']}\n"
        )

    prompt = f"""
You are a world-class US Stock Market and Macro Strategist.
Below is a list of recent financial news headlines from MarketWatch, CNBC Markets, Yahoo Finance, Investing.com, and the Federal Reserve.
Your task:
1. Select 8 to 12 MOST IMPORTANT news items that reveal the overall US market direction (Fed policy, tariffs, inflation, bond yields, macro shifts) OR major catalysts moving specific key stocks (e.g., NVDA, AAPL, MSFT, TSLA, AMZN, META, GOOGL).
2. For each selected news, TRANSLATE the title into clear, engaging THAI language and add a suitable emoji at the beginning of the 'title' field.
3. Provide a DETAILED and COMPREHENSIVE analysis in THAI language.
4. Format the 'analysis' field clearly using double newline characters (\\n\\n) between each section, and Markdown bold (**text**) for key metrics, tickers, or market direction. Include:
   - 📖 เนื้อหาข่าว (Detailed summary of what happened - at least 3-4 sentences)
   - 🎯 ประเด็นสำคัญ (Key takeaway / catalyst)
   - 💡 ผลกระทบต่อทิศทางตลาดและนักลงทุน (How this affects overall US market direction, sector rotation, or the specific stock)
5. Determine the sentiment of the news (POSITIVE, NEGATIVE, or NEUTRAL) for the 'sentiment' field.

Here is the news list:
{news_text}
"""
    result_text, used_model = generate_gemini_with_fallback(
        client,
        prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[CuratedNewsItem],
            temperature=0.4,
        ),
    )
    print(f"   Curated using model: {used_model}")
    return json.loads(result_text)


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
            original_link = "https://finance.yahoo.com/"
            source_name = "Financial News"

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
                "text": "🤖 วิเคราะห์โดย Gemini AI • AlphaZone Intelligence"
            },
            "timestamp": now_iso,
        })

    return embeds


def main():
    print("1. Loading cache & fetching fresh stock/macro news...")
    sent_cache = load_sent_cache(CACHE_FILE)
    print(f"   Sent history cache has {len(sent_cache)} items.")

    raw_news = fetch_rss_feeds(RSS_FEEDS, sent_cache=sent_cache, max_per_feed=20)
    print(f"   Found {len(raw_news)} fresh articles.")

    if not raw_news:
        print("No new unseen articles since last run.")
        return

    print("2. Sending to AI for curation and market-direction analysis...")
    curated_news = curate_and_summarize_news(raw_news)

    if not curated_news:
        print("Failed to curate news. Check API key and quota.")
        return

    utc_hour = datetime.now(timezone.utc).hour
    if 10 <= utc_hour <= 18:
        session_header = "✨ **[Pre-Market] อัปเดตข่าวสำคัญก่อนตลาดสหรัฐฯ เปิดคืนนี้!** 🇺🇸"
    else:
        session_header = "🌅 **[Morning Recap] สรุปข่าวเด่นและทิศทางตลาดสหรัฐฯ ยามเช้า!** 🇺🇸"

    print("3. Sending curated news to Discord...")
    embeds = build_discord_embeds(curated_news, raw_news)
    if send_discord_embeds(DISCORD_WEBHOOK_URL, session_header, embeds, chunk_size=2):
        for item in curated_news:
            orig_idx = int(item.get("original_index", 1)) - 1
            if 0 <= orig_idx < len(raw_news):
                sent_cache.add(raw_news[orig_idx]["link"])
        save_sent_cache(CACHE_FILE, sent_cache)
        print("4. Updated sent news cache successfully.")


if __name__ == "__main__":
    main()
