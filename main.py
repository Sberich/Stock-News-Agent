import os
import feedparser
import requests
from google import genai
from google.genai import types
from pydantic import BaseModel
from datetime import datetime, timezone
import json
import re
from urllib.parse import urlparse

# ==========================================
# CONFIGURATION
# ==========================================
# Use environment variables for security (especially for GitHub Actions)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CACHE_FILE = os.path.join(os.path.dirname(__file__), "sent_news_cache.json")

if not DISCORD_WEBHOOK_URL or not GEMINI_API_KEY:
    raise ValueError("Missing API Keys! Please set DISCORD_WEBHOOK_URL and GEMINI_API_KEY as environment variables.")

# Configure Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = 'gemini-2.5-flash'

def load_sent_cache():
    """Loads previously sent article URLs to prevent duplicate news alerts."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data) if isinstance(data, list) else set()
        except Exception as e:
            print(f"Notice: Cache load skipped ({e})")
    return set()

def save_sent_cache(sent_set):
    """Saves sent article URLs, keeping the most recent 200 items."""
    try:
        items = list(sent_set)[-200:]
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Notice: Cache save skipped ({e})")

def to_google_translate_url(url: str) -> str:
    """Creates a Google Translate web proxy link for instant 1-click Thai translation."""
    try:
        parsed = urlparse(url)
        if not parsed.hostname or "translate.goog" in parsed.hostname:
            return url
        # Google Translate domain conversion: '-' -> '--', '.' -> '-'
        clean_host = parsed.hostname.lower().replace("-", "--").replace(".", "-")
        trans_host = f"{clean_host}.translate.goog"
        params = "_x_tr_sl=auto&_x_tr_tl=th&_x_tr_hl=th&_x_tr_pto=wapp"
        query_str = f"{parsed.query}&{params}" if parsed.query else params
        return f"https://{trans_host}{parsed.path}?{query_str}"
    except Exception:
        return url

class CuratedNewsItem(BaseModel):
    title: str
    analysis: str
    sentiment: str
    original_index: int

# RSS Feeds to monitor
RSS_FEEDS = {
    "Yahoo Finance": "https://finance.yahoo.com/news/rssurl",
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"
}

def fetch_rss_headlines(sent_cache=None):
    """Fetches top headlines from defined RSS feeds, filtering out previously sent news."""
    if sent_cache is None:
        sent_cache = set()
    all_news = []
    seen_links = set()

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            # Take top 25 from each source to give AI enough context
            for entry in feed.entries[:25]:
                link = entry.get("link", "").strip()
                if not link or link in seen_links or link in sent_cache:
                    continue
                seen_links.add(link)

                # Clean up summary HTML tags if any
                raw_summary = entry.get("summary", "")
                clean_summary = re.sub(r'<[^>]+>', '', raw_summary)[:400]
                
                all_news.append({
                    "source": source,
                    "title": entry.title,
                    "link": link,
                    "published": entry.get("published", ""),
                    "summary": clean_summary
                })
        except Exception as e:
            print(f"Error fetching from {source}: {e}")
            
    # Fallback: if all articles were in cache (e.g. slow weekend), allow top articles
    if not all_news and sent_cache:
        print("All latest items were in cache, fetching fresh top items as fallback...")
        return fetch_rss_headlines(sent_cache=set())

    return all_news

def curate_and_summarize_news(news_list):
    """Uses Gemini to pick top news and summarize them in Thai."""
    news_text = ""
    for i, news in enumerate(news_list):
        news_text += f"[{i+1}] Source: {news['source']} | Title: {news['title']} | Summary: {news['summary']}\n"

    prompt = f"""
You are a world-class US Stock Market Analyst.
Below is a list of recent financial news headlines and short summaries.
Your task:
1. Select 8 to 12 MOST IMPORTANT news items that have the biggest impact on the US macro economy, investment trends, or major specific tech stocks (like AAPL, TSLA, NVDA). If there are major breaking news, you can select up to 15 items.
2. For each selected news, TRANSLATE the title into THAI language.
3. Provide a DETAILED and COMPREHENSIVE analysis translated into THAI language. Do not make it too short.
4. Format the 'analysis' field to be engaging and easy to read using bullet points. VERY IMPORTANT: You MUST use double newline characters (\\n\\n) to create a clear blank line between each section for breathing space. Also, use Markdown bold (**text**) to highlight key terms, numbers, or market directions to make it easy to skim. Include:
   - 📖 เนื้อหาข่าว (Detailed summary of what happened - at least 3-4 sentences)
   - 🎯 ประเด็นสำคัญ (Key takeaway)
   - 💡 ผลกระทบต่อนักลงทุน (Detailed impact on markets and investors)
5. Determine the sentiment of the news (POSITIVE, NEGATIVE, or NEUTRAL) for the 'sentiment' field.
6. Provide a suitable emoji at the beginning of the 'title' field.

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
                temperature=0.4
            )
        )
        result_text = response.text.strip()
        curated_news = json.loads(result_text)
        return curated_news
    except Exception as e:
        print(f"Error in AI curation: {e}")
        return []

def send_to_discord(curated_news, original_news_list):
    """Sends the formatted news to Discord Webhook using Embeds in chunks of 2."""
    import time
    if not curated_news:
        print("No news to send.")
        return

    # Color mapping based on sentiment
    colors = {
        "POSITIVE": 5763719,  # Green
        "NEGATIVE": 15548997, # Red
        "NEUTRAL": 3447003    # Blue
    }

    embeds = []
    # Get current time for footer in strict ISO8601 for Discord
    now_iso = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
    for item in curated_news:
        orig_idx = int(item.get("original_index", 1)) - 1
        if 0 <= orig_idx < len(original_news_list):
            original_link = original_news_list[orig_idx]["link"]
            source_name = original_news_list[orig_idx]["source"]
        else:
            original_link = "https://finance.yahoo.com/"
            source_name = "Unknown"

        sentiment = item.get("sentiment", "NEUTRAL").upper()
        embed_color = colors.get(sentiment, 3447003)

        # Fix literal \n outputted by LLM
        analysis_text = item.get("analysis", "ไม่มีเนื้อหา").replace("\\n", "\n")

        # Generate 1-click Google Translate proxy URL
        translate_link = to_google_translate_url(original_link)
        link_line = f"📰 **[อ่านข่าวต้นฉบับ ({source_name})]({original_link})**"
        if translate_link and translate_link != original_link:
            link_line += f" · 🌐 **[อ่านแบบแปลไทย]({translate_link})**"

        embed = {
            "title": item.get("title", "ไม่มีหัวข้อ")[:250],
            "description": (analysis_text + f"\n\n{link_line}")[:4000],
            "color": embed_color,
            "footer": {
                "text": "🤖 วิเคราะห์โดย Gemini AI • AlphaZone Intelligence"
            },
            "timestamp": now_iso
        }
        embeds.append(embed)

    # Determine session title based on execution time (Pre-Market vs Morning Recap)
    utc_hour = datetime.now(timezone.utc).hour
    if 10 <= utc_hour <= 18:
        session_header = "✨ **[Pre-Market] อัปเดตข่าวสำคัญก่อนตลาดสหรัฐฯ เปิดคืนนี้!** 🇺🇸"
    else:
        session_header = "🌅 **[Morning Recap] สรุปข่าวเด่นและผลกระทบตลาดสหรัฐฯ ยามเช้า!** 🇺🇸"

    # Discord allows max 10 embeds per message, and 6000 chars total.
    # To prevent 500 Internal Server Errors from payload size or rate limits, we use chunk_size = 2.
    chunk_size = 2
    for i in range(0, len(embeds), chunk_size):
        chunk = embeds[i:i + chunk_size]
        payload = {
            "content": session_header if i == 0 else "",
            "embeds": chunk
        }

        try:
            response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
            response.raise_for_status()
            print(f"Successfully sent chunk {i//chunk_size + 1} to Discord!")
            time.sleep(2) # Add delay to avoid Discord rate limits
        except Exception as e:
            print(f"Error sending chunk to Discord: {e}")

def main():
    print("1. Loading cache & fetching fresh news...")
    sent_cache = load_sent_cache()
    print(f"   Sent history cache has {len(sent_cache)} items.")

    raw_news = fetch_rss_headlines(sent_cache)
    print(f"   Found {len(raw_news)} fresh articles.")
    
    if not raw_news:
        print("No new articles to process.")
        return

    print("2. Sending to AI for curation and summarization...")
    curated_news = curate_and_summarize_news(raw_news)
    
    if curated_news:
        print("3. AI Processing complete. Sending to Discord...")
        send_to_discord(curated_news, raw_news)
        
        # Save newly sent articles to cache to prevent duplicate alerts
        for item in curated_news:
            orig_idx = int(item.get("original_index", 1)) - 1
            if 0 <= orig_idx < len(raw_news):
                sent_cache.add(raw_news[orig_idx]["link"])
        save_sent_cache(sent_cache)
        print("4. Updated sent news cache successfully.")
    else:
        print("Failed to curate news. Check API key and quota.")

if __name__ == "__main__":
    main()
